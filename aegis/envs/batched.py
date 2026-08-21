"""Vectorised AEGIS environment: many independent episodes stepped together.

:class:`~aegis.envs.flutter_marl.FlutterSuppressionEnv` is the API-conformant
reference implementation. This is the training workhorse: identical semantics,
but the plant is integrated for every environment at once with batched einsums,
which is what makes a full experiment suite affordable.

Two implementations of the same physics is a real risk of silent drift, so
``tests/test_batched_env.py`` asserts that this class and the PettingZoo one
produce bit-comparable trajectories from the same seed. If they ever diverge,
that test fails rather than the results quietly changing.

Environments run at *different* airspeeds, so the state matrix has a leading
batch axis and the integrator uses ``einsum`` rather than a single matmul.
Episodes auto-reset on termination, which is what on-policy RL expects.
"""

from __future__ import annotations

import numpy as np

from aegis.envs.flutter_marl import (
    _LOCAL_OBS_SIZE,
    EnvConfig,
    NormalizationScales,
    _span_neighbors,
)
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.sensors import SensorArray


class BatchedFlutterEnv:
    """A fixed-size batch of independent flutter-suppression episodes."""

    def __init__(
        self,
        config: EnvConfig,
        n_envs: int = 64,
        seed: int | None = None,
        auto_reset: bool = True,
    ):
        if n_envs < 1:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        self.config = config
        # Training wants auto-reset; evaluation must not, or episodes would be
        # replaced mid-comparison and controllers would face different work.
        self.auto_reset = auto_reset
        self.n_envs = n_envs
        self.model = AeroelasticModel(config.wing, config.density)
        self.scales = NormalizationScales.derive(self.model, config.reference_tip_plunge)
        self.flutter_speed = self.model.flutter_point().airspeed

        self.agent_names = tuple(s.name for s in config.wing.surfaces)
        self.n_agents = len(self.agent_names)
        self.n_modes = self.model.n_modes
        self.n_plant = self.model.n_states
        self._neighbors = _span_neighbors(self.n_agents)
        self.obs_dim = _LOCAL_OBS_SIZE + self._message_size()

        sensors = SensorArray(self.model)
        self._sensor_plunge = np.stack([s.plunge_row for s in sensors.stations])
        self._sensor_twist = np.stack([s.twist_row for s in sensors.stations])
        self._forward_offset = np.asarray([s.forward_offset for s in sensors.stations])
        self._aft_offset = np.asarray([s.aft_offset for s in sensors.stations])

        self._tau = np.asarray([s.actuator_tau for s in config.wing.surfaces])
        self._travel = np.asarray([s.max_deflection for s in config.wing.surfaces])
        self._max_rate = np.asarray([s.max_rate for s in config.wing.surfaces])

        self.control_dt = config.control_dt
        self.substeps = config.substeps
        self._dt = config.control_dt / config.substeps
        self._max_steps = round(config.episode_duration / config.control_dt)

        self.rng = np.random.default_rng(seed)
        self._allocate()
        self.reset()

    # -------------------------------------------------------------- allocation
    def _message_size(self) -> int:
        if self.config.comm_mode == "none":
            return 0
        if self.config.comm_mode == "neighbor_local":
            return 2 * _LOCAL_OBS_SIZE
        return 2

    def _allocate(self) -> None:
        n, k = self.n_envs, self.n_agents
        self._plant = np.zeros((n, self.n_plant))
        self._deflection = np.zeros((n, k))
        self._jam_mask = np.zeros((n, k), dtype=bool)
        self._jam_value = np.zeros((n, k))
        self._airspeed = np.zeros(n)
        self._step_count = np.zeros(n, dtype=int)
        self._a_matrix = np.zeros((n, self.n_plant, self.n_plant))
        self._control_matrix = np.zeros((n, self.n_plant, k))
        self._gust_matrix = np.zeros((n, self.n_plant))
        self._force_matrix = np.zeros((n, self.n_modes, k))
        self._gust_samples: np.ndarray | None = None
        self._episode_return = np.zeros(n)
        self._active = np.ones(n, dtype=bool)
        self._finished = np.zeros(n, dtype=bool)
        self._previous_energy = np.zeros(n)

    # ------------------------------------------------------------------ reset
    def reset(self) -> np.ndarray:
        """Reset every environment and return observations ``(n_envs, n_agents, obs)``."""
        self._reset_subset(np.arange(self.n_envs))
        self._active[:] = True
        self._finished[:] = False
        return self._observations()

    def reset_to(
        self,
        speeds: np.ndarray,
        tip_plunge: np.ndarray,
        tip_twist: np.ndarray | None = None,
        jam_surface: np.ndarray | None = None,
        jam_angle: np.ndarray | None = None,
        gust_seed: int | None = None,
    ) -> np.ndarray:
        """Reset to explicitly specified conditions, one per environment.

        Evaluation has to put every controller on identical episodes -- same
        airspeed, same initial deformation, same turbulence, same failure. Drawing
        them randomly per controller would confound the comparison with luck, so
        the conditions are passed in rather than sampled.

        ``jam_surface`` holds an agent index per environment, or -1 for none.
        """
        speeds = np.asarray(speeds, dtype=float)
        if speeds.shape != (self.n_envs,):
            raise ValueError(f"expected {self.n_envs} speeds, got {speeds.shape}")
        indices = np.arange(self.n_envs)

        self._airspeed[:] = speeds
        a, control, gust, force = self.model.batch_state_space(speeds)
        self._a_matrix[:] = a
        self._control_matrix[:] = control
        self._gust_matrix[:] = gust
        self._force_matrix[:] = force

        self._plant[:] = 0.0
        self._plant[:, 0] = np.asarray(tip_plunge, dtype=float)
        if tip_twist is not None:
            self._plant[:, self.config.wing.n_bending] = np.asarray(tip_twist, dtype=float)

        self._jam_mask[:] = False
        self._jam_value[:] = 0.0
        if jam_surface is not None:
            surfaces = np.asarray(jam_surface, dtype=int)
            angles = (
                np.asarray(jam_angle, dtype=float)
                if jam_angle is not None
                else np.zeros(self.n_envs)
            )
            valid = surfaces >= 0
            rows = indices[valid]
            columns = surfaces[valid]
            held = np.clip(angles[valid], -self._travel[columns], self._travel[columns])
            self._jam_mask[rows, columns] = True
            self._jam_value[rows, columns] = held

        self._deflection[:] = np.where(self._jam_mask, self._jam_value, 0.0)
        self._step_count[:] = 0
        self._episode_return[:] = 0.0
        self._active[:] = True
        self._finished[:] = False
        self._previous_energy[:] = self._structural_energy()

        if gust_seed is not None:
            self.rng = np.random.default_rng(gust_seed)
        self._draw_gust(indices)
        return self._observations()

    def _reset_subset(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        count = indices.size
        low, high = self.config.speed_ratio_range
        speeds = self.flutter_speed * self.rng.uniform(low, high, size=count)
        self._airspeed[indices] = speeds

        a, control, gust, force = self.model.batch_state_space(speeds)
        self._a_matrix[indices] = a
        self._control_matrix[indices] = control
        self._gust_matrix[indices] = gust
        self._force_matrix[indices] = force

        self._plant[indices] = 0.0
        plunge = self.rng.uniform(*self.config.tip_plunge_range, size=count)
        plunge = plunge * self.rng.choice([-1.0, 1.0], size=count)
        twist = self.rng.uniform(*self.config.tip_twist_range, size=count)
        self._plant[indices, 0] = plunge
        self._plant[indices, self.config.wing.n_bending] = twist

        self._jam_mask[indices] = False
        self._jam_value[indices] = 0.0
        if self.config.jam_probability > 0.0:
            jammed = self.rng.random(count) < self.config.jam_probability
            victims = self.rng.integers(0, self.n_agents, size=count)
            angles = self.rng.uniform(*self.config.jam_angle_range, size=count)
            rows = indices[jammed]
            columns = victims[jammed]
            held = np.clip(angles[jammed], -self._travel[columns], self._travel[columns])
            self._jam_mask[rows, columns] = True
            self._jam_value[rows, columns] = held

        self._deflection[indices] = np.where(
            self._jam_mask[indices], self._jam_value[indices], 0.0
        )
        self._step_count[indices] = 0
        self._episode_return[indices] = 0.0
        self._draw_gust(indices)
        self._previous_energy[indices] = self._structural_energy()[indices]

    def _draw_gust(self, indices: np.ndarray) -> None:
        """Pre-sample turbulence for the affected episodes."""
        n_samples = self._max_steps * self.substeps + 2
        if self._gust_samples is None:
            self._gust_samples = np.zeros((self.n_envs, n_samples))
        if self.config.turbulence_intensity <= 0.0:
            self._gust_samples[indices] = 0.0
            return

        from aegis.gusts import DrydenTurbulence

        for row in indices:
            turbulence = DrydenTurbulence(
                intensity=self.config.turbulence_intensity,
                airspeed=float(self._airspeed[row]),
                duration=n_samples * self._dt,
                sample_rate=1.0 / self._dt,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            samples = turbulence._samples[:n_samples]
            self._gust_samples[row, : samples.size] = samples

    # ------------------------------------------------------------------- step
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Advance one control interval.

        ``actions`` has shape ``(n_envs, n_agents)`` in [-1, 1]. Returns
        ``(observations, rewards, dones, info)`` with rewards shaped
        ``(n_envs, n_agents)``.
        """
        actions = np.asarray(actions, dtype=float).reshape(self.n_envs, self.n_agents)
        command = np.clip(actions, -1.0, 1.0) * self._travel
        previous = self._deflection.copy()
        frozen_plant = self._plant.copy()
        frozen_deflection = self._deflection.copy()
        frozen_steps = self._step_count.copy()

        for substep in range(self.substeps):
            offset = self._step_count * self.substeps + substep
            self._rk4(command, offset)

        self._step_count += 1
        rewards = self._rewards(previous)
        diverged = self._diverged()
        truncated = self._step_count >= self._max_steps
        dones = diverged | truncated
        self._episode_return += rewards.mean(axis=1)

        info = {
            "diverged": diverged.copy(),
            "truncated": truncated.copy(),
            "speed_ratio": self._airspeed / self.flutter_speed,
            "energy": self._structural_energy(),
            "episode_return": self._episode_return.copy(),
            "jammed_any": self._jam_mask.any(axis=1),
            "finished": self._finished.copy(),
        }
        if self.auto_reset:
            if np.any(dones):
                self._reset_subset(np.nonzero(dones)[0])
        else:
            # Freeze finished episodes so their final state and verdict persist
            # while the rest of the batch runs on.
            self._finished |= dones
            self._active = ~self._finished
            self._plant = np.where(self._active[:, None], self._plant, frozen_plant)
            self._deflection = np.where(
                self._active[:, None], self._deflection, frozen_deflection
            )
            self._step_count = np.where(self._active, self._step_count, frozen_steps)
        return self._observations(), rewards, dones, info

    def _rk4(self, command: np.ndarray, sample_offset: np.ndarray) -> None:
        gust = self._gust_at(sample_offset)
        plant, deflection = self._plant, self._deflection

        k1p, k1d = self._derivative(plant, deflection, command, gust)
        k2p, k2d = self._derivative(
            plant + 0.5 * self._dt * k1p, deflection + 0.5 * self._dt * k1d, command, gust
        )
        k3p, k3d = self._derivative(
            plant + 0.5 * self._dt * k2p, deflection + 0.5 * self._dt * k2d, command, gust
        )
        k4p, k4d = self._derivative(
            plant + self._dt * k3p, deflection + self._dt * k3d, command, gust
        )
        self._plant = plant + (self._dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        self._deflection = deflection + (self._dt / 6.0) * (
            k1d + 2.0 * k2d + 2.0 * k3d + k4d
        )

    def _derivative(
        self,
        plant: np.ndarray,
        deflection: np.ndarray,
        command: np.ndarray,
        gust: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        plant_rate = (
            np.einsum("nij,nj->ni", self._a_matrix, plant)
            + np.einsum("nik,nk->ni", self._control_matrix, deflection)
            + self._gust_matrix * gust[:, None]
        )
        return plant_rate, self._actuator_rate(deflection, command)

    def _actuator_rate(self, deflection: np.ndarray, command: np.ndarray) -> np.ndarray:
        target = np.clip(command, -self._travel, self._travel)
        rate = np.clip((target - deflection) / self._tau, -self._max_rate, self._max_rate)
        at_upper = (deflection >= self._travel) & (rate > 0.0)
        at_lower = (deflection <= -self._travel) & (rate < 0.0)
        return np.where(at_upper | at_lower | self._jam_mask, 0.0, rate)

    def _gust_at(self, offset: np.ndarray) -> np.ndarray:
        assert self._gust_samples is not None
        clipped = np.clip(offset, 0, self._gust_samples.shape[1] - 1)
        return self._gust_samples[np.arange(self.n_envs), clipped]

    # ----------------------------------------------------------- observations
    def _modal_acceleration(self) -> np.ndarray:
        rate = (
            np.einsum("nij,nj->ni", self._a_matrix, self._plant)
            + np.einsum("nik,nk->ni", self._control_matrix, self._deflection)
        )
        return rate[:, self.n_modes : 2 * self.n_modes]

    def _observations(self) -> np.ndarray:
        position = self._plant[:, : self.n_modes]
        rate = self._plant[:, self.n_modes : 2 * self.n_modes]
        acceleration = self._modal_acceleration()

        plunge_accel = acceleration @ self._sensor_plunge.T
        twist_accel = acceleration @ self._sensor_twist.T
        forward = plunge_accel - self._forward_offset * twist_accel
        aft = plunge_accel + self._aft_offset * twist_accel

        local = np.stack(
            [
                forward / self.scales.acceleration,
                aft / self.scales.acceleration,
                (rate @ self._sensor_plunge.T) / self.scales.velocity,
                (position @ self._sensor_twist.T) / self.scales.twist,
                (rate @ self._sensor_twist.T) / self.scales.twist_rate,
                self._deflection / self._travel,
                np.tile(
                    (self._airspeed / self.flutter_speed - 1.0)[:, None],
                    (1, self.n_agents),
                ),
            ],
            axis=-1,
        )
        return np.concatenate([local, self._messages(local)], axis=-1).astype(np.float32)

    def _messages(self, local: np.ndarray) -> np.ndarray:
        mode = self.config.comm_mode
        if mode == "none":
            return np.zeros((self.n_envs, self.n_agents, 0))
        if mode == "neighbor_local":
            messages = np.zeros((self.n_envs, self.n_agents, 2 * _LOCAL_OBS_SIZE))
            for i, (left, right) in enumerate(self._neighbors):
                if left is not None:
                    messages[:, i, :_LOCAL_OBS_SIZE] = local[:, left]
                if right is not None:
                    messages[:, i, _LOCAL_OBS_SIZE:] = local[:, right]
            return messages

        amplitude = self._plant[:, 0] / self.config.reference_tip_plunge
        rate = self._plant[:, self.n_modes] / self.scales.velocity
        oracle = np.stack([amplitude, rate], axis=-1)
        return np.tile(oracle[:, None, :], (1, self.n_agents, 1))

    # --------------------------------------------------------------- rewards
    def _structural_energy(self) -> np.ndarray:
        position = self._plant[:, : self.n_modes]
        rate = self._plant[:, self.n_modes : 2 * self.n_modes]
        return 0.5 * np.einsum("ni,ij,nj->n", rate, self.model.mass, rate) + 0.5 * np.einsum(
            "ni,ij,nj->n", position, self.model.stiffness, position
        )

    def control_power(self) -> np.ndarray:
        """Per-agent instantaneous control power, ``(n_envs, n_agents)``."""
        rate = self._plant[:, self.n_modes : 2 * self.n_modes]
        return np.einsum("ni,nik->nk", rate, self._force_matrix) * self._deflection

    def _rewards(self, previous: np.ndarray) -> np.ndarray:
        energy = self._structural_energy()
        floor = self.config.energy_floor_fraction * self.scales.energy
        bounded = energy + floor
        shared = -(energy / self.scales.energy)[:, None]

        # Credit is the agent's contribution to the LOGARITHMIC energy decay
        # rate, not to the raw energy rate. Dividing the exact energy-rate
        # decomposition by E leaves it exactly additive across agents, because E
        # is a shared scalar:
        #
        #     d(log E)/dt = [ -qdot^T C qdot + qdot^T L x_wake + sum_k P_k ] / E
        #
        # so agent k's term is still P_k / E and Proposition 1 still applies --
        # now to the quantity the evaluation actually measures.
        #
        # This matters because the unnormalised credit -P_k is maximised by
        # having energy available to extract, which rewards *sustaining* the
        # oscillation. Policies trained on it learned to keep the wing ringing
        # instead of damping it, scoring worse than a hand-tuned local damper.
        physics = -self.control_power() / bounded[:, None]

        # Potential-based shaping, also on log energy so that it telescopes to
        # the evaluated decay rate and shares the credit signal's units.
        potential_now = -np.log(bounded / self.scales.energy)
        potential_before = -np.log(
            (self._previous_energy + floor) / self.scales.energy
        )
        shaping = (
            self.config.shaping_gamma * potential_now - potential_before
        )[:, None] / self.control_dt
        self._previous_energy = energy

        effort = (self._deflection / self._travel) ** 2
        travel = (
            np.abs(self._deflection - previous) / (self._max_rate * self.control_dt)
        ) ** 2
        penalty = self.config.effort_penalty * effort + self.config.rate_penalty * travel

        mode = self.config.reward_mode
        if mode == "shared":
            base = np.broadcast_to(shared, physics.shape).copy()
        elif mode == "physics":
            base = physics
        else:
            base = physics + self.config.shaping_weight * shaping

        total = base - penalty
        total[self._diverged()] -= self.config.divergence_penalty
        return total

    def _diverged(self) -> np.ndarray:
        tip = self._plant[:, : self.n_modes] @ self.model.tip_rows()[0]
        return ~np.isfinite(tip) | (np.abs(tip) > self.config.divergence_limit)

    # ----------------------------------------------------------------- extras
    @property
    def airspeed(self) -> np.ndarray:
        return self._airspeed

    @property
    def deflection(self) -> np.ndarray:
        return self._deflection

    @property
    def plant_state(self) -> np.ndarray:
        return self._plant
