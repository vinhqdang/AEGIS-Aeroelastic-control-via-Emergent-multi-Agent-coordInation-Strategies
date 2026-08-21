"""PettingZoo environment: one agent per control surface.

Design commitments, and why each one is there:

**Observations are genuinely local.** Each agent sees its own accelerometer pair
and local motion, plus air data (which really is broadcast on an aircraft). It
does *not* see the modal state. The coupled bending-torsion flutter mode is a
global quantity no single station can measure, and that partial observability is
the problem AEGIS exists to study -- so it is enforced here rather than assumed
away.

**Communication is a configured ablation axis, not a fixed design.** Three modes
span the range the paper needs: nothing, neighbours' raw local observations, and
an oracle that hands over the true critical-mode amplitude and rate. The oracle
is not a proposed method -- it is the upper bound that says how much the learned
phasor encoder could possibly buy.

**Reward mode is likewise an ablation axis.** ``shared`` is the naive baseline;
``physics`` is the closed-form difference reward of Proposition 1; ``blended``
is the exact-plus-residual form MoCCA actually proposes.

**Scales are derived, not guessed.** Observations and rewards are normalised by
reference values built from the flutter frequency and a reference tip amplitude,
so the numbers an agent sees stay O(1) across flight conditions instead of
drifting by orders of magnitude with dynamic pressure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from aegis.gusts import DrydenTurbulence, GustModel, NoGust
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING, WingProperties
from aegis.sensors import SensorArray
from aegis.simulator import WingSimulation

COMM_MODES = ("none", "neighbor_local", "modal_oracle")
REWARD_MODES = ("shared", "physics", "blended")

_LOCAL_OBS_SIZE = 7  # see FlutterSuppressionEnv._local_observation


@dataclass
class EnvConfig:
    """Everything that defines an AEGIS training task."""

    wing: WingProperties = GOLAND_WING
    density: float = 1.225

    # Flight envelope, as multiples of the open-loop flutter speed. Training
    # across a range is the whole point: a fixed-gain LQR cannot do it.
    speed_ratio_range: tuple[float, float] = (1.00, 1.60)

    episode_duration: float = 2.0
    control_dt: float = 0.005
    substeps: int = 5

    # Initial excitation, drawn per episode.
    tip_plunge_range: tuple[float, float] = (0.02, 0.06)
    tip_twist_range: tuple[float, float] = (-0.01, 0.01)

    turbulence_intensity: float = 0.0  # 0 disables turbulence
    divergence_limit: float = 0.9

    # Failure injection. Probability that some surface jams, and the angle range
    # it jams at, in radians.
    jam_probability: float = 0.0
    jam_angle_range: tuple[float, float] = (-0.21, 0.21)

    comm_mode: str = "none"
    reward_mode: str = "physics"
    physics_weight: float = 0.7  # blend factor for reward_mode="blended"

    effort_penalty: float = 0.02
    rate_penalty: float = 0.01
    divergence_penalty: float = 20.0

    reference_tip_plunge: float = 0.05  # sets the observation and reward scales

    def __post_init__(self) -> None:
        if self.comm_mode not in COMM_MODES:
            raise ValueError(f"comm_mode must be one of {COMM_MODES}, got {self.comm_mode}")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, got {self.reward_mode}"
            )
        low, high = self.speed_ratio_range
        if not 0 < low <= high:
            raise ValueError(f"invalid speed_ratio_range {self.speed_ratio_range}")
        if not 0.0 <= self.jam_probability <= 1.0:
            raise ValueError("jam_probability must lie in [0, 1]")
        if not 0.0 <= self.physics_weight <= 1.0:
            raise ValueError("physics_weight must lie in [0, 1]")


@dataclass(frozen=True)
class NormalizationScales:
    """Reference magnitudes used to keep observations and rewards O(1).

    Built from the flutter frequency and a reference tip amplitude rather than
    hand-tuned, so they stay meaningful if the wing or flight envelope changes.
    """

    acceleration: float
    velocity: float
    twist: float
    twist_rate: float
    energy: float
    power: float

    @classmethod
    def derive(
        cls, model: AeroelasticModel, reference_tip_plunge: float
    ) -> NormalizationScales:
        omega = model.flutter_point().frequency
        if not np.isfinite(omega) or omega <= 0.0:
            omega = float(model.structural_frequencies[0])
        amplitude = reference_tip_plunge
        twist = amplitude / model.wing.semispan  # a comparable rotation scale

        plunge_row, _ = model.tip_rows()
        modal = np.zeros(model.n_modes)
        modal[0] = amplitude
        energy = float(0.5 * modal @ model.stiffness @ modal)
        del plunge_row

        return cls(
            acceleration=max(omega**2 * amplitude, 1e-6),
            velocity=max(omega * amplitude, 1e-6),
            twist=max(twist, 1e-6),
            twist_rate=max(omega * twist, 1e-6),
            energy=max(energy, 1e-9),
            power=max(energy * omega, 1e-9),
        )


class FlutterSuppressionEnv(ParallelEnv):
    """Distributed flutter suppression, one agent per trailing-edge surface."""

    metadata: ClassVar[dict] = {
        "name": "aegis_flutter_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: EnvConfig | None = None):
        self.config = config if config is not None else EnvConfig()
        self.model = AeroelasticModel(self.config.wing, self.config.density)
        self.sensors = SensorArray(self.model)
        self.scales = NormalizationScales.derive(
            self.model, self.config.reference_tip_plunge
        )
        self.flutter_speed = self.model.flutter_point().airspeed

        self.possible_agents = [s.name for s in self.config.wing.surfaces]
        self.agents: list[str] = []
        self._agent_index = {name: i for i, name in enumerate(self.possible_agents)}
        self._neighbors = _span_neighbors(len(self.possible_agents))

        self._observation_size = _LOCAL_OBS_SIZE + self._message_size()
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._observation_size,), dtype=np.float32
        )
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.sim: WingSimulation | None = None
        self._rng = np.random.default_rng()
        self._max_steps = round(
            self.config.episode_duration / self.config.control_dt
        )

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str) -> spaces.Box:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._action_space

    def _message_size(self) -> int:
        if self.config.comm_mode == "none":
            return 0
        if self.config.comm_mode == "neighbor_local":
            return 2 * _LOCAL_OBS_SIZE  # two span neighbours, padded at the ends
        return 2  # modal oracle: critical-mode amplitude and rate

    # ----------------------------------------------------------------- rollout
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._diverged = False

        airspeed = self.flutter_speed * self._rng.uniform(*self.config.speed_ratio_range)
        self.sim = WingSimulation(
            self.model,
            airspeed,
            gust=self._make_gust(airspeed),
            control_dt=self.config.control_dt,
            substeps=self.config.substeps,
            divergence_limit=self.config.divergence_limit,
        )
        self._apply_random_jam()
        self.sim.reset(
            self._rng,
            initial_tip_plunge=self._rng.uniform(*self.config.tip_plunge_range)
            * self._rng.choice([-1.0, 1.0]),
            initial_tip_twist=self._rng.uniform(*self.config.tip_twist_range),
        )
        # Return real infos, not empty dicts: the flight condition and any
        # injected failure are drawn during reset, so a caller that only
        # reads step() infos would never learn what episode it is in.
        return self._observations(), self._infos()

    def step(self, actions: dict[str, np.ndarray]):
        if self.sim is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.agents:
            return {}, {}, {}, {}, {}

        command = np.zeros(len(self.possible_agents))
        for name, action in actions.items():
            command[self._agent_index[name]] = float(np.asarray(action).reshape(-1)[0])

        previous_deflection = self.sim.deflection.copy()
        self.sim.step(command)
        self._step_count += 1
        self._diverged = self.sim.diverged

        rewards = self._rewards(command, previous_deflection)
        truncated = self._step_count >= self._max_steps
        terminations = {agent: self._diverged for agent in self.agents}
        truncations = {agent: truncated and not self._diverged for agent in self.agents}
        infos = self._infos()
        observations = self._observations()

        if self._diverged or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------ observations
    def _observations(self) -> dict[str, np.ndarray]:
        if not self.possible_agents:
            return {}
        local = np.stack(
            [self._local_observation(i) for i in range(len(self.possible_agents))]
        )
        messages = self._messages(local)
        return {
            name: np.concatenate([local[i], messages[i]]).astype(np.float32)
            for i, name in enumerate(self.possible_agents)
            if name in self.agents or not self.agents
        }

    def _local_observation(self, index: int) -> np.ndarray:
        """What one agent can actually measure at its own station."""
        sim = self.sim
        assert sim is not None
        acceleration = self.sensors.read(sim.modal_acceleration)[index]
        local = self.sensors.local_states(sim.modal_position, sim.modal_rate)[index]
        plunge_rate, twist, twist_rate = local

        return np.asarray(
            [
                acceleration[0] / self.scales.acceleration,
                acceleration[1] / self.scales.acceleration,
                plunge_rate / self.scales.velocity,
                twist / self.scales.twist,
                twist_rate / self.scales.twist_rate,
                sim.actuators.saturation_fraction(sim.deflection)[index],
                # Air data is broadcast on a real aircraft, so sharing it costs
                # the distributed story nothing and lets one policy span the
                # envelope instead of memorising one condition.
                sim.airspeed / self.flutter_speed - 1.0,
            ]
        )

    def _messages(self, local: np.ndarray) -> np.ndarray:
        mode = self.config.comm_mode
        n_agents = local.shape[0]
        if mode == "none":
            return np.zeros((n_agents, 0))
        if mode == "neighbor_local":
            messages = np.zeros((n_agents, 2 * _LOCAL_OBS_SIZE))
            for i, (left, right) in enumerate(self._neighbors):
                if left is not None:
                    messages[i, :_LOCAL_OBS_SIZE] = local[left]
                if right is not None:
                    messages[i, _LOCAL_OBS_SIZE:] = local[right]
            return messages

        sim = self.sim
        assert sim is not None
        # Oracle upper bound: the true critical-mode amplitude and rate, which a
        # learned phasor encoder would have to estimate from local data alone.
        amplitude = sim.modal_position[0] / self.config.reference_tip_plunge
        rate = sim.modal_rate[0] / self.scales.velocity
        return np.tile(np.asarray([amplitude, rate]), (n_agents, 1))

    # ----------------------------------------------------------------- rewards
    def _rewards(
        self, command: np.ndarray, previous_deflection: np.ndarray
    ) -> dict[str, float]:
        sim = self.sim
        assert sim is not None

        shared = -sim.structural_energy / self.scales.energy
        # Negative control power means extracting energy, so it is the reward.
        physics = -sim.control_power() / self.scales.power

        effort = sim.actuators.saturation_fraction(sim.deflection) ** 2
        travel = (
            np.abs(sim.deflection - previous_deflection)
            / (sim.actuators.max_rate * sim.control_dt)
        ) ** 2
        penalty = self.config.effort_penalty * effort + self.config.rate_penalty * travel

        if self.config.reward_mode == "shared":
            base = np.full(len(self.possible_agents), shared)
        elif self.config.reward_mode == "physics":
            base = physics
        else:
            weight = self.config.physics_weight
            base = weight * physics + (1.0 - weight) * shared

        total = base - penalty
        if self._diverged:
            total = total - self.config.divergence_penalty

        del command
        return {
            name: float(total[self._agent_index[name]])
            for name in self.agents
        }

    def _infos(self) -> dict[str, dict]:
        sim = self.sim
        assert sim is not None
        shared_info = {
            "airspeed": sim.airspeed,
            "speed_ratio": sim.airspeed / self.flutter_speed,
            "tip_plunge": sim.tip_plunge,
            "structural_energy": sim.structural_energy,
            "diverged": self._diverged,
            "jammed": tuple(
                name
                for i, name in enumerate(self.possible_agents)
                if sim.actuators.jam_mask[i]
            ),
        }
        power = sim.control_power()
        return {
            name: {**shared_info, "control_power": float(power[self._agent_index[name]])}
            for name in self.agents
        }

    # ---------------------------------------------------------------- episode
    def _make_gust(self, airspeed: float) -> GustModel:
        if self.config.turbulence_intensity <= 0.0:
            return NoGust()
        return DrydenTurbulence(
            intensity=self.config.turbulence_intensity,
            airspeed=airspeed,
            duration=self.config.episode_duration + 0.5,
            sample_rate=1.0 / (self.config.control_dt / self.config.substeps),
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )

    def _apply_random_jam(self) -> None:
        assert self.sim is not None
        if self.config.jam_probability <= 0.0:
            return
        if self._rng.random() >= self.config.jam_probability:
            return
        victim = str(self._rng.choice(self.possible_agents))
        limit = self.config.wing.surfaces[self._agent_index[victim]].max_deflection
        angle = float(np.clip(self._rng.uniform(*self.config.jam_angle_range), -limit, limit))
        self.sim.actuators.jam({victim: angle})

    def render(self) -> None:
        """Rendering is offline: see aegis.viz.animation for rollout playback."""
        raise NotImplementedError(
            "AEGIS renders rollouts offline -- use aegis.viz.animation.render_animation"
        )

    def close(self) -> None:
        self.sim = None


def _span_neighbors(n_agents: int) -> list[tuple[int | None, int | None]]:
    """Inboard/outboard neighbour indices -- a line graph along the span.

    This is what a distributed avionics bus physically looks like: each unit
    talks to the ones next to it, not to everything at once.
    """
    return [
        (i - 1 if i > 0 else None, i + 1 if i < n_agents - 1 else None)
        for i in range(n_agents)
    ]
