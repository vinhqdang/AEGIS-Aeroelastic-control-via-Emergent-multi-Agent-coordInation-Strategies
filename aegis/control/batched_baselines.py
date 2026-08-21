"""Classical baselines that run on the batched environment.

The baseline ladder matters more than the proposed method here, because a weak
baseline is how a paper gets desk-rejected. Four rungs, in order of how hard
they are to beat and how implementable they are:

1. :class:`BatchedLocalFeedback` -- collocated rate feedback, no communication.
   The honest floor for a distributed system.
2. :class:`BatchedLQR` -- fixed gain, **full 56-state plant including the wake
   lag states**. Not implementable on an aircraft (nobody measures shed
   vorticity), so this is a ceiling, not a competitor.
3. :class:`BatchedScheduledLQR` -- the same, but the gain is scheduled on
   airspeed. This is the obvious answer to "just retune for the flight
   condition", and any claim about off-design failure has to survive it.
4. :class:`BatchedLQG` -- gain-scheduled LQR driven by a steady-state Kalman
   observer that sees **only the same local accelerometers the agents get**.
   This is the implementable classical controller, and it is the one AEGIS has
   to actually beat.

Rung 4 exists because comparing a learned policy that uses local accelerometers
against an LQR handed the exact modal state and the wake states would be a
rigged comparison in the *other* direction -- and a reviewer would say so
immediately.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are

from aegis.physics.aeroelastic import AeroelasticModel
from aegis.sensors import SensorArray

_GRID_POINTS = 9


def _augmented_plant(
    model: AeroelasticModel, airspeed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Plant augmented with first-order actuator lag: ``z = [plant, deflection]``."""
    plant = model.state_space(airspeed)
    n_plant = model.n_states
    n_surfaces = model.wing.n_surfaces
    taus = np.asarray([s.actuator_tau for s in model.wing.surfaces])

    state = np.zeros((n_plant + n_surfaces, n_plant + n_surfaces))
    state[:n_plant, :n_plant] = plant.a_matrix
    state[:n_plant, n_plant:] = plant.control_matrix
    state[n_plant:, n_plant:] = -np.diag(1.0 / taus)

    control = np.zeros((n_plant + n_surfaces, n_surfaces))
    control[n_plant:, :] = np.diag(1.0 / taus)
    return state, control


def _discrete_plant(
    model: AeroelasticModel, airspeed: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation of the augmented plant.

    These controllers run at a fixed 200 Hz on a plant whose fastest states (the
    wake lag) are orders of magnitude faster than that sample interval. Designing
    in continuous time and then integrating the observer with a forward Euler
    step at the control interval is unstable -- ``dt * |lambda|`` reaches ~60 --
    so the design is done in discrete time and the observer update is exact.
    """
    state, control = _augmented_plant(model, airspeed)
    n, m = state.shape[0], control.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n] = state
    block[:n, n:] = control
    transition = expm(block * dt)
    return transition[:n, :n], transition[:n, n:]


def _cost_matrices(
    model: AeroelasticModel, effort_weight: float, rate_weight: float
) -> tuple[np.ndarray, np.ndarray]:
    """State and input cost: physical structural energy plus a deflection penalty."""
    n_plant, n_modes = model.n_states, model.n_modes
    n_surfaces = model.wing.n_surfaces
    limits = np.asarray([s.max_deflection for s in model.wing.surfaces])

    state_cost = np.zeros((n_plant + n_surfaces, n_plant + n_surfaces))
    state_cost[:n_modes, :n_modes] = model.stiffness
    state_cost[n_modes : 2 * n_modes, n_modes : 2 * n_modes] = model.mass
    state_cost[n_plant:, n_plant:] = np.diag(rate_weight / limits**2)
    return state_cost, np.diag(effort_weight / limits**2)


def _discrete_lqr_gain(
    model: AeroelasticModel,
    airspeed: float,
    dt: float,
    effort_weight: float,
    rate_weight: float,
) -> np.ndarray:
    """Discrete-time LQR gain for the sampled-data loop."""
    transition, input_matrix = _discrete_plant(model, airspeed, dt)
    state_cost, input_cost = _cost_matrices(model, effort_weight, rate_weight)
    riccati = solve_discrete_are(transition, input_matrix, state_cost, input_cost)
    return np.linalg.solve(
        input_cost + input_matrix.T @ riccati @ input_matrix,
        input_matrix.T @ riccati @ transition,
    )


def _discrete_observer_gain(
    model: AeroelasticModel,
    airspeed: float,
    dt: float,
    measurement: np.ndarray,
    process_noise: float,
    measurement_noise: float,
) -> np.ndarray:
    """Steady-state discrete Kalman predictor gain.

    The measurement rows have norms of order 1e4-1e5 because accelerometers
    respond to the fast states, so the measurement noise has to be scaled to
    match or the filter demands impossible precision and the gain explodes.
    """
    transition, _ = _discrete_plant(model, airspeed, dt)
    dimension = transition.shape[0]
    noise_w = process_noise * np.eye(dimension)
    row_scale = np.linalg.norm(measurement, axis=1) ** 2
    noise_v = measurement_noise * np.diag(row_scale)

    covariance = solve_discrete_are(transition.T, measurement.T, noise_w, noise_v)
    innovation = measurement @ covariance @ measurement.T + noise_v
    return transition @ covariance @ measurement.T @ np.linalg.inv(innovation)


def _lqr_gain(
    model: AeroelasticModel,
    airspeed: float,
    effort_weight: float,
    rate_weight: float,
) -> np.ndarray:
    state, control = _augmented_plant(model, airspeed)
    n_plant, n_modes = model.n_states, model.n_modes
    limits = np.asarray([s.max_deflection for s in model.wing.surfaces])

    state_cost = np.zeros_like(state)
    state_cost[:n_modes, :n_modes] = model.stiffness
    state_cost[n_modes : 2 * n_modes, n_modes : 2 * n_modes] = model.mass
    state_cost[n_plant:, n_plant:] = np.diag(rate_weight / limits**2)
    input_cost = np.diag(effort_weight / limits**2)

    riccati = solve_continuous_are(state, control, state_cost, input_cost)
    return np.linalg.solve(input_cost, control.T @ riccati)


def _measurement_model(
    model: AeroelasticModel, airspeed: float
) -> np.ndarray:
    """Rows mapping the augmented state to the local accelerometer readings.

    Accelerometers measure modal acceleration, which is an algebraic function of
    the augmented state, so the measurement matrix is built from the plant rows
    rather than being a simple selection.
    """
    plant = model.state_space(airspeed)
    n_plant, n_modes = model.n_states, model.n_modes
    acceleration_rows = np.hstack(
        [plant.a_matrix[n_modes : 2 * n_modes, :], plant.control_matrix[n_modes : 2 * n_modes, :]]
    )

    sensors = SensorArray(model)
    rows = []
    for station in sensors.stations:
        plunge = station.plunge_row @ acceleration_rows
        twist = station.twist_row @ acceleration_rows
        rows.append(plunge - station.forward_offset * twist)
        rows.append(plunge + station.aft_offset * twist)
    del n_plant
    return np.vstack(rows)


class BatchedLocalFeedback:
    """Collocated rate feedback from the agent's own observation vector."""

    name = "local_feedback"

    def __init__(self, plunge_gain: float = 0.35, twist_gain: float = 0.9):
        self.plunge_gain = plunge_gain
        self.twist_gain = twist_gain

    def reset(self, n_envs: int) -> None:
        return None

    def __call__(self, env, observation: np.ndarray) -> np.ndarray:
        # Channels 2 and 4 are the NORMALISED local plunge rate and twist rate
        # (see BatchedFlutterEnv._observations). These gains are defined against
        # physical units, so the normalisation has to be undone first -- feeding
        # normalised values straight in silently rescales the controller.
        plunge_rate = observation[..., 2] * env.scales.velocity
        twist_rate = observation[..., 4] * env.scales.twist_rate
        command = -(self.plunge_gain * plunge_rate + self.twist_gain * twist_rate)
        return np.clip(command, -1.0, 1.0)


class BatchedLQR:
    """Fixed-gain full-state LQR. A ceiling, not an implementable controller."""

    name = "lqr_fixed"

    def __init__(
        self,
        model: AeroelasticModel,
        design_speed: float,
        control_dt: float,
        effort_weight: float = 2.0e4,
        rate_weight: float = 1.0e2,
    ):
        self.model = model
        # Discrete-time design, matching the sampled-data loop it actually runs
        # in. Designing in continuous time and applying at 200 Hz measurably
        # weakens it, and a handicapped baseline is worthless.
        self.gain = _discrete_lqr_gain(
            model, design_speed, control_dt, effort_weight, rate_weight
        )
        self._travel = np.asarray([s.max_deflection for s in model.wing.surfaces])

    def reset(self, n_envs: int) -> None:
        return None

    def __call__(self, env, observation: np.ndarray) -> np.ndarray:
        augmented = np.concatenate([env.plant_state, env.deflection], axis=-1)
        command = -augmented @ self.gain.T
        return np.clip(command / self._travel, -1.0, 1.0)


class BatchedScheduledLQR:
    """Full-state LQR with the gain scheduled on airspeed.

    Gains are precomputed on a speed grid and interpolated. This is the answer to
    "just retune per flight condition", so the off-design claim must survive it.
    """

    name = "lqr_scheduled"

    def __init__(
        self,
        model: AeroelasticModel,
        speed_range: tuple[float, float],
        control_dt: float,
        effort_weight: float = 2.0e4,
        rate_weight: float = 1.0e2,
    ):
        self.model = model
        self.speeds = np.linspace(speed_range[0], speed_range[1], _GRID_POINTS)
        self.gains = np.stack(
            [
                _discrete_lqr_gain(model, u, control_dt, effort_weight, rate_weight)
                for u in self.speeds
            ]
        )
        self._travel = np.asarray([s.max_deflection for s in model.wing.surfaces])

    def reset(self, n_envs: int) -> None:
        return None

    def _interpolate(self, airspeed: np.ndarray) -> np.ndarray:
        position = np.interp(airspeed, self.speeds, np.arange(self.speeds.size))
        lower = np.clip(np.floor(position).astype(int), 0, self.speeds.size - 1)
        upper = np.clip(lower + 1, 0, self.speeds.size - 1)
        blend = (position - lower)[:, None, None]
        return (1.0 - blend) * self.gains[lower] + blend * self.gains[upper]

    def __call__(self, env, observation: np.ndarray) -> np.ndarray:
        gains = self._interpolate(env.airspeed)
        augmented = np.concatenate([env.plant_state, env.deflection], axis=-1)
        command = -np.einsum("nkj,nj->nk", gains, augmented)
        return np.clip(command / self._travel, -1.0, 1.0)


class BatchedLQG:
    """Scheduled LQR on a Kalman observer fed only by the local accelerometers.

    This is the implementable classical controller and the one that matters: it
    sees exactly what the agents see -- six accelerometer channels, no modal
    state, no wake states. Both the regulator and the observer are designed in
    discrete time at the control rate, and the observer update is the exact
    zero-order-hold recursion, because the plant carries states far faster than
    the 200 Hz sample interval.
    """

    name = "lqg_local"

    def __init__(
        self,
        model: AeroelasticModel,
        speed_range: tuple[float, float],
        control_dt: float,
        effort_weight: float = 2.0e4,
        rate_weight: float = 1.0e2,
        process_noise: float = 1.0e-4,
        measurement_noise: float = 1.0e-6,
    ):
        self.model = model
        self.n_plant = model.n_states
        self.n_surfaces = model.wing.n_surfaces
        self.dimension = self.n_plant + self.n_surfaces
        self.control_dt = control_dt
        self.speeds = np.linspace(speed_range[0], speed_range[1], _GRID_POINTS)

        gains, transitions, inputs, measurements, observers = [], [], [], [], []
        for speed in self.speeds:
            transition, input_matrix = _discrete_plant(model, speed, control_dt)
            measurement = _measurement_model(model, speed)
            gains.append(
                _discrete_lqr_gain(model, speed, control_dt, effort_weight, rate_weight)
            )
            observers.append(
                _discrete_observer_gain(
                    model, speed, control_dt, measurement, process_noise, measurement_noise
                )
            )
            transitions.append(transition)
            inputs.append(input_matrix)
            measurements.append(measurement)

        self.gains = np.stack(gains)
        self.transitions = np.stack(transitions)
        self.inputs = np.stack(inputs)
        self.measurements = np.stack(measurements)
        self.observer_gains = np.stack(observers)

        self._travel = np.asarray([s.max_deflection for s in model.wing.surfaces])
        self._estimate: np.ndarray | None = None

    def reset(self, n_envs: int) -> None:
        self._estimate = np.zeros((n_envs, self.dimension))

    def _blend(self, airspeed: np.ndarray, stack: np.ndarray) -> np.ndarray:
        position = np.interp(airspeed, self.speeds, np.arange(self.speeds.size))
        lower = np.clip(np.floor(position).astype(int), 0, self.speeds.size - 1)
        upper = np.clip(lower + 1, 0, self.speeds.size - 1)
        weight = (position - lower).reshape(-1, *([1] * (stack.ndim - 1)))
        return (1.0 - weight) * stack[lower] + weight * stack[upper]

    def __call__(self, env, observation: np.ndarray) -> np.ndarray:
        if self._estimate is None or self._estimate.shape[0] != env.n_envs:
            self.reset(env.n_envs)
        assert self._estimate is not None

        # Forward/aft readings interleaved per station to line up with the rows
        # built by _measurement_model, and de-normalised because the observer is
        # designed in physical units. It gets nothing the policy cannot see.
        scale = env.scales.acceleration
        n_stations = observation.shape[1]
        measured = np.empty((env.n_envs, 2 * n_stations))
        measured[:, 0::2] = observation[..., 0] * scale
        measured[:, 1::2] = observation[..., 1] * scale

        transition = self._blend(env.airspeed, self.transitions)
        input_matrix = self._blend(env.airspeed, self.inputs)
        measurement = self._blend(env.airspeed, self.measurements)
        observer_gain = self._blend(env.airspeed, self.observer_gains)
        gains = self._blend(env.airspeed, self.gains)

        command = -np.einsum("nkj,nj->nk", gains, self._estimate)
        command = np.clip(command, -self._travel, self._travel)

        innovation = measured - np.einsum("nij,nj->ni", measurement, self._estimate)
        self._estimate = (
            np.einsum("nij,nj->ni", transition, self._estimate)
            + np.einsum("nik,nk->ni", input_matrix, command)
            + np.einsum("nik,nk->ni", observer_gain, innovation)
        )
        # A diverging plant drives the estimate to infinity; keep it finite so the
        # episode still reports a divergence rather than a NaN.
        self._estimate = np.clip(np.nan_to_num(self._estimate), -1e8, 1e8)
        return np.clip(command / self._travel, -1.0, 1.0)
