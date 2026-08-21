"""Time-domain closed-loop simulation of the wing plus its actuator bank.

The plant states and the actuator states are integrated together with a single
fixed-step RK4, sub-stepped inside each control interval. Integrating them
jointly matters: the servo lag sits right on top of the flutter frequency, so
staggering the two integrators smears exactly the phase relationship the
controller is trying to exploit.

Every rollout records the per-agent **control power** ``P_k = (xdot^T b_k) delta_k``.
That quantity is the closed-form difference reward of ``docs/ALGORITHM.md``
(Proposition 1), so it is a first-class output of the simulator rather than
something reconstructed later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from aegis.actuators import ActuatorBank
from aegis.gusts import GustModel, NoGust
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import WingProperties

DEFAULT_CONTROL_DT = 0.005  # 200 Hz control update
DEFAULT_SUBSTEPS = 5        # 1 kHz integration


class Controller(Protocol):
    """Anything that maps a simulation state to normalized commands in [-1, 1]."""

    def __call__(self, sim: WingSimulation) -> np.ndarray: ...


@dataclass
class Trajectory:
    """Recorded rollout. All arrays are indexed by control step."""

    time: list[float] = field(default_factory=list)
    modal_position: list[np.ndarray] = field(default_factory=list)
    modal_rate: list[np.ndarray] = field(default_factory=list)
    modal_acceleration: list[np.ndarray] = field(default_factory=list)
    deflection: list[np.ndarray] = field(default_factory=list)
    command: list[np.ndarray] = field(default_factory=list)
    gust: list[float] = field(default_factory=list)
    control_power: list[np.ndarray] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    tip_plunge: list[float] = field(default_factory=list)
    tip_twist: list[float] = field(default_factory=list)
    airspeed: float = 0.0
    diverged: bool = False

    def as_arrays(self) -> TrajectoryArrays:
        return TrajectoryArrays(
            time=np.asarray(self.time),
            modal_position=np.asarray(self.modal_position),
            modal_rate=np.asarray(self.modal_rate),
            modal_acceleration=np.asarray(self.modal_acceleration),
            deflection=np.asarray(self.deflection),
            command=np.asarray(self.command),
            gust=np.asarray(self.gust),
            control_power=np.asarray(self.control_power),
            energy=np.asarray(self.energy),
            tip_plunge=np.asarray(self.tip_plunge),
            tip_twist=np.asarray(self.tip_twist),
            airspeed=self.airspeed,
            diverged=self.diverged,
        )


@dataclass(frozen=True)
class TrajectoryArrays:
    """Immutable, array-backed view of a :class:`Trajectory`."""

    time: np.ndarray
    modal_position: np.ndarray
    modal_rate: np.ndarray
    modal_acceleration: np.ndarray
    deflection: np.ndarray
    command: np.ndarray
    gust: np.ndarray
    control_power: np.ndarray
    energy: np.ndarray
    tip_plunge: np.ndarray
    tip_twist: np.ndarray
    airspeed: float
    diverged: bool

    @property
    def n_steps(self) -> int:
        return self.time.size

    @property
    def peak_tip_plunge(self) -> float:
        return float(np.abs(self.tip_plunge).max())

    def energy_growth_rate(self) -> float:
        """Least-squares exponential growth rate of the structural energy, 1/s.

        Positive means the rollout was diverging. This is the single number that
        says whether a controller stabilised the wing.
        """
        positive = self.energy > 0.0
        if positive.sum() < 3:
            return float("nan")
        slope = np.polyfit(self.time[positive], np.log(self.energy[positive]), 1)[0]
        return float(0.5 * slope)  # energy ~ amplitude^2


class WingSimulation:
    """Closed-loop aeroelastic simulation at a fixed flight condition."""

    def __init__(
        self,
        model: AeroelasticModel,
        airspeed: float,
        gust: GustModel | None = None,
        control_dt: float = DEFAULT_CONTROL_DT,
        substeps: int = DEFAULT_SUBSTEPS,
        divergence_limit: float = 2.0,
    ):
        if control_dt <= 0 or substeps < 1:
            raise ValueError("control_dt must be positive and substeps at least 1")
        if divergence_limit <= 0:
            raise ValueError("divergence_limit must be positive")

        self.model = model
        self.airspeed = airspeed
        self.gust = gust if gust is not None else NoGust()
        self.control_dt = control_dt
        self.substeps = substeps
        self.divergence_limit = divergence_limit

        self.actuators = ActuatorBank(model.wing)
        self.plant = model.state_space(airspeed)
        self.control_influence = model.control_influence(airspeed)
        self.total_mass = model.total_mass(airspeed)
        self.total_damping = model.total_damping(airspeed)
        self.total_stiffness = model.total_stiffness(airspeed)
        self.wake_gain = model.wake_gain(airspeed)
        self.tip_plunge_row, self.tip_twist_row = model.tip_rows()

        self._n_plant = model.n_states
        self._n_modes = model.n_modes
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def wing(self) -> WingProperties:
        return self.model.wing

    @property
    def plant_state(self) -> np.ndarray:
        return self._state[: self._n_plant]

    @property
    def deflection(self) -> np.ndarray:
        return self._state[self._n_plant :]

    @property
    def modal_position(self) -> np.ndarray:
        return self._state[: self._n_modes]

    @property
    def modal_rate(self) -> np.ndarray:
        return self._state[self._n_modes : 2 * self._n_modes]

    @property
    def modal_acceleration(self) -> np.ndarray:
        """Current modal acceleration -- what the accelerometers see."""
        return self._derivative(self.time, self._state, self._last_command)[
            self._n_modes : 2 * self._n_modes
        ]

    @property
    def tip_plunge(self) -> float:
        """Tip vertical displacement, m, positive **down**."""
        return float(self.tip_plunge_row @ self.modal_position)

    @property
    def tip_twist(self) -> float:
        """Tip twist, rad, positive nose-up."""
        return float(self.tip_twist_row @ self.modal_position)

    @property
    def structural_energy(self) -> float:
        """Mechanical energy in the structure, J. Positive definite by construction."""
        q, qdot = self.modal_position, self.modal_rate
        return float(
            0.5 * qdot @ self.model.mass @ qdot + 0.5 * q @ self.model.stiffness @ q
        )

    @property
    def aeroelastic_energy(self) -> float:
        """Energy functional built from the equation-of-motion matrices, J.

        Unlike :attr:`structural_energy` this is *not* guaranteed positive above
        the divergence speed, but its time derivative is the one whose control
        term splits exactly per agent, so it is the quantity the credit-assignment
        argument is about.
        """
        q, qdot = self.modal_position, self.modal_rate
        return float(
            0.5 * qdot @ self.total_mass @ qdot + 0.5 * q @ self.total_stiffness @ q
        )

    def energy_rate(self) -> float:
        """Exact time derivative of :attr:`aeroelastic_energy`, W.

        Evaluated from the energy budget
        ``Edot = -qdot^T C qdot + qdot^T L x_wake + sum_k P_k`` rather than by
        differencing, so the split into dissipation, wake exchange and per-agent
        control power is exact and testable.
        """
        return float(
            -self.modal_rate @ self.total_damping @ self.modal_rate
            + self.modal_rate @ self.wake_gain @ self.wake_state
            + self.control_power().sum()
        )

    @property
    def wake_state(self) -> np.ndarray:
        """Wagner lag states, two per aerodynamic strip."""
        return self._state[2 * self._n_modes : self._n_plant]

    def control_power(self) -> np.ndarray:
        """Per-agent instantaneous control power, W. Negative means extracting.

        This is exactly the closed-form difference reward: agent ``k``'s term in
        the energy-rate budget is ``(xdot^T b_k) * delta_k`` and no other term
        depends on ``delta_k``.
        """
        return (self.modal_rate @ self.control_influence) * self.deflection

    # ------------------------------------------------------------------ rollout
    def reset(
        self,
        rng: np.random.Generator | None = None,
        initial_tip_plunge: float = 0.0,
        initial_tip_twist: float = 0.0,
    ) -> None:
        """Reset to rest, optionally with a plucked initial deformation.

        The mode shapes are normalised to unit tip value, so the requested tip
        plunge and twist are achieved by loading the first bending and first
        torsion modes directly.
        """
        generator = rng if rng is not None else np.random.default_rng()
        self.gust.reset(generator)

        self._state = np.zeros(self._n_plant + self.wing.n_surfaces)
        self._state[self._n_plant :] = self.actuators.initial_deflection()
        self._state[0] = initial_tip_plunge
        self._state[self.wing.n_bending] = initial_tip_twist
        self.time = 0.0
        self.step_index = 0
        self._last_command = np.zeros(self.wing.n_surfaces)

    def step(self, action: np.ndarray) -> None:
        """Advance one control interval with normalized actions in [-1, 1]."""
        action = np.atleast_1d(np.asarray(action, dtype=float))
        if action.shape != (self.wing.n_surfaces,):
            raise ValueError(
                f"expected {self.wing.n_surfaces} actions, got shape {action.shape}"
            )
        command = self.actuators.normalized_command(action)
        self._last_command = command

        dt = self.control_dt / self.substeps
        for _ in range(self.substeps):
            self._state = _rk4_step(self._derivative, self.time, self._state, dt, command)
            self.time += dt
        self.step_index += 1

    @property
    def diverged(self) -> bool:
        """True once the tip motion leaves the range the linear model can claim."""
        return not np.isfinite(self.tip_plunge) or abs(self.tip_plunge) > self.divergence_limit

    def rollout(
        self,
        controller: Controller | None,
        duration: float,
        rng: np.random.Generator | None = None,
        initial_tip_plunge: float = 0.0,
        initial_tip_twist: float = 0.0,
        stop_on_divergence: bool = True,
    ) -> TrajectoryArrays:
        """Simulate for ``duration`` seconds, recording everything.

        ``controller=None`` means open loop (all surfaces held at zero).
        """
        self.reset(rng, initial_tip_plunge, initial_tip_twist)
        trajectory = Trajectory(airspeed=self.airspeed)
        n_steps = round(duration / self.control_dt)
        zeros = np.zeros(self.wing.n_surfaces)

        for _ in range(n_steps):
            action = zeros if controller is None else np.asarray(controller(self))
            self._record(trajectory, action)
            self.step(action)
            if stop_on_divergence and self.diverged:
                trajectory.diverged = True
                break

        return trajectory.as_arrays()

    def _record(self, trajectory: Trajectory, action: np.ndarray) -> None:
        trajectory.time.append(self.time)
        trajectory.modal_position.append(self.modal_position.copy())
        trajectory.modal_rate.append(self.modal_rate.copy())
        trajectory.modal_acceleration.append(self.modal_acceleration.copy())
        trajectory.deflection.append(self.deflection.copy())
        trajectory.command.append(self.actuators.normalized_command(action))
        trajectory.gust.append(self.gust.velocity(self.time))
        trajectory.control_power.append(self.control_power())
        trajectory.energy.append(self.structural_energy)
        trajectory.tip_plunge.append(self.tip_plunge)
        trajectory.tip_twist.append(self.tip_twist)

    # -------------------------------------------------------------- internals
    def _derivative(
        self, time: float, state: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        plant = state[: self._n_plant]
        deflection = state[self._n_plant :]
        gust_velocity = self.gust.velocity(time)

        plant_rate = (
            self.plant.a_matrix @ plant
            + self.plant.control_matrix @ deflection
            + self.plant.gust_matrix[:, 0] * gust_velocity
        )
        actuator_rate = self.actuators.derivative(deflection, command)
        return np.concatenate([plant_rate, actuator_rate])


def _rk4_step(
    derivative: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    time: float,
    state: np.ndarray,
    dt: float,
    command: np.ndarray,
) -> np.ndarray:
    k1 = derivative(time, state, command)
    k2 = derivative(time + 0.5 * dt, state + 0.5 * dt * k1, command)
    k3 = derivative(time + 0.5 * dt, state + 0.5 * dt * k2, command)
    k4 = derivative(time + dt, state + dt * k3, command)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
