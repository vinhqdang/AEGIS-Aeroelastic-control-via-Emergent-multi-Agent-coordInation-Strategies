"""Centralised full-state LQR -- the reference the distributed policy must match.

This is the baseline that decides whether AEGIS has a result. Making it weak
would be self-defeating, so the synthesis:

* includes the actuator lag in the design plant, rather than designing against
  an ideal actuator and paying the phase loss at run time;
* penalises the physical structural energy (``Q`` built from the modal mass and
  stiffness) rather than an arbitrary diagonal, so the cost means something;
* gets the full plant state, wake lag states included -- an advantage no
  distributed policy has.

Its weakness is exactly the one the thesis targets: it is synthesised for one
airspeed and one actuator set. Change either and it is no longer optimal, while
a learned distributed policy can be trained to span the range.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from aegis.physics.aeroelastic import AeroelasticModel
from aegis.simulator import WingSimulation


class CentralizedLQR:
    """Full-state LQR over the plant augmented with first-order actuator lag."""

    def __init__(
        self,
        model: AeroelasticModel,
        airspeed: float,
        effort_weight: float = 2.0e3,
        rate_weight: float = 1.0e2,
        failed_surfaces: tuple[str, ...] = (),
    ):
        if effort_weight <= 0 or rate_weight <= 0:
            raise ValueError("effort_weight and rate_weight must be positive")

        self.model = model
        self.airspeed = airspeed
        self.failed_surfaces = failed_surfaces
        plant = model.state_space(airspeed)

        n_plant = model.n_states
        n_surfaces = model.wing.n_surfaces
        taus = np.asarray([s.actuator_tau for s in model.wing.surfaces])
        limits = np.asarray([s.max_deflection for s in model.wing.surfaces])

        state_matrix = np.zeros((n_plant + n_surfaces, n_plant + n_surfaces))
        state_matrix[:n_plant, :n_plant] = plant.a_matrix
        state_matrix[:n_plant, n_plant:] = plant.control_matrix
        state_matrix[n_plant:, n_plant:] = -np.diag(1.0 / taus)

        input_matrix = np.zeros((n_plant + n_surfaces, n_surfaces))
        input_matrix[n_plant:, :] = np.diag(1.0 / taus)

        # Cost on structural energy, plus a light penalty on held deflection so
        # the solution does not park the surfaces at the stops.
        n_modes = model.n_modes
        state_cost = np.zeros_like(state_matrix)
        state_cost[:n_modes, :n_modes] = model.stiffness
        state_cost[n_modes : 2 * n_modes, n_modes : 2 * n_modes] = model.mass
        state_cost[n_plant:, n_plant:] = np.diag(rate_weight / limits**2)
        input_cost = np.diag(effort_weight / limits**2)

        riccati = solve_continuous_are(state_matrix, input_matrix, state_cost, input_cost)
        self.gain = np.linalg.solve(input_cost, input_matrix.T @ riccati)

        # Kept so the robustness analysis can rebuild the closed loop at a
        # different flight condition, or with a surface removed, without
        # re-deriving the augmentation.
        self.design_state_matrix = state_matrix
        self.design_input_matrix = input_matrix
        self.n_plant = n_plant
        self.time_constants = taus

        # A jammed surface contributes no command; the others must compensate.
        self._active = np.asarray(
            [s.name not in failed_surfaces for s in model.wing.surfaces], dtype=float
        )
        self._limits = limits

    def __call__(self, sim: WingSimulation) -> np.ndarray:
        """Normalized commands in [-1, 1] from the full augmented state."""
        augmented = np.concatenate([sim.plant_state, sim.deflection])
        command = -self.gain @ augmented
        return np.clip(command / self._limits, -1.0, 1.0) * self._active


class LocalRateFeedback:
    """Decentralised baseline: each surface reacts only to its own local motion.

    A collocated rate-feedback law -- the simplest thing a distributed system can
    do without communication, and the honest floor for the comparison. It damps
    plunge locally but has no knowledge of the coupled bending-torsion mode, so
    it is expected to fail exactly where coordination matters.
    """

    def __init__(
        self,
        model: AeroelasticModel,
        plunge_gain: float = 0.35,
        twist_gain: float = 0.9,
        failed_surfaces: tuple[str, ...] = (),
    ):
        from aegis.sensors import SensorArray

        self.sensors = SensorArray(model)
        self.plunge_gain = plunge_gain
        self.twist_gain = twist_gain
        self._active = np.asarray(
            [s.name not in failed_surfaces for s in model.wing.surfaces], dtype=float
        )

    def __call__(self, sim: WingSimulation) -> np.ndarray:
        local = self.sensors.local_states(sim.modal_position, sim.modal_rate)
        plunge_rate, _, twist_rate = local[:, 0], local[:, 1], local[:, 2]
        command = -(self.plunge_gain * plunge_rate + self.twist_gain * twist_rate)
        return np.clip(command, -1.0, 1.0) * self._active
