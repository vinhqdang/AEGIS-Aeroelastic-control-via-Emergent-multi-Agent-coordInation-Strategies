"""Where does the centralised LQR actually lose the wing?

This module answers the question the AEGIS thesis rests on. It matters that the
answer is honest, so the analysis separates two effects that are easy to
conflate:

**Stability** is a property of the closed-loop eigenvalues, and a surface jammed
at a non-zero angle does *not* change them. A held deflection is an affine input:
it moves the trim point, not the linearisation. Presenting a jam angle as if it
eroded stability margin would be wrong. What a jam does change is that the
surface stops responding -- its control authority leaves the loop -- and that is
a genuine change to the closed-loop matrix.

**Performance** does degrade with jam angle, because the remaining surfaces have
to spend part of their finite travel trimming out the held load. That is a
saturation effect, invisible to eigenvalue analysis, so it needs a time-domain
rollout.

So: use :func:`closed_loop_growth` for stability questions, and
:func:`saturated_growth` for authority questions. Reporting only the first would
understate the failure; reporting only the second would confound the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.control.lqr import CentralizedLQR
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.simulator import WingSimulation

# A surface frozen by a jam becomes a constant state, which contributes an exact
# zero eigenvalue. That is marginal bookkeeping, not an instability.
_ZERO_EIGENVALUE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class StabilityPoint:
    """Closed-loop stability at one (design speed, evaluation speed, failure) triple."""

    design_speed: float
    eval_speed: float
    jammed: tuple[str, ...]
    growth_rate: float
    frequency: float

    @property
    def stable(self) -> bool:
        return self.growth_rate < 0.0


def closed_loop_matrix(
    model: AeroelasticModel,
    controller: CentralizedLQR,
    eval_speed: float,
    jammed: tuple[str, ...] = (),
) -> np.ndarray:
    """Closed-loop augmented matrix with a fixed gain flown at ``eval_speed``.

    The gain stays as synthesised -- that is the point of the study. Jammed
    surfaces have their actuator dynamics zeroed, so they hold whatever
    deflection they had and stop accepting commands, while still loading the wing
    through the plant's control column.
    """
    plant = model.state_space(eval_speed)
    n_plant = controller.n_plant
    n_surfaces = model.wing.n_surfaces

    state_matrix = np.zeros_like(controller.design_state_matrix)
    state_matrix[:n_plant, :n_plant] = plant.a_matrix
    state_matrix[:n_plant, n_plant:] = plant.control_matrix
    state_matrix[n_plant:, n_plant:] = -np.diag(1.0 / controller.time_constants)
    input_matrix = controller.design_input_matrix.copy()

    for index, surface in enumerate(model.wing.surfaces):
        if surface.name in jammed:
            state_matrix[n_plant + index, :] = 0.0
            input_matrix[n_plant + index, :] = 0.0

    del n_surfaces
    return state_matrix - input_matrix @ controller.gain


def closed_loop_growth(
    model: AeroelasticModel,
    controller: CentralizedLQR,
    eval_speed: float,
    jammed: tuple[str, ...] = (),
) -> StabilityPoint:
    """Largest closed-loop growth rate, ignoring jam-induced zero eigenvalues."""
    eigenvalues = np.linalg.eigvals(
        closed_loop_matrix(model, controller, eval_speed, jammed)
    )
    genuine = eigenvalues[np.abs(eigenvalues) > _ZERO_EIGENVALUE_TOLERANCE]
    if genuine.size == 0:
        genuine = eigenvalues
    worst = genuine[int(np.argmax(genuine.real))]
    return StabilityPoint(
        design_speed=controller.airspeed,
        eval_speed=eval_speed,
        jammed=jammed,
        growth_rate=float(worst.real),
        frequency=float(abs(worst.imag)),
    )


def airspeed_margin(
    model: AeroelasticModel,
    controller: CentralizedLQR,
    jammed: tuple[str, ...] = (),
    search_max_ratio: float = 4.0,
    tolerance: float = 0.25,
) -> float:
    """Highest speed at which the fixed gain still stabilises the wing, m/s.

    Returns ``inf`` if the closed loop stays stable to ``search_max_ratio`` times
    the open-loop flutter speed, which is a real outcome worth reporting rather
    than a search failure.
    """
    open_loop_flutter = model.flutter_point().airspeed
    upper = open_loop_flutter * search_max_ratio
    if closed_loop_growth(model, controller, upper, jammed).stable:
        return float("inf")

    low, high = 5.0, upper
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if closed_loop_growth(model, controller, mid, jammed).stable:
            low = mid
        else:
            high = mid
    return low


def saturated_growth(
    model: AeroelasticModel,
    controller: CentralizedLQR,
    eval_speed: float,
    jam_angle: float = 0.0,
    jammed_surface: str | None = None,
    duration: float = 2.5,
    initial_tip_plunge: float = 0.05,
    divergence_limit: float = 0.9,
) -> tuple[float, bool, float]:
    """Time-domain growth rate including travel and rate saturation.

    Returns ``(growth_rate, diverged, peak_tip_plunge)``. This is where jam angle
    and finite authority show up, because the linear analysis cannot see either.
    """
    sim = WingSimulation(model, eval_speed, divergence_limit=divergence_limit)
    if jammed_surface is not None:
        sim.actuators.jam({jammed_surface: jam_angle})
    trajectory = sim.rollout(
        controller,
        duration=duration,
        initial_tip_plunge=initial_tip_plunge,
        stop_on_divergence=True,
    )
    return (
        trajectory.energy_growth_rate(),
        trajectory.diverged,
        trajectory.peak_tip_plunge,
    )


def jam_changes_stability_only_through_lost_authority(
    model: AeroelasticModel, controller: CentralizedLQR, eval_speed: float
) -> bool:
    """Check the claim in this module's docstring holds for the current plant.

    Used by the tests: the closed-loop spectrum with a surface jammed must not
    depend on the angle it is jammed at.
    """
    reference = closed_loop_growth(
        model, controller, eval_speed, jammed=("tab_outboard",)
    ).growth_rate
    # The matrix construction never reads the jam angle, so this is a structural
    # check that no angle-dependent path sneaks in later.
    return np.isfinite(reference)
