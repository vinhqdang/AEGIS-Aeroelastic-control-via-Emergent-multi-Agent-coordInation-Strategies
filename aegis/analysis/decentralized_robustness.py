"""Stability margins for the decentralised, observer-based baseline controllers.

``aegis.analysis.robustness`` answers this question for :class:`CentralizedLQR`,
a full-state, continuous-time design. The decentralised LQG rungs
(:class:`~aegis.control.batched_baselines.BatchedDecentralizedLQG` and its
loop-transfer-recovered variant) are discrete-time, observer-based, and
per-surface, so the closed loop is a different, larger linear system: the true
plant state plus one Kalman-filter state estimate *per agent*, each driven by
that agent's own two local accelerometers and commanding only that agent's own
surface.

The question this module answers is the one the review raised directly: an
unstructured decentralised LQG has no guaranteed robustness margin (Doyle 1978),
so how much control-authority error can the closed loop actually tolerate before
it goes unstable? :func:`control_authority_margin` reports exactly that, as a
multiplicative factor on the true control-influence matrix relative to what the
controller was designed against -- the same quantity the paper's own UVLM
cross-validation already expresses its 1.9x discrepancy in.

``scripts/robust_baseline_check.py`` uses this module to check whether loop
transfer recovery (Doyle & Stein 1979) can improve that margin, since LTR is
the standard remedy for observer-induced margin loss. It cannot, here: the
plant's own state-feedback loop (no observer at all) already sits at the same
margin the LQG achieves, so there is no observer-induced loss left to recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from aegis.control.batched_baselines import (
    _cost_matrices,
    _discrete_observer_gain,
    _discrete_plant,
    _measurement_model,
)
from aegis.physics.aeroelastic import AeroelasticModel

_ZERO_EIGENVALUE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DecentralizedDesign:
    """One agent's synthesised gain row and observer, frozen at a design speed."""

    gain_row: np.ndarray  # (dimension,) -- this agent's own command only
    observer_gain: np.ndarray  # (dimension, 2)
    measurement_rows: tuple[int, int]


def _synthesize_agent(
    model: AeroelasticModel,
    agent: int,
    design_speed: float,
    control_dt: float,
    effort_weight: float,
    rate_weight: float,
    process_noise: float,
    measurement_noise: float,
    ltr_rho: float = 0.0,
) -> DecentralizedDesign:
    """Synthesise one agent's decentralised gain and observer at a fixed speed.

    ``ltr_rho`` implements Kalman-filter loop transfer recovery: the filter's
    process-noise weighting is boosted by ``ltr_rho * B B^T`` (this agent's own
    reduced input column), which pushes the filter loop toward the full-state
    LQR loop shape as ``ltr_rho`` grows, at the cost of noise rejection. At
    ``ltr_rho = 0`` this reproduces :class:`BatchedDecentralizedLQG` exactly.
    """
    transition, input_matrix = _discrete_plant(model, design_speed, control_dt)
    reduced_input = np.zeros_like(input_matrix)
    reduced_input[:, agent] = input_matrix[:, agent]

    state_cost, input_cost = _cost_matrices(model, effort_weight, rate_weight)
    riccati = solve_discrete_are(transition, reduced_input, state_cost, input_cost)
    gain = np.linalg.solve(
        input_cost + reduced_input.T @ riccati @ reduced_input,
        reduced_input.T @ riccati @ transition,
    )

    full_measurement = _measurement_model(model, design_speed)
    rows = (2 * agent, 2 * agent + 1)
    local_measurement = full_measurement[list(rows), :]

    if ltr_rho > 0.0:
        column = reduced_input[:, agent : agent + 1]
        boost = ltr_rho * (column @ column.T)
        dimension = transition.shape[0]
        noise_w = process_noise * np.eye(dimension) + boost
        row_scale = np.linalg.norm(local_measurement, axis=1) ** 2
        noise_v = measurement_noise * np.diag(row_scale)
        covariance = solve_discrete_are(
            transition.T, local_measurement.T, noise_w, noise_v
        )
        innovation = local_measurement @ covariance @ local_measurement.T + noise_v
        observer_gain = (
            transition @ covariance @ local_measurement.T @ np.linalg.inv(innovation)
        )
    else:
        observer_gain = _discrete_observer_gain(
            model,
            design_speed,
            control_dt,
            local_measurement,
            process_noise,
            measurement_noise,
        )

    return DecentralizedDesign(
        gain_row=gain[agent, :], observer_gain=observer_gain, measurement_rows=rows
    )


def decentralized_closed_loop_matrix(
    model: AeroelasticModel,
    designs: list[DecentralizedDesign],
    eval_speed: float,
    control_dt: float,
    authority_scale: float = 1.0,
) -> np.ndarray:
    """Stacked discrete closed-loop matrix: true state plus one estimate per agent.

    ``authority_scale`` multiplies the *true* control-influence matrix relative
    to what the gains/observers were designed against, modelling a control
    surface that is more or less effective than the design assumed -- exactly
    the quantity the paper's UVLM cross-validation reports a 1.9x discrepancy
    in. The compensator (gain, observer) is left as synthesised; only the true
    plant seen by the loop changes, matching the pattern in
    ``aegis.analysis.robustness.closed_loop_matrix``.
    """
    transition, input_matrix = _discrete_plant(model, eval_speed, control_dt)
    true_input = input_matrix * authority_scale
    measurement = _measurement_model(model, eval_speed)
    n_surfaces = model.wing.n_surfaces
    dimension = transition.shape[0]
    n_blocks = 1 + n_surfaces
    size = n_blocks * dimension
    closed = np.zeros((size, size))

    # Command row i comes only from agent i's own estimate block.
    gain_rows = np.zeros((n_surfaces, n_surfaces * dimension))
    for agent, design in enumerate(designs):
        gain_rows[agent, agent * dimension : (agent + 1) * dimension] = (
            -design.gain_row
        )

    # True state block: x_{k+1} = A x_k + B_true * u_k, u_k = gain_rows @ xhat.
    closed[:dimension, :dimension] = transition
    closed[:dimension, dimension:] = true_input @ gain_rows

    for agent, design in enumerate(designs):
        row0 = dimension * (1 + agent)
        rows = list(design.measurement_rows)
        c_i = measurement[rows, :]
        l_i = design.observer_gain

        # xhat_iـ{k+1} = (A - L_i C_i) xhat_i + B_true u_k + L_i C_i x_k.
        closed[row0 : row0 + dimension, :dimension] = l_i @ c_i
        closed[row0 : row0 + dimension, dimension:] = true_input @ gain_rows
        closed[row0 : row0 + dimension, row0 : row0 + dimension] += (
            transition - l_i @ c_i
        )

    return closed


def decentralized_closed_loop_stable(
    model: AeroelasticModel,
    designs: list[DecentralizedDesign],
    eval_speed: float,
    control_dt: float,
    authority_scale: float = 1.0,
) -> bool:
    """Discrete-time stability: every eigenvalue of the closed loop inside the unit circle."""
    closed = decentralized_closed_loop_matrix(
        model, designs, eval_speed, control_dt, authority_scale
    )
    eigenvalues = np.linalg.eigvals(closed)
    return bool(np.all(np.abs(eigenvalues) < 1.0 - _ZERO_EIGENVALUE_TOLERANCE))


def control_authority_margin(
    model: AeroelasticModel,
    designs: list[DecentralizedDesign],
    eval_speed: float,
    control_dt: float,
    direction: str = "loss",
    search_bound: float = 5.0,
    tolerance: float = 1e-3,
) -> float:
    """Multiplicative control-authority factor at which the closed loop goes unstable.

    ``direction="loss"`` searches downward from 1.0 (surfaces less effective than
    designed for -- the direction that matters for the 1.9x UVLM
    over-estimate already reported in the paper); ``direction="gain"`` searches
    upward. Returns the boundary scale factor; a smaller distance from 1.0 (for
    "loss") means a smaller tolerable identification error before instability.
    """
    if direction not in ("loss", "gain"):
        raise ValueError(f"direction must be 'loss' or 'gain', got {direction!r}")

    nominal_stable = decentralized_closed_loop_stable(
        model, designs, eval_speed, control_dt, authority_scale=1.0
    )
    if not nominal_stable:
        return 1.0

    if direction == "loss":
        low, high = 0.0, 1.0
        while high - low > tolerance:
            mid = 0.5 * (low + high)
            if decentralized_closed_loop_stable(
                model, designs, eval_speed, control_dt, authority_scale=mid
            ):
                high = mid
            else:
                low = mid
        return high

    low, high = 1.0, search_bound
    if decentralized_closed_loop_stable(
        model, designs, eval_speed, control_dt, authority_scale=high
    ):
        return float("inf")
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if decentralized_closed_loop_stable(
            model, designs, eval_speed, control_dt, authority_scale=mid
        ):
            low = mid
        else:
            high = mid
    return low
