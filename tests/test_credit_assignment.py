"""Numerical verification of Proposition 1 -- the closed-form difference reward.

The claim in ``docs/ALGORITHM.md`` is that in this plant the Wolpert-Tumer
difference reward needs no counterfactual rollout, because the only term of the
energy-rate budget that depends on agent ``k``'s deflection is its own control
power ``P_k = (qdot^T b_k) delta_k``.

That is a claim about the plant, not about a controller, so it is worth checking
against a brute-force counterfactual rather than trusting the algebra. These
tests do exactly that: they zero one agent's deflection, recompute the energy
rate from the plant, and require the change to equal ``-P_k`` to floating-point
precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING
from aegis.simulator import WingSimulation

AIRSPEED = 151.0


@pytest.fixture(scope="module")
def model() -> AeroelasticModel:
    return AeroelasticModel(GOLAND_WING)


@pytest.fixture()
def excited_sim(model: AeroelasticModel) -> WingSimulation:
    """A simulation driven to a generic, non-degenerate state."""
    sim = WingSimulation(model, AIRSPEED)
    rng = np.random.default_rng(7)
    sim.reset(rng, initial_tip_plunge=0.04, initial_tip_twist=0.02)
    for _ in range(40):
        sim.step(rng.uniform(-0.7, 0.7, size=model.wing.n_surfaces))
    return sim


def _energy_rate_with_deflection(sim: WingSimulation, deflection: np.ndarray) -> float:
    """Energy rate the plant would have at the current state for a given deflection."""
    original = sim.deflection.copy()
    sim._state[sim._n_plant :] = deflection
    try:
        return sim.energy_rate()
    finally:
        sim._state[sim._n_plant :] = original


def test_energy_rate_matches_state_space_derivative(excited_sim: WingSimulation) -> None:
    """Cross-check the energy budget against the assembled state-space matrix.

    ``energy_rate`` is evaluated from the M/C/K/L matrices; the modal
    acceleration comes from the ``A`` matrix built separately in
    :meth:`AeroelasticModel.state_space`. Agreement therefore tests both the
    budget algebra *and* that the two assembly paths are consistent -- something
    a finite difference over a control interval could never resolve, since the
    fastest mode here is 54 Hz and the actuators move during the interval.
    """
    sim = excited_sim
    modal_acceleration = sim.modal_acceleration
    expected = (
        sim.modal_rate @ sim.total_mass @ modal_acceleration
        + sim.modal_rate @ sim.total_stiffness @ sim.modal_position
    )
    assert sim.energy_rate() == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_difference_reward_equals_control_power(excited_sim: WingSimulation) -> None:
    """Proposition 1: removing agent k changes the energy rate by exactly -P_k."""
    sim = excited_sim
    full_rate = sim.energy_rate()
    power = sim.control_power()

    for k in range(sim.wing.n_surfaces):
        counterfactual = sim.deflection.copy()
        counterfactual[k] = 0.0
        removed_rate = _energy_rate_with_deflection(sim, counterfactual)
        assert full_rate - removed_rate == pytest.approx(power[k], rel=1e-10, abs=1e-9)


def test_control_power_is_exactly_additive(excited_sim: WingSimulation) -> None:
    """Zeroing every agent removes exactly the sum of the individual powers."""
    sim = excited_sim
    full_rate = sim.energy_rate()
    silent_rate = _energy_rate_with_deflection(sim, np.zeros(sim.wing.n_surfaces))
    assert full_rate - silent_rate == pytest.approx(
        sim.control_power().sum(), rel=1e-10, abs=1e-9
    )


def test_no_cross_agent_coupling_in_credit(excited_sim: WingSimulation) -> None:
    """Agent k's credit is independent of what the other agents are doing.

    This is the part that makes the decomposition useful: it is not merely
    additive at one operating point, it has no cross terms at all. A learned
    mixing network would have to discover that.
    """
    sim = excited_sim
    baseline = sim.control_power()

    perturbed = sim.deflection.copy()
    perturbed[1:] = 0.0
    with_others_silent = _energy_rate_with_deflection(
        sim, perturbed
    ) - _energy_rate_with_deflection(sim, np.zeros_like(perturbed))
    assert with_others_silent == pytest.approx(baseline[0], rel=1e-10, abs=1e-9)


def test_positive_power_means_energy_injection(model: AeroelasticModel) -> None:
    """Sign convention: negative control power extracts energy from the structure."""
    sim = WingSimulation(model, AIRSPEED)
    sim.reset(initial_tip_plunge=0.05)
    for _ in range(5):
        sim.step(np.zeros(model.wing.n_surfaces))

    influence = sim.modal_rate @ sim.control_influence
    # Deflect each surface along its own influence direction: that must inject.
    aligned = np.sign(influence) * 0.5
    injecting = _energy_rate_with_deflection(
        sim, aligned * sim.actuators.max_deflection
    )
    silent = _energy_rate_with_deflection(sim, np.zeros(model.wing.n_surfaces))
    assert injecting > silent


def test_modal_residues_expose_control_reversal(model: AeroelasticModel) -> None:
    """Residues must differ in magnitude across surfaces, and be signed.

    If every surface had the same residue on the critical mode there would be no
    credit-assignment problem to solve, so this is a sanity check on the testbed
    rather than on the algorithm.
    """
    residues = model.modal_residues(AIRSPEED)
    assert residues.shape == (model.n_modes, model.wing.n_surfaces)

    critical = residues[0]  # lowest in-vacuo mode
    spread = np.abs(critical).max() / max(np.abs(critical).min(), 1e-12)
    assert spread > 2.0, f"surfaces too similar to be interesting: {critical}"
