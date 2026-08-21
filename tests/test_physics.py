"""Validation of the aeroelastic plant against known results.

The load-bearing test here is :func:`test_goland_flutter_matches_published`. If
the flutter point drifts, every downstream result is meaningless, so it is
pinned against the published Goland benchmark rather than against a value this
code once produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.actuators import ActuatorBank
from aegis.gusts import DiscreteGust, DrydenTurbulence
from aegis.physics import modes
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.thin_airfoil import flap_derivatives, glauert_coefficients_numeric
from aegis.physics.wing import GOLAND_WING, ControlSurface, WingProperties
from aegis.simulator import WingSimulation

# Published sea-level Goland cantilever wing flutter point.
PUBLISHED_FLUTTER_SPEED = 137.2   # m/s
PUBLISHED_FLUTTER_OMEGA = 70.7    # rad/s
PUBLISHED_BENDING_HZ = 7.66
PUBLISHED_TORSION_HZ = 15.24


@pytest.fixture(scope="module")
def model() -> AeroelasticModel:
    return AeroelasticModel(GOLAND_WING)


# ------------------------------------------------------------------ mode shapes
def test_bending_modes_satisfy_clamped_boundary_conditions() -> None:
    semispan = GOLAND_WING.semispan
    near_root = np.asarray([0.0, 1e-6])
    for mode in (1, 2, 3, 4):
        shape = modes.bending_shape(near_root, semispan, mode)
        assert shape[0] == pytest.approx(0.0, abs=1e-12)
        # Zero slope at the root: the second sample must still be negligible.
        assert abs(shape[1]) < 1e-10


def test_mode_shapes_are_normalised_to_unit_tip() -> None:
    tip = np.asarray([GOLAND_WING.semispan])
    for mode in (1, 2, 3):
        assert modes.bending_shape(tip, GOLAND_WING.semispan, mode)[0] == pytest.approx(1.0)
        assert modes.torsion_shape(tip, GOLAND_WING.semispan, mode)[0] == pytest.approx(
            1.0, abs=1e-12
        )


def test_bending_curvature_is_the_second_derivative() -> None:
    semispan = GOLAND_WING.semispan
    y = np.linspace(0.5, semispan - 0.5, 40)
    step = 1e-4
    numeric = (
        modes.bending_shape(y + step, semispan, 2)
        - 2.0 * modes.bending_shape(y, semispan, 2)
        + modes.bending_shape(y - step, semispan, 2)
    ) / step**2
    assert modes.bending_curvature(y, semispan, 2) == pytest.approx(numeric, rel=1e-4)


def test_unknown_mode_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside tabulated range"):
        modes.bending_shape(np.asarray([1.0]), 6.0, 9)


# ---------------------------------------------------------------- thin airfoil
@pytest.mark.parametrize("hinge_frac", [0.6, 0.7, 0.75, 0.8, 0.9])
def test_closed_form_flap_derivatives_match_quadrature(hinge_frac: float) -> None:
    """Guards the transcribed Glauert algebra against an independent integration."""
    closed = flap_derivatives(hinge_frac, ea_frac=0.33)
    numeric_cl, numeric_cm = glauert_coefficients_numeric(hinge_frac)
    assert closed.cl_delta == pytest.approx(numeric_cl, rel=1e-4)
    assert closed.cm_delta_quarter_chord == pytest.approx(numeric_cm, rel=1e-4)


def test_quarter_chord_flap_effectiveness_is_physical() -> None:
    """A 25%-chord plain flap gives roughly 3.8-4.0 per radian by thin-airfoil theory."""
    derivatives = flap_derivatives(0.75, ea_frac=0.33)
    assert 3.7 < derivatives.cl_delta < 4.1
    # Flap down produces lift up and a nose-down moment about the quarter chord.
    assert derivatives.cm_delta_quarter_chord < 0.0


def test_elastic_axis_transfer_uses_the_right_sign() -> None:
    """Lift ahead of the elastic axis must pitch the section nose-up."""
    forward_ea = flap_derivatives(0.75, ea_frac=0.10)
    aft_ea = flap_derivatives(0.75, ea_frac=0.50)
    assert aft_ea.cm_delta_elastic_axis > forward_ea.cm_delta_elastic_axis


# -------------------------------------------------------------- structural modes
def test_goland_natural_frequencies_match_published(model: AeroelasticModel) -> None:
    frequencies_hz = model.structural_frequencies / (2.0 * np.pi)
    assert frequencies_hz[0] == pytest.approx(PUBLISHED_BENDING_HZ, rel=0.01)
    assert frequencies_hz[1] == pytest.approx(PUBLISHED_TORSION_HZ, rel=0.01)


def test_mass_matrix_is_symmetric_positive_definite(model: AeroelasticModel) -> None:
    assert model.mass == pytest.approx(model.mass.T)
    assert np.all(np.linalg.eigvalsh(model.mass) > 0.0)


def test_static_unbalance_couples_bending_and_torsion(model: AeroelasticModel) -> None:
    """Inertial coupling is what makes flutter possible; it must be non-zero."""
    n_bending = GOLAND_WING.n_bending
    coupling = model.mass[:n_bending, n_bending:]
    assert np.abs(coupling).max() > 1.0


# --------------------------------------------------------------------- flutter
def test_goland_flutter_matches_published(model: AeroelasticModel) -> None:
    point = model.flutter_point()
    assert point.found
    assert point.airspeed == pytest.approx(PUBLISHED_FLUTTER_SPEED, rel=0.03)
    assert point.frequency == pytest.approx(PUBLISHED_FLUTTER_OMEGA, rel=0.05)


def test_stability_switches_across_the_flutter_speed(model: AeroelasticModel) -> None:
    point = model.flutter_point()
    assert model.max_growth_rate(point.airspeed * 0.9) < 0.0
    assert model.max_growth_rate(point.airspeed * 1.1) > 0.0


def test_state_space_dimensions(model: AeroelasticModel) -> None:
    plant = model.state_space(120.0)
    expected = 2 * model.n_modes + 2 * GOLAND_WING.n_strips
    assert plant.a_matrix.shape == (expected, expected)
    assert plant.control_matrix.shape == (expected, GOLAND_WING.n_surfaces)
    assert plant.gust_matrix.shape == (expected, 1)


def test_flutter_speed_is_insensitive_to_strip_count() -> None:
    """Strip-count convergence: 16 and 32 strips must agree closely."""
    speeds = []
    for n_strips in (16, 32):
        wing = WingProperties(
            **{**GOLAND_WING.__dict__, "n_strips": n_strips}
        )
        speeds.append(AeroelasticModel(wing).flutter_point().airspeed)
    assert speeds[0] == pytest.approx(speeds[1], rel=0.01)


def test_nonpositive_airspeed_is_rejected(model: AeroelasticModel) -> None:
    with pytest.raises(ValueError, match="airspeed must be positive"):
        model.state_space(0.0)


# -------------------------------------------------------------------- actuators
def test_actuator_respects_deflection_limit() -> None:
    bank = ActuatorBank(GOLAND_WING)
    at_limit = bank.max_deflection.copy()
    rate = bank.derivative(at_limit, bank.max_deflection * 5.0)
    assert np.all(rate == 0.0)


def test_actuator_respects_rate_limit() -> None:
    bank = ActuatorBank(GOLAND_WING)
    rate = bank.derivative(np.zeros(3), bank.max_deflection)
    assert np.all(np.abs(rate) <= bank.max_rate + 1e-12)


def test_jammed_surface_never_moves() -> None:
    bank = ActuatorBank(GOLAND_WING)
    bank.jam({"tab_outboard": np.deg2rad(8.0)})
    rate = bank.derivative(bank.initial_deflection(), np.full(3, 0.2))
    assert rate[2] == 0.0
    assert np.any(rate[:2] != 0.0)
    assert bank.initial_deflection()[2] == pytest.approx(np.deg2rad(8.0))


def test_jam_beyond_travel_is_rejected() -> None:
    bank = ActuatorBank(GOLAND_WING)
    with pytest.raises(ValueError, match="exceeds travel limit"):
        bank.jam({"tab_outboard": 1.0})


def test_unknown_surface_name_is_rejected() -> None:
    bank = ActuatorBank(GOLAND_WING)
    with pytest.raises(ValueError, match="unknown surface names"):
        bank.jam({"winglet": 0.1})


# ------------------------------------------------------------------------ gusts
def test_discrete_gust_starts_and_ends_at_zero() -> None:
    gust = DiscreteGust(amplitude=5.0, gust_length=25.0, airspeed=150.0, randomize=False)
    gust.reset(np.random.default_rng(0))
    assert gust.velocity(0.0) == 0.0
    assert gust.velocity(gust.start_time) == pytest.approx(0.0)
    assert gust.velocity(gust.start_time + gust.duration) == pytest.approx(0.0)
    peak = gust.velocity(gust.start_time + 0.5 * gust.duration)
    assert peak == pytest.approx(5.0, rel=1e-6)


def test_dryden_turbulence_is_reproducible() -> None:
    turbulence = DrydenTurbulence(intensity=2.0, duration=2.0, seed=3)
    first = [turbulence.velocity(t) for t in np.linspace(0.5, 1.5, 200)]

    turbulence.reset(np.random.default_rng(3))
    second = [turbulence.velocity(t) for t in np.linspace(0.5, 1.5, 200)]
    assert first == pytest.approx(second)


@pytest.mark.parametrize("intensity", [1.0, 2.0, 5.0])
def test_dryden_realises_the_requested_intensity(intensity: float) -> None:
    """RMS must equal the requested sigma.

    The record has to be long relative to the L/U correlation time -- about 3.6 s
    at these settings -- or the sample RMS is dominated by statistical scatter
    rather than by the filter gain. A 600 s record gives ~170 independent
    samples, enough to catch a real normalisation error while staying fast.
    """
    turbulence = DrydenTurbulence(
        intensity=intensity, duration=600.0, sample_rate=200.0, seed=11
    )
    assert turbulence.realised_rms() == pytest.approx(intensity, rel=0.10)


def test_dryden_is_zero_mean_over_a_long_record() -> None:
    turbulence = DrydenTurbulence(
        intensity=2.0, duration=600.0, sample_rate=200.0, seed=17
    )
    assert abs(turbulence._samples.mean()) < 0.25 * turbulence.realised_rms()


# -------------------------------------------------------------------- simulator
def test_open_loop_diverges_above_flutter_speed(model: AeroelasticModel) -> None:
    speed = model.flutter_point().airspeed * 1.10
    sim = WingSimulation(model, speed, divergence_limit=0.9)
    trajectory = sim.rollout(None, duration=3.0, initial_tip_plunge=0.05)
    assert trajectory.diverged
    assert trajectory.energy_growth_rate() > 0.5


def test_open_loop_decays_below_flutter_speed(model: AeroelasticModel) -> None:
    speed = model.flutter_point().airspeed * 0.85
    sim = WingSimulation(model, speed)
    trajectory = sim.rollout(None, duration=1.5, initial_tip_plunge=0.05)
    assert not trajectory.diverged
    assert trajectory.energy_growth_rate() < 0.0


def test_centralized_lqr_stabilises_above_flutter_speed(model: AeroelasticModel) -> None:
    from aegis.control.lqr import CentralizedLQR

    speed = model.flutter_point().airspeed * 1.10
    sim = WingSimulation(model, speed, divergence_limit=0.9)
    trajectory = sim.rollout(
        CentralizedLQR(model, speed), duration=2.0, initial_tip_plunge=0.05
    )
    assert not trajectory.diverged
    assert trajectory.energy_growth_rate() < -1.0
    assert trajectory.peak_tip_plunge < 0.10


def test_action_shape_is_validated(model: AeroelasticModel) -> None:
    sim = WingSimulation(model, 120.0)
    with pytest.raises(ValueError, match="expected 3 actions"):
        sim.step(np.zeros(5))


def test_wing_rejects_overlapping_control_surface_names() -> None:
    duplicated = (
        ControlSurface("flap", 0.0, 0.4),
        ControlSurface("flap", 0.5, 0.9),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        WingProperties(**{**GOLAND_WING.__dict__, "surfaces": duplicated})


def test_control_surface_band_is_validated() -> None:
    with pytest.raises(ValueError, match="0 <= start < end <= 1"):
        ControlSurface("bad", 0.7, 0.3)
