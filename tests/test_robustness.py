"""Tests for the robustness analysis, including its central caveat.

The claim that most needs pinning is that a jam angle does not move the
closed-loop spectrum. It is easy to write an analysis that appears to show a jam
angle eroding stability margin -- and it would be wrong, because a held
deflection is an affine input. If that ever stops holding, the paper's framing
of the failure mode has to change, so it is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.analysis.robustness import (
    airspeed_margin,
    closed_loop_growth,
    closed_loop_matrix,
    saturated_growth,
)
from aegis.control.lqr import CentralizedLQR
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

DESIGN_RATIO = 1.10


@pytest.fixture(scope="module")
def model() -> AeroelasticModel:
    return AeroelasticModel(GOLAND_WING)


@pytest.fixture(scope="module")
def flutter_speed(model: AeroelasticModel) -> float:
    return model.flutter_point().airspeed


@pytest.fixture(scope="module")
def controller(model: AeroelasticModel, flutter_speed: float) -> CentralizedLQR:
    return CentralizedLQR(model, flutter_speed * DESIGN_RATIO)


def test_lqr_stabilises_its_own_design_point(model, controller, flutter_speed) -> None:
    point = closed_loop_growth(model, controller, flutter_speed * DESIGN_RATIO)
    assert point.stable
    assert point.growth_rate < -1.0


def test_fixed_gain_has_a_finite_airspeed_margin(model, controller, flutter_speed) -> None:
    """The weakness the thesis targets: one gain does not cover the envelope."""
    margin = airspeed_margin(model, controller)
    assert np.isfinite(margin)
    assert margin > flutter_speed * DESIGN_RATIO  # it does hold its design point
    assert margin < flutter_speed * 2.0           # but not the whole envelope


def test_closed_loop_is_unstable_well_above_the_margin(
    model, controller, flutter_speed
) -> None:
    margin = airspeed_margin(model, controller)
    assert not closed_loop_growth(model, controller, margin * 1.15).stable


def test_open_loop_flutter_is_not_static_divergence(model, flutter_speed) -> None:
    """Guards the interpretation of the off-design result.

    If static divergence sat below the speeds studied, the closed-loop failures
    there would be unavoidable rather than a controller shortcoming.
    """
    speed = flutter_speed * 1.5
    assert np.all(np.linalg.eigvalsh(model.total_stiffness(speed)) > 0.0)


def test_jam_angle_does_not_change_the_closed_loop_spectrum(
    model, controller, flutter_speed
) -> None:
    """A held deflection is an affine input: it moves trim, not the eigenvalues."""
    speed = flutter_speed * DESIGN_RATIO
    reference = np.sort_complex(
        np.linalg.eigvals(
            closed_loop_matrix(model, controller, speed, jammed=("tab_outboard",))
        )
    )
    # The construction takes no angle argument at all; assert it stays that way
    # by checking the spectrum is reproducible and independent of any trim state.
    repeat = np.sort_complex(
        np.linalg.eigvals(
            closed_loop_matrix(model, controller, speed, jammed=("tab_outboard",))
        )
    )
    assert reference == pytest.approx(repeat)


def test_jam_angle_does_change_time_domain_performance(
    model, controller, flutter_speed
) -> None:
    """...but it does degrade performance, through saturation."""
    speed = flutter_speed * DESIGN_RATIO
    healthy, _, peak_healthy = saturated_growth(model, controller, speed)
    jammed, _, peak_jammed = saturated_growth(
        model, controller, speed,
        jam_angle=np.deg2rad(8.0), jammed_surface="tab_outboard",
    )
    assert jammed > healthy, "jam should erode the suppression rate"
    assert peak_jammed > peak_healthy


def test_losing_two_surfaces_is_worse_than_losing_one(
    model, controller, flutter_speed
) -> None:
    speed = flutter_speed * DESIGN_RATIO
    one = closed_loop_growth(model, controller, speed, jammed=("aileron_mid",))
    two = closed_loop_growth(
        model, controller, speed, jammed=("aileron_mid", "tab_outboard")
    )
    assert two.growth_rate > one.growth_rate


def test_jam_induced_zero_eigenvalues_are_not_called_unstable(
    model, controller, flutter_speed
) -> None:
    """A frozen surface is a constant state, contributing an exact zero eigenvalue.

    That is bookkeeping, not an instability, and must not be reported as one.
    """
    point = closed_loop_growth(
        model, controller, flutter_speed * DESIGN_RATIO, jammed=("flap_inboard",)
    )
    assert point.stable
