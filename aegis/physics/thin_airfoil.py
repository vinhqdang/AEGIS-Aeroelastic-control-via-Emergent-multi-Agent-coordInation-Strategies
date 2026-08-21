"""Thin-airfoil control-surface effectiveness derivatives.

A deflected plain flap is a camber-line kink, so Glauert's thin-airfoil series
gives the steady lift and moment increments in closed form. Deflection is
positive trailing-edge down, which lowers the aft camber line
(``dz/dx = -delta`` behind the hinge).

The closed-form results are

    Cl_delta      = 2 (pi - theta_h + sin theta_h)
    Cm_delta,c/4  = -(1/2)(sin theta_h - (1/2) sin 2 theta_h)

with ``cos theta_h = 1 - 2 x_h / c``. :func:`glauert_coefficients_numeric`
recomputes the same integrals by quadrature and exists so the tests can
cross-check the algebra rather than trusting a transcribed formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.physics.wing import QUARTER_CHORD_FRAC


@dataclass(frozen=True)
class FlapDerivatives:
    """Per-radian steady aerodynamic derivatives of one control surface."""

    cl_delta: float
    cm_delta_quarter_chord: float
    cm_delta_elastic_axis: float


def flap_derivatives(hinge_frac: float, ea_frac: float) -> FlapDerivatives:
    """Steady lift/moment derivatives for a plain flap hinged at ``hinge_frac``.

    ``cm_delta_elastic_axis`` is nose-up positive about the elastic axis, which
    is what the modal torsion equation needs. Transferring the quarter-chord
    moment aft to the EA adds ``+Cl * (x_ea - x_c/4) / c``: lift acting ahead of
    the elastic axis pitches the section nose-up.
    """
    if not 0.0 < hinge_frac < 1.0:
        raise ValueError(f"hinge_frac must lie in (0, 1), got {hinge_frac}")

    theta_h = np.arccos(1.0 - 2.0 * hinge_frac)
    cl_delta = 2.0 * (np.pi - theta_h + np.sin(theta_h))
    cm_quarter = -0.5 * (np.sin(theta_h) - 0.5 * np.sin(2.0 * theta_h))
    cm_ea = cm_quarter + cl_delta * (ea_frac - QUARTER_CHORD_FRAC)
    return FlapDerivatives(float(cl_delta), float(cm_quarter), float(cm_ea))


def glauert_coefficients_numeric(
    hinge_frac: float, n_quad: int = 20_001
) -> tuple[float, float]:
    """Quadrature evaluation of ``(Cl_delta, Cm_delta,c/4)``.

    Independent reference implementation for the closed forms above.
    """
    theta = np.linspace(0.0, np.pi, n_quad)
    theta_h = np.arccos(1.0 - 2.0 * hinge_frac)
    # Camber slope per unit deflection: -1 aft of the hinge, 0 ahead of it.
    slope = np.where(theta > theta_h, -1.0, 0.0)

    a0 = -np.trapezoid(slope, theta) / np.pi
    a1 = 2.0 * np.trapezoid(slope * np.cos(theta), theta) / np.pi
    a2 = 2.0 * np.trapezoid(slope * np.cos(2.0 * theta), theta) / np.pi

    cl_delta = 2.0 * np.pi * (a0 + 0.5 * a1)
    cm_quarter = -0.25 * np.pi * (a1 - a2)
    return float(cl_delta), float(cm_quarter)
