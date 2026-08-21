"""Assumed-mode shapes for a uniform cantilever wing.

Bending uses the exact clamped-free Euler-Bernoulli eigenfunctions; torsion uses
the exact fixed-free harmonic modes. Both families are mass-orthogonal for a
uniform beam, but AEGIS never relies on that -- every generalized mass and
stiffness term is integrated numerically in :mod:`aegis.physics.aeroelastic`, so
the modal basis can be swapped without touching the assembly code.
"""

from __future__ import annotations

import numpy as np

#: Roots of ``cos(x)cosh(x) + 1 = 0`` -- the clamped-free bending eigenvalues.
BENDING_EIGENVALUES = (1.87510407, 4.69409113, 7.85475744, 10.99554073)


def bending_shape(y: np.ndarray, semispan: float, mode: int) -> np.ndarray:
    """Clamped-free bending mode ``mode`` (1-based), unit tip deflection."""
    beta = _bending_beta(semispan, mode)
    sigma = _bending_sigma(beta * semispan)
    raw = _bending_raw(beta * y, sigma)
    return raw / _bending_raw(np.asarray(beta * semispan), sigma)


def bending_curvature(y: np.ndarray, semispan: float, mode: int) -> np.ndarray:
    """Second spanwise derivative of :func:`bending_shape`."""
    beta = _bending_beta(semispan, mode)
    sigma = _bending_sigma(beta * semispan)
    arg = beta * y
    curv = beta**2 * (
        np.cosh(arg) + np.cos(arg) - sigma * (np.sinh(arg) + np.sin(arg))
    )
    return curv / _bending_raw(np.asarray(beta * semispan), sigma)


def torsion_shape(y: np.ndarray, semispan: float, mode: int) -> np.ndarray:
    """Fixed-free torsion mode ``mode`` (1-based), unit tip rotation.

    The raw harmonic ``sin((2m-1) pi y / 2L)`` has tip value ``(-1)^(m-1)``, so
    even modes are sign-flipped to make "unit tip rotation" true for every mode.
    That matters because callers set initial conditions and read out tip motion
    through these shapes, and a silent sign flip on mode 2 would invert them.
    """
    return _torsion_sign(mode) * np.sin(_torsion_wavenumber(semispan, mode) * y)


def torsion_gradient(y: np.ndarray, semispan: float, mode: int) -> np.ndarray:
    """Spanwise derivative of :func:`torsion_shape`."""
    k = _torsion_wavenumber(semispan, mode)
    return _torsion_sign(mode) * k * np.cos(k * y)


def _torsion_sign(mode: int) -> float:
    """Normalisation sign that puts unit rotation at the tip."""
    return -1.0 if mode % 2 == 0 else 1.0


def _bending_beta(semispan: float, mode: int) -> float:
    if not 1 <= mode <= len(BENDING_EIGENVALUES):
        raise ValueError(f"bending mode {mode} outside tabulated range 1..4")
    return BENDING_EIGENVALUES[mode - 1] / semispan


def _torsion_wavenumber(semispan: float, mode: int) -> float:
    if mode < 1:
        raise ValueError(f"torsion mode must be >= 1, got {mode}")
    return (2 * mode - 1) * np.pi / (2.0 * semispan)


def _bending_sigma(beta_l: float) -> float:
    return (np.cosh(beta_l) + np.cos(beta_l)) / (np.sinh(beta_l) + np.sin(beta_l))


def _bending_raw(arg: np.ndarray, sigma: float) -> np.ndarray:
    return (
        np.cosh(arg) - np.cos(arg) - sigma * (np.sinh(arg) - np.sin(arg))
    )
