"""Modal aeroelastic plant: coupled bending-torsion wing with unsteady strip aero.

Model chain
-----------
1. Structure. Galerkin projection of a uniform cantilever wing onto the exact
   clamped-free bending and fixed-free torsion modes (aegis.physics.modes). All
   generalized mass/stiffness integrals are evaluated by Gauss-Legendre
   quadrature, so the modal basis is swappable without touching the assembly.
2. Aerodynamics. Strip theory. Each strip carries the full Theodorsen
   non-circulatory (apparent-mass) terms plus a circulatory term whose
   Theodorsen lag is realised in the time domain by two Wagner-function lag
   states per strip, using the R.T. Jones two-exponential approximation
   Phi(s) = 1 - 0.165 exp(-0.0455 s) - 0.335 exp(-0.3 s), with s = U t / b.
3. Control surfaces. Quasi-steady thin-airfoil flap derivatives
   (aegis.physics.thin_airfoil) integrated over each surface's spanwise band.
   The flap's own wake shedding is neglected -- a standard simplification for
   control-design models, noted here because it slightly over-estimates
   control authority at high frequency.

Every aerodynamic matrix scales as a fixed power of the freestream speed, so the
speed-independent blocks are assembled once and rescaled per U. That makes a
flutter sweep almost free and lets the RL environment change flight condition
between episodes without rebuilding the model.

Sign conventions follow aegis.physics.wing: plunge positive down, twist positive
nose-up, deflection positive trailing-edge down.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import eigh

from aegis.physics import modes
from aegis.physics.thin_airfoil import flap_derivatives
from aegis.physics.wing import SEA_LEVEL_DENSITY, WingProperties

# Two-lag approximation of the Wagner function (R.T. Jones).
WAGNER_AMPLITUDES = (0.165, 0.335)
WAGNER_EXPONENTS = (0.0455, 0.3)
N_LAG_PER_STRIP = len(WAGNER_AMPLITUDES)

_QUAD_ORDER = 200


@dataclass(frozen=True)
class StateSpace:
    """Continuous-time plant at one freestream speed.

    The dynamics are zdot = A z + B_delta * delta + B_gust * w_gust, with state
    z = [q, qdot, wake]: q stacks the bending then torsion modal amplitudes and
    wake holds two Wagner lag states per strip.
    """

    a_matrix: np.ndarray
    control_matrix: np.ndarray
    gust_matrix: np.ndarray
    airspeed: float
    n_modes: int

    @property
    def n_states(self) -> int:
        return self.a_matrix.shape[0]

    def eigenvalues(self) -> np.ndarray:
        """Aeroelastic eigenvalues, sorted by ascending |imaginary part|."""
        values = np.linalg.eigvals(self.a_matrix)
        return values[np.argsort(np.abs(values.imag))]


@dataclass(frozen=True)
class FlutterPoint:
    """Result of a flutter search."""

    airspeed: float
    frequency: float  # rad/s
    found: bool

    def __str__(self) -> str:
        if not self.found:
            return "no flutter crossing inside the search bracket"
        return f"U_f = {self.airspeed:.2f} m/s, omega_f = {self.frequency:.2f} rad/s"


class AeroelasticModel:
    """Speed-parameterised modal aeroelastic model of a wing with control surfaces."""

    def __init__(self, wing: WingProperties, density: float = SEA_LEVEL_DENSITY):
        if density <= 0:
            raise ValueError(f"density must be positive, got {density}")
        self.wing = wing
        self.density = density

        self._strip_centers, self._strip_widths = _strip_grid(wing)
        self._plunge_rows, self._twist_rows = self._modal_rows()
        self._assemble_structural()
        self._assemble_aerodynamic()
        self._assemble_control()

    # ------------------------------------------------------------------ sizes
    @property
    def n_modes(self) -> int:
        return self.wing.n_modes

    @property
    def n_lag(self) -> int:
        return self.wing.n_strips * N_LAG_PER_STRIP

    @property
    def n_states(self) -> int:
        return 2 * self.n_modes + self.n_lag

    @property
    def n_inputs(self) -> int:
        return self.wing.n_surfaces

    @cached_property
    def structural_frequencies(self) -> np.ndarray:
        """In-vacuo natural frequencies, rad/s, ascending."""
        eigenvalues = eigh(self.stiffness, self.mass, eigvals_only=True)
        return np.sqrt(np.maximum(eigenvalues, 0.0))

    # -------------------------------------------------------------- assembly
    def _modal_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-strip modal participation rows for plunge and twist."""
        w = self.wing
        y = self._strip_centers
        plunge = np.zeros((w.n_strips, self.n_modes))
        twist = np.zeros((w.n_strips, self.n_modes))
        for i in range(w.n_bending):
            plunge[:, i] = modes.bending_shape(y, w.semispan, i + 1)
        for j in range(w.n_torsion):
            twist[:, w.n_bending + j] = modes.torsion_shape(y, w.semispan, j + 1)
        return plunge, twist

    def _assemble_structural(self) -> None:
        w = self.wing
        nodes, weights = np.polynomial.legendre.leggauss(_QUAD_ORDER)
        y = 0.5 * w.semispan * (nodes + 1.0)
        jacobian = 0.5 * w.semispan * weights

        phi = np.column_stack(
            [modes.bending_shape(y, w.semispan, i + 1) for i in range(w.n_bending)]
        )
        phi_dd = np.column_stack(
            [modes.bending_curvature(y, w.semispan, i + 1) for i in range(w.n_bending)]
        )
        psi = np.column_stack(
            [modes.torsion_shape(y, w.semispan, j + 1) for j in range(w.n_torsion)]
        )
        psi_d = np.column_stack(
            [modes.torsion_gradient(y, w.semispan, j + 1) for j in range(w.n_torsion)]
        )

        def integrate(left: np.ndarray, right: np.ndarray, coeff: float) -> np.ndarray:
            return coeff * (left * jacobian[:, None]).T @ right

        n, nb = self.n_modes, w.n_bending
        mass = np.zeros((n, n))
        mass[:nb, :nb] = integrate(phi, phi, w.mass_per_length)
        coupling = integrate(phi, psi, w.static_unbalance)
        mass[:nb, nb:] = coupling
        mass[nb:, :nb] = coupling.T
        mass[nb:, nb:] = integrate(psi, psi, w.inertia_per_length)

        stiffness = np.zeros((n, n))
        stiffness[:nb, :nb] = integrate(phi_dd, phi_dd, w.bending_stiffness)
        stiffness[nb:, nb:] = integrate(psi_d, psi_d, w.torsional_stiffness)

        self.mass = mass
        self.stiffness = stiffness
        self.damping = _modal_damping_matrix(mass, stiffness, w.modal_damping)

    def _assemble_aerodynamic(self) -> None:
        """Speed-independent factors of the strip-theory aerodynamic operators."""
        w = self.wing
        b, a = w.semichord, w.ea_offset_semichords
        rho = self.density
        plunge, twist = self._plunge_rows, self._twist_rows

        # Downwash at the 3/4-chord point: hdot + U*alpha + b(1/2 - a)*alphadot.
        self._downwash_from_rate = plunge + b * (0.5 - a) * twist
        self._downwash_from_state = twist  # scaled by U where it is used

        apparent = np.pi * rho * b**2
        n = self.n_modes
        aero_mass = np.zeros((n, n))
        apparent_damping = np.zeros((n, n))
        circulatory = np.zeros((n, w.n_strips))

        for s in range(w.n_strips):
            dy = self._strip_widths[s]
            h_row = plunge[s][None, :]
            a_row = twist[s][None, :]

            aero_mass += dy * apparent * (
                h_row.T @ (h_row - b * a * a_row)
                - a_row.T @ (b * a * h_row - b**2 * (0.125 + a**2) * a_row)
            )
            apparent_damping += dy * apparent * (
                h_row.T @ a_row + b * (0.5 - a) * a_row.T @ a_row
            )
            # Generalized load per unit U and unit effective downwash. Lift is up
            # while the plunge DOF is down, hence the leading minus sign.
            circulatory[:, s] = dy * (
                -h_row.ravel() * 2.0 * np.pi * rho * b
                + a_row.ravel() * 2.0 * np.pi * rho * b**2 * (a + 0.5)
            )

        self._circulatory_unit = circulatory
        self._aero_mass = aero_mass
        # The Wagner realisation splits the circulatory downwash into an
        # instantaneous half plus the wake-lag contribution (Phi(0) = 1/2).
        self._aero_damping_unit = apparent_damping - 0.5 * (
            circulatory @ self._downwash_from_rate
        )
        self._aero_stiffness_unit = -0.5 * (circulatory @ twist)

        lag_gain = np.zeros((n, self.n_lag))
        for s in range(w.n_strips):
            for i, (amp, eps) in enumerate(zip(WAGNER_AMPLITUDES, WAGNER_EXPONENTS)):
                lag_gain[:, s * N_LAG_PER_STRIP + i] = circulatory[:, s] * amp * eps / b
        self._lag_gain_unit = lag_gain

        # Lag dynamics: xdot = -eps*(U/b)*x + w_{3/4}.
        self._lag_decay_unit = np.tile(np.asarray(WAGNER_EXPONENTS), w.n_strips) / b
        self._lag_from_rate = np.repeat(
            self._downwash_from_rate, N_LAG_PER_STRIP, axis=0
        )
        self._lag_from_state = np.repeat(twist, N_LAG_PER_STRIP, axis=0)

    def _assemble_control(self) -> None:
        """Per-surface generalized-load columns, divided by dynamic pressure."""
        w = self.wing
        control = np.zeros((self.n_modes, w.n_surfaces))
        weights = np.zeros((w.n_surfaces, w.n_strips))

        for k, surface in enumerate(w.surfaces):
            derivatives = flap_derivatives(surface.hinge_frac, w.ea_frac)
            overlap = _band_overlap(
                self._strip_centers,
                self._strip_widths,
                surface.y_start_frac * w.semispan,
                surface.y_end_frac * w.semispan,
            )
            weights[k] = overlap
            for s in np.nonzero(overlap)[0]:
                control[:, k] += overlap[s] * 0.5 * self.density * (
                    -self._plunge_rows[s] * w.chord * derivatives.cl_delta
                    + self._twist_rows[s] * w.chord**2 * derivatives.cm_delta_elastic_axis
                )
        self.surface_strip_weights = weights
        self._control_unit = control

    def total_mass(self, airspeed: float) -> np.ndarray:
        """Structural plus apparent-mass matrix appearing in the equation of motion."""
        del airspeed  # apparent mass is speed-independent
        return self.mass + self._aero_mass

    def total_damping(self, airspeed: float) -> np.ndarray:
        """Structural plus aerodynamic damping at ``airspeed``."""
        return self.damping + airspeed * self._aero_damping_unit

    def total_stiffness(self, airspeed: float) -> np.ndarray:
        """Structural plus aerodynamic stiffness at ``airspeed``.

        Loses positive definiteness above the divergence speed, which is why the
        energy-rate argument in docs/ALGORITHM.md is stated in terms of work done
        by each surface rather than of a positive-definite energy functional.
        """
        return self.stiffness + airspeed**2 * self._aero_stiffness_unit

    def wake_gain(self, airspeed: float) -> np.ndarray:
        """Wake-lag feed-in matrix ``L`` in the equation of motion."""
        return airspeed**2 * self._lag_gain_unit

    # ---------------------------------------------------------- state space
    def state_space(self, airspeed: float) -> StateSpace:
        """Assemble the plant at freestream speed ``airspeed`` in m/s."""
        if airspeed <= 0.0:
            raise ValueError(f"airspeed must be positive, got {airspeed}")
        u = airspeed
        n = self.n_modes

        total_mass = self.mass + self._aero_mass
        total_damping = self.damping + u * self._aero_damping_unit
        total_stiffness = self.stiffness + u**2 * self._aero_stiffness_unit
        inverse_mass = np.linalg.inv(total_mass)

        a_matrix = np.zeros((self.n_states, self.n_states))
        a_matrix[:n, n : 2 * n] = np.eye(n)
        a_matrix[n : 2 * n, :n] = -inverse_mass @ total_stiffness
        a_matrix[n : 2 * n, n : 2 * n] = -inverse_mass @ total_damping
        a_matrix[n : 2 * n, 2 * n :] = inverse_mass @ (u**2 * self._lag_gain_unit)
        a_matrix[2 * n :, :n] = u * self._lag_from_state
        a_matrix[2 * n :, n : 2 * n] = self._lag_from_rate
        a_matrix[2 * n :, 2 * n :] = -np.diag(u * self._lag_decay_unit)

        control_matrix = np.zeros((self.n_states, self.n_inputs))
        control_matrix[n : 2 * n, :] = inverse_mass @ (u**2 * self._control_unit)

        # A uniform upward gust subtracts from the (downward-positive) downwash
        # at every strip, so it enters through the same circulatory path.
        gust_matrix = np.zeros((self.n_states, 1))
        gust_matrix[n : 2 * n, 0] = inverse_mass @ (
            -0.5 * u * self._circulatory_unit.sum(axis=1)
        )
        gust_matrix[2 * n :, 0] = -1.0
        return StateSpace(a_matrix, control_matrix, gust_matrix, u, n)

    def control_influence(self, airspeed: float) -> np.ndarray:
        """Generalized control force matrix ``B`` at ``airspeed``, shape (n_modes, n_surfaces).

        This is the matrix that appears on the right-hand side of
        ``M xddot + C xdot + K x = ... + B delta``, so column ``k`` dotted with
        the modal rate gives agent ``k``'s instantaneous control power. That
        product is the closed-form difference reward of docs/ALGORITHM.md.
        """
        if airspeed <= 0.0:
            raise ValueError(f"airspeed must be positive, got {airspeed}")
        return airspeed**2 * self._control_unit

    def modal_residues(self, airspeed: float) -> np.ndarray:
        """Control authority per (in-vacuo mode, surface): ``v_r^T b_k``.

        The sign of an entry decides whether a surface can damp or will excite
        that mode -- it is what flips under control reversal.
        """
        _, eigenvectors = eigh(self.stiffness, self.mass)
        return eigenvectors.T @ self.control_influence(airspeed)

    # ------------------------------------------------------- span recovery
    def station_rows(self, y_frac: float) -> tuple[np.ndarray, np.ndarray]:
        """Plunge and twist modal rows at a spanwise fraction of the semispan."""
        if not 0.0 <= y_frac <= 1.0:
            raise ValueError(f"y_frac must lie in [0, 1], got {y_frac}")
        w = self.wing
        y = np.asarray([y_frac * w.semispan])
        plunge = np.zeros(self.n_modes)
        twist = np.zeros(self.n_modes)
        for i in range(w.n_bending):
            plunge[i] = modes.bending_shape(y, w.semispan, i + 1)[0]
        for j in range(w.n_torsion):
            twist[w.n_bending + j] = modes.torsion_shape(y, w.semispan, j + 1)[0]
        return plunge, twist

    def tip_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Modal rows for tip plunge (m, down) and tip twist (rad, nose-up)."""
        return self.station_rows(1.0)

    # ------------------------------------------------------------- flutter
    def flutter_point(
        self,
        speed_min: float = 20.0,
        speed_max: float = 400.0,
        coarse_steps: int = 200,
        tolerance: float = 0.01,
    ) -> FlutterPoint:
        """Lowest speed at which any aeroelastic mode becomes unstable.

        A coarse scan brackets the first sign change of the maximum eigenvalue
        real part, then bisection refines the crossing to ``tolerance`` m/s.
        """
        speeds = np.linspace(speed_min, speed_max, coarse_steps)
        unstable = np.nonzero([self.max_growth_rate(u) > 0.0 for u in speeds])[0]
        if unstable.size == 0:
            return FlutterPoint(float("nan"), float("nan"), found=False)

        first = int(unstable[0])
        if first == 0:
            return FlutterPoint(speeds[0], self.flutter_frequency(speeds[0]), found=True)

        low, high = float(speeds[first - 1]), float(speeds[first])
        while high - low > tolerance:
            mid = 0.5 * (low + high)
            if self.max_growth_rate(mid) > 0.0:
                high = mid
            else:
                low = mid
        return FlutterPoint(high, self.flutter_frequency(high), found=True)

    def max_growth_rate(self, airspeed: float) -> float:
        """Largest eigenvalue real part at ``airspeed`` -- positive means unstable."""
        return float(np.max(np.linalg.eigvals(self.state_space(airspeed).a_matrix).real))

    def flutter_frequency(self, airspeed: float) -> float:
        """Imaginary part of the least-damped eigenvalue at ``airspeed``, rad/s."""
        values = np.linalg.eigvals(self.state_space(airspeed).a_matrix)
        return float(abs(values[int(np.argmax(values.real))].imag))


def _modal_damping_matrix(
    mass: np.ndarray, stiffness: np.ndarray, zeta: float
) -> np.ndarray:
    """Uniform-``zeta`` damping, built so every in-vacuo mode gets that ratio."""
    if zeta <= 0.0:
        return np.zeros_like(mass)
    eigenvalues, eigenvectors = eigh(stiffness, mass)
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))
    return mass @ (eigenvectors * (2.0 * zeta * omega)) @ eigenvectors.T @ mass


def _strip_grid(wing: WingProperties) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width midpoint strips over the semispan."""
    edges = np.linspace(0.0, wing.semispan, wing.n_strips + 1)
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)


def _band_overlap(
    centers: np.ndarray, widths: np.ndarray, y_start: float, y_end: float
) -> np.ndarray:
    """Length of each strip falling inside the band ``[y_start, y_end]``.

    Overlap rather than a centre-inside test keeps control authority a smooth
    function of the surface layout, which matters when sweeping span fractions.
    """
    lower = centers - 0.5 * widths
    upper = centers + 0.5 * widths
    return np.clip(np.minimum(upper, y_end) - np.maximum(lower, y_start), 0.0, None)
