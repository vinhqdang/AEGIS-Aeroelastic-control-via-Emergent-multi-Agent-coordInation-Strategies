"""Wing geometry, structural properties, and control-surface layout.

All quantities are SI. Sign conventions used consistently across AEGIS:

* ``y``     spanwise coordinate, 0 at the root, ``semispan`` at the tip.
* ``h``     local plunge, **positive downward**.
* ``alpha`` local twist, **positive nose-up**.
* ``delta`` control-surface deflection, **positive trailing-edge down**.
* lift is positive up, pitching moment about the elastic axis positive nose-up.

The plunge-down convention is the classical Theodorsen/Fung one, which keeps the
unsteady aerodynamic operators in their textbook form (see ``strip_aero``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Thin-airfoil / atmospheric constants.
SEA_LEVEL_DENSITY = 1.225  # kg/m^3
QUARTER_CHORD_FRAC = 0.25


@dataclass(frozen=True)
class ControlSurface:
    """One trailing-edge control surface occupying a spanwise band.

    ``y_start_frac``/``y_end_frac`` are fractions of the semispan, so the same
    layout transfers to a wing of any size.
    """

    name: str
    y_start_frac: float
    y_end_frac: float
    hinge_frac: float = 0.75          # hinge station, x/c from the leading edge
    max_deflection: float = 0.2618    # rad (15 deg)
    max_rate: float = 3.4907          # rad/s (200 deg/s)
    actuator_tau: float = 0.02        # first-order actuator lag, s

    def __post_init__(self) -> None:
        if not 0.0 <= self.y_start_frac < self.y_end_frac <= 1.0:
            raise ValueError(
                f"{self.name}: span band must satisfy 0 <= start < end <= 1, "
                f"got ({self.y_start_frac}, {self.y_end_frac})"
            )
        if not 0.0 < self.hinge_frac < 1.0:
            raise ValueError(f"{self.name}: hinge_frac must lie in (0, 1)")
        if self.max_deflection <= 0 or self.max_rate <= 0 or self.actuator_tau <= 0:
            raise ValueError(f"{self.name}: limits and actuator_tau must be positive")


@dataclass(frozen=True)
class WingProperties:
    """Uniform cantilever wing with bending/torsion structural dynamics."""

    semispan: float
    chord: float
    mass_per_length: float          # kg/m
    inertia_per_length: float       # kg*m, polar inertia about the elastic axis
    bending_stiffness: float        # EI, N*m^2
    torsional_stiffness: float      # GJ, N*m^2
    ea_frac: float                  # elastic axis, x/c from the leading edge
    cg_frac: float                  # section c.g., x/c from the leading edge
    surfaces: tuple[ControlSurface, ...]
    n_bending: int = 2
    n_torsion: int = 2
    n_strips: int = 24
    modal_damping: float = 0.0      # uniform modal damping ratio
    name: str = "wing"

    def __post_init__(self) -> None:
        if min(self.n_bending, self.n_torsion) < 1:
            raise ValueError("need at least one bending and one torsion mode")
        if self.n_bending > 4:
            raise ValueError("only the first four cantilever bending modes are tabulated")
        if self.n_strips < 4:
            raise ValueError("n_strips must be >= 4 for a meaningful span integration")
        if not self.surfaces:
            raise ValueError("at least one control surface is required")
        names = [s.name for s in self.surfaces]
        if len(set(names)) != len(names):
            raise ValueError(f"control-surface names must be unique, got {names}")

    @property
    def semichord(self) -> float:
        """b, half the chord."""
        return 0.5 * self.chord

    @property
    def ea_offset_semichords(self) -> float:
        """``a``: elastic-axis offset from mid-chord, in semichords, positive aft."""
        return 2.0 * self.ea_frac - 1.0

    @property
    def static_unbalance(self) -> float:
        """S_alpha = m * (x_cg - x_ea), positive when the c.g. is aft of the EA."""
        return self.mass_per_length * (self.cg_frac - self.ea_frac) * self.chord

    @property
    def n_modes(self) -> int:
        return self.n_bending + self.n_torsion

    @property
    def n_surfaces(self) -> int:
        return len(self.surfaces)


def _goland_surfaces() -> tuple[ControlSurface, ...]:
    """Three spanwise-segmented trailing-edge surfaces.

    Inboard-to-outboard the modal influence changes sharply: the outboard tab has
    strong authority over the first bending/torsion modes but little inertia
    behind it, while the inboard flap barely moves mode 1. That asymmetry is the
    whole point of the testbed -- it is what makes credit assignment hard.
    """
    return (
        ControlSurface("flap_inboard", 0.05, 0.40),
        ControlSurface("aileron_mid", 0.40, 0.72),
        ControlSurface("tab_outboard", 0.72, 0.98),
    )


#: Goland cantilever wing -- the standard aeroelastic flutter benchmark.
#: Published sea-level flutter point: U_f ~ 137 m/s, omega_f ~ 70 rad/s.
GOLAND_WING = WingProperties(
    semispan=6.096,
    chord=1.8288,
    mass_per_length=35.71,
    inertia_per_length=8.64,
    bending_stiffness=9.77e6,
    torsional_stiffness=0.987e6,
    ea_frac=0.33,
    cg_frac=0.43,
    surfaces=_goland_surfaces(),
    name="goland",
)
