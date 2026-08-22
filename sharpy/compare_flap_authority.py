"""Like-for-like comparison of control-surface effectiveness, AEGIS against UVLM.

The comparison has to account for geometry before it means anything. AEGIS uses a
two-dimensional sectional derivative from thin-airfoil theory, Cl_delta = 3.826
per radian for a hinge at 75 per cent chord. UVLM returns the whole-wing
CL_delta, and the flap covers only part of the span. Comparing the two numbers
directly -- which a first pass did -- gives a ratio of 0.133 that is pure
bookkeeping error.

Integrating the sectional value over the flapped span gives what the AEGIS plant
would report for the same wing:

    dCL/d(delta) = c * Cl_delta * b_flap / S

with ``b_flap`` the total flapped span and ``S`` the reference area. Whatever
remains after that is genuine three-dimensional physics: spanwise carryover and
the loss around the ends of a part-span flap.

Usage::

    python compare_flap_authority.py --uvlm-slope 0.508 --flap-fraction 0.25
"""

from __future__ import annotations

import argparse

# Thin-airfoil plain-flap derivative used by the AEGIS plant, hinge at 75% chord.
AEGIS_SECTIONAL_CL_DELTA = 3.826

SPAN = 2.0 * 6.096
CHORD = 1.8288


def main() -> None:
    args = _parse_args()

    area = SPAN * CHORD
    flapped_span = args.flap_fraction * SPAN
    aegis_wing_slope = AEGIS_SECTIONAL_CL_DELTA * CHORD * flapped_span / area

    print("geometry")
    print(f"  span                 {SPAN:.4f} m")
    print(f"  chord                {CHORD:.4f} m")
    print(f"  reference area       {area:.4f} m^2")
    print(f"  flapped span         {flapped_span:.4f} m "
          f"({100 * args.flap_fraction:.1f} per cent of span)")
    print()
    print("control effectiveness, whole wing")
    print(f"  AEGIS  dCL/d(delta)  {aegis_wing_slope:.4f} per rad "
          "(sectional value integrated over the flapped span)")
    print(f"  UVLM   dCL/d(delta)  {args.uvlm_slope:.4f} per rad")
    ratio = args.uvlm_slope / aegis_wing_slope
    print(f"  ratio UVLM / AEGIS   {ratio:.3f}")
    print()
    print(f"  AEGIS over-estimates control authority by a factor of {1 / ratio:.2f}.")
    print()
    print("  This is the expected direction. A part-span flap loses effectiveness to")
    print("  spanwise carryover and to the flow around its ends, neither of which a")
    print("  strip-theory model with a sectional derivative can represent.")
    print()
    print("  Consequence for the study: every controller, classical and learned, was")
    print("  evaluated on the same plant and therefore received the same inflated")
    print("  authority, so relative comparisons and ablations are unaffected. Absolute")
    print("  decay rates would not transfer unchanged to a UVLM plant.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvlm-slope", type=float, required=True,
                        help="whole-wing dCL/ddelta measured by UVLM, per rad")
    parser.add_argument("--flap-fraction", type=float, default=0.25,
                        help="fraction of span covered by control surfaces")
    return parser.parse_args()


if __name__ == "__main__":
    main()
