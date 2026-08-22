"""Report how much of the wing the SHARPy control surfaces actually cover.

Needed to make the flap-effectiveness comparison meaningful. AEGIS quotes a two
dimensional sectional derivative, Cl_delta = 3.826 per radian, whereas a UVLM run
returns the whole-wing CL_delta. Those differ by the fraction of span the flap
occupies and by finite-span effects, so comparing them directly is meaningless --
a first attempt did exactly that and produced a ratio of 0.133, which is a
bookkeeping error rather than an aerodynamic finding.

This reads the generated .aero.h5 and reports the spanwise and chordwise extent
of the control surfaces, so the sectional value can be integrated over the same
region for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    args = _parse_args()
    # An empty --aero-file becomes Path("."), which exists as a directory, so
    # the emptiness has to be tested before the existence.
    path = Path(args.aero_file) if args.aero_file else Path("__unset__")
    if not path.is_file():
        candidates = sorted(Path(args.search_root).rglob("*.aero.h5"))
        if not candidates:
            raise SystemExit(f"no .aero.h5 found under {args.search_root}")
        path = candidates[0]
    print(f"reading {path}")

    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        print("datasets:", keys)
        control_surface = np.asarray(handle["control_surface"]) if "control_surface" in handle else None
        chord = np.asarray(handle["chord"]) if "chord" in handle else None
        surface_distribution = (
            np.asarray(handle["surface_distribution"])
            if "surface_distribution" in handle
            else None
        )

    if control_surface is None:
        raise SystemExit("no control_surface dataset in the aero file")

    print(f"\ncontrol_surface array shape {control_surface.shape}")
    unique = np.unique(control_surface)
    print("unique values:", unique.tolist(), "  (-1 means no control surface)")

    total_panels = control_surface.size
    for value in unique:
        if value < 0:
            continue
        count = int(np.sum(control_surface == value))
        print(f"  surface {value}: {count} of {total_panels} element-nodes "
              f"({100.0 * count / total_panels:.1f} per cent)")

    flapped = int(np.sum(control_surface >= 0))
    fraction = flapped / total_panels
    print(f"\nflapped fraction of the wing (by element-node count): {fraction:.3f}")

    if chord is not None:
        print(f"chord array: min {chord.min():.4f} max {chord.max():.4f}")
    if surface_distribution is not None:
        print("surface_distribution unique:", np.unique(surface_distribution).tolist())

    print("\nUse this fraction to integrate the AEGIS sectional Cl_delta over the")
    print("same span before comparing with the UVLM whole-wing CL_delta.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aero-file", default="")
    parser.add_argument("--search-root", default="/work/flap_cases")
    return parser.parse_args()


if __name__ == "__main__":
    main()
