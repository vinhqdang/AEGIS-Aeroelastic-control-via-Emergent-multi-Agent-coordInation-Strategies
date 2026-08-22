"""Reduce SHARPy's velocity-analysis output to one flutter point.

The file AsymptoticStability writes carries every retained eigenvalue at every
swept speed, as (speed, real, imag) rows. Treating those rows as one per speed --
which a first pass at this did -- produces nonsense: the crossing search walks
across eigenvalues rather than across speeds, and reports the frequency of
whichever high-frequency aerodynamic mode happened to sit at the boundary.

This groups by speed, takes the largest real part at each, and reports the
frequency of that eigenvalue. A physical band filter is applied because the UVLM
carries fast, heavily damped wake modes whose frequencies are two orders of
magnitude above the structural ones and which are not what flutter means here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AEGIS_FLUTTER_SPEED = 137.35
AEGIS_FLUTTER_OMEGA = 69.34

# Structural modes of this wing run from about 48 to 360 rad/s. Anything far
# above that is a wake mode, not a flutter mechanism.
MAX_PHYSICAL_OMEGA = 600.0


def main() -> None:
    args = _parse_args()
    rows = _load(Path(args.path))
    if rows.size == 0:
        raise SystemExit(f"no numeric rows found in {args.path}")

    speeds = np.unique(rows[:, 0])
    growth = np.zeros(speeds.size)
    omega = np.zeros(speeds.size)

    for index, speed in enumerate(speeds):
        block = rows[rows[:, 0] == speed]
        physical = block[np.abs(block[:, 2]) <= MAX_PHYSICAL_OMEGA]
        if physical.size == 0:
            physical = block
        worst = int(np.argmax(physical[:, 1]))
        growth[index] = physical[worst, 1]
        omega[index] = abs(physical[worst, 2])

    print(f"{'speed':>9} {'max Re':>12} {'omega':>10}")
    for speed, rate, frequency in zip(speeds, growth, omega):
        flag = "  <-- unstable" if rate > 0 else ""
        print(f"{speed:9.2f} {rate:12.4f} {frequency:10.2f}{flag}")

    flutter_speed, flutter_omega = _crossing(speeds, growth, omega)
    if flutter_speed is None:
        print("\nno crossing inside the swept range")
        return

    difference = 100.0 * (AEGIS_FLUTTER_SPEED - flutter_speed) / flutter_speed
    print(f"\n  SHARPy UVLM   flutter = {flutter_speed:.2f} m/s at {flutter_omega:.1f} rad/s")
    print(f"  AEGIS  strip  flutter = {AEGIS_FLUTTER_SPEED:.2f} m/s "
          f"at {AEGIS_FLUTTER_OMEGA:.1f} rad/s")
    print(f"  strip theory is {difference:+.1f} % relative to UVLM")
    print("\n  Strip theory has no tip relief, so it over-predicts loading near the")
    print("  tip and should under-predict the flutter speed. A negative percentage")
    print("  is therefore the physically expected direction.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "speeds": speeds.tolist(),
                    "max_real": growth.tolist(),
                    "omega_rad_s": omega.tolist(),
                    "uvlm_flutter_speed": flutter_speed,
                    "uvlm_flutter_omega": flutter_omega,
                    "aegis_flutter_speed": AEGIS_FLUTTER_SPEED,
                    "aegis_flutter_omega": AEGIS_FLUTTER_OMEGA,
                    "aegis_relative_percent": difference,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


def _load(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            values = [float(p) for p in parts[:3]]
        except ValueError:
            continue
        rows.append(values)
    return np.asarray(rows) if rows else np.asarray([])


def _crossing(speeds, growth, omega):
    for index in range(len(speeds) - 1):
        if growth[index] <= 0.0 < growth[index + 1]:
            span = growth[index + 1] - growth[index]
            weight = -growth[index] / span if span else 0.0
            speed = speeds[index] + weight * (speeds[index + 1] - speeds[index])
            return float(speed), float(omega[index + 1])
    return None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="velocity analysis .dat written by SHARPy")
    parser.add_argument("--out", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
