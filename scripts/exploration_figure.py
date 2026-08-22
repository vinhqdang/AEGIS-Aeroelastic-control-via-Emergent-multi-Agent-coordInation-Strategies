"""Suppression performance against exploration scale.

The central empirical finding of the study. Every learned variant plateaued in a
narrow band regardless of credit signal, communication, action parameterisation,
memory, auxiliary supervision or base controller. Changing one thing -- the
exploration standard deviation -- moved performance several-fold.

The reason is a scale mismatch that can be read straight off the axis. A
controller that achieves the classical result uses about 0.1 degrees RMS of
deflection; an exploration sigma of 0.30 in normalised action units is 4.5
degrees. Where the noise sits far to the right of the useful amplitude, the
policy cannot experience the actuation that works, and no credit decomposition
can recover a signal buried underneath it.

Usage::

    python scripts/exploration_figure.py --out paper/figures/fig_exploration.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Each run directory was trained at one exploration sigma. The mapping lives here
# because the earliest runs predate recording it in the payload; newer runs carry
# it and are cross-checked against this table on load.
SIGMA_BY_DIRECTORY = {
    "runs/v2": 0.30,
    "runs/sigma_015": 0.15,
    "runs/fine": 0.05,
    "runs/sigma_002": 0.02,
    "runs/finer": 0.01,
}
VARIANT = "residual_only"

#: RMS deflection used by the classical controllers, as a fraction of travel,
#: converted to the same normalised action units as sigma.
USEFUL_AMPLITUDE = 0.007      # centralised LQG, 0.7 per cent of travel
BASELINE_DECENTRALISED = -2.06
BASELINE_CENTRALISED = -5.52


def main() -> None:
    args = _parse_args()
    points = _collect(Path(args.root))
    if not points:
        raise SystemExit("no runs found; train some first")

    for sigma, values in sorted(points.items()):
        print(
            f"  sigma = {sigma:5.2f}  ({np.rad2deg(sigma * 0.2618):5.2f} deg)   "
            f"decay = {np.mean(values):+7.3f} +- {np.std(values):.3f} 1/s   "
            f"n = {len(values)}"
        )
    _plot(points, Path(args.out))
    print(f"\nwrote {args.out}")


def _collect(root: Path) -> dict[float, list[float]]:
    points: dict[float, list[float]] = {}
    for directory, table_sigma in SIGMA_BY_DIRECTORY.items():
        for path in sorted((root / directory).glob(f"{VARIANT}_s*.json")):
            sigma = table_sigma
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Prefer the value the run recorded; the table is only a fallback
            # for runs that predate recording it. Disagree by more than a few
            # per cent and the mapping is wrong, which is worth stopping for.
            recorded = payload.get("initial_sigma")
            if recorded is not None:
                if abs(recorded - sigma) / sigma > 0.05:
                    raise SystemExit(
                        f"{path} was trained at sigma {recorded:.4f}, but the "
                        f"directory table says {sigma}"
                    )
                sigma = round(recorded, 4)
            healthy = [
                r["suppression_rate"]
                for r in payload["results"]
                if r["condition"]["jam_surface"] < 0 and r["divergence_rate"] == 0.0
            ]
            if healthy:
                points.setdefault(sigma, []).append(float(np.mean(healthy)))
    return points


def _plot(points: dict[float, list[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz.paper import INK_MUTED, controller_color, use_paper_style

    use_paper_style()
    figure, axes = plt.subplots(figsize=(5.8, 3.7))

    sigmas = np.asarray(sorted(points))
    means = np.asarray([np.mean(points[s]) for s in sigmas])
    spread = np.asarray([np.std(points[s]) for s in sigmas])

    # More negative is better, so the axis is inverted: up is better, which is
    # what a reader assumes without being told.
    axes.invert_yaxis()

    axes.axhline(BASELINE_DECENTRALISED, color=INK_MUTED, lw=1.0, ls=":")
    axes.text(
        sigmas.min(), BASELINE_DECENTRALISED, "decentralised LQG (same information structure)",
        va="bottom", ha="left", color=INK_MUTED, fontsize=7.5,
    )

    axes.axvline(USEFUL_AMPLITUDE, color="#b03a2e", lw=1.0, alpha=0.75)
    axes.text(
        USEFUL_AMPLITUDE * 1.18, means.min(),
        "deflection amplitude the\nclassical controller uses",
        color="#b03a2e", fontsize=7.5, va="bottom",
    )

    axes.errorbar(
        sigmas, means, yerr=spread, fmt="o-", color=controller_color(3),
        capsize=3, lw=1.7, label="learned policy, mean of 2 seeds",
    )
    axes.set_xscale("log")
    axes.set_xlim(USEFUL_AMPLITUDE * 0.8, sigmas.max() * 1.5)
    axes.set_ylim(-0.4, BASELINE_DECENTRALISED - 0.25)
    axes.set_xlabel("initial exploration $\\sigma$  [normalised action units]")
    axes.set_ylabel("energy decay rate  [1/s]\nbetter $\\longrightarrow$")
    axes.set_title(
        "Performance is set by the exploration scale, "
        "not by the credit signal or the architecture",
        loc="left",
    )
    axes.legend(loc="lower left", fontsize=7.5)
    figure.savefig(path)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="paper/figures/fig_exploration.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
