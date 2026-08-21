"""Find where the centralised LQR actually loses the wing.

Three studies, printed as tables and written as a figure:

1. **Off-design airspeed.** Gain synthesised at one speed, flown at others.
   Reports the airspeed margin of the fixed gain.
2. **Lost authority.** Same, with each surface (and each pair) jammed so its
   authority leaves the loop. Stability only -- jam *angle* is irrelevant here
   and the analysis does not pretend otherwise.
3. **Finite authority.** Time-domain rollouts with real travel and rate limits,
   sweeping jam angle and control-effort weight. This is where a jam angle bites,
   through saturation the eigenvalues cannot see.

Usage::

    python scripts/robustness_sweep.py --outdir media
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from aegis.analysis.robustness import (
    airspeed_margin,
    closed_loop_growth,
    saturated_growth,
)
from aegis.control.lqr import CentralizedLQR
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

DESIGN_RATIO = 1.10          # gain synthesised at 1.10 x open-loop flutter speed
EVAL_RATIOS = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
JAM_ANGLES_DEG = (0.0, 4.0, 8.0, 12.0, 15.0)
EFFORT_WEIGHTS = (2.0e2, 2.0e3, 2.0e4, 2.0e5)


def main() -> None:
    args = _parse_args()
    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point()
    design_speed = flutter.airspeed * DESIGN_RATIO
    controller = CentralizedLQR(model, design_speed)

    print(f"open-loop flutter: {flutter}")
    print(f"LQR designed at U = {design_speed:.1f} m/s (U/U_f = {DESIGN_RATIO})\n")

    _study_offdesign(model, controller, flutter.airspeed)
    _study_lost_authority(model, controller, flutter.airspeed)
    results = _study_finite_authority(model, flutter.airspeed)
    if args.outdir:
        _plot(results, Path(args.outdir) / "robustness_map.png", flutter.airspeed)


def _study_offdesign(model, controller, flutter_speed: float) -> None:
    print("=" * 74)
    print("1. OFF-DESIGN AIRSPEED  (fixed gain, all surfaces healthy)")
    print("=" * 74)
    print(f"{'U/U_f':>8}  {'U [m/s]':>9}  {'growth [1/s]':>13}  {'freq [rad/s]':>13}  verdict")
    for ratio in EVAL_RATIOS:
        point = closed_loop_growth(model, controller, flutter_speed * ratio)
        print(
            f"{ratio:8.2f}  {point.eval_speed:9.1f}  {point.growth_rate:13.3f}"
            f"  {point.frequency:13.2f}  {'stable' if point.stable else 'UNSTABLE'}"
        )
    margin = airspeed_margin(model, controller)
    _report_margin("all healthy", margin, flutter_speed)
    print()


def _study_lost_authority(model, controller, flutter_speed: float) -> None:
    print("=" * 74)
    print("2. LOST AUTHORITY  (surface jammed -> stops responding)")
    print("   Stability is independent of the angle it jams at: a held")
    print("   deflection is an affine input, so it moves trim, not the spectrum.")
    print("=" * 74)

    names = [s.name for s in model.wing.surfaces]
    failures: list[tuple[str, ...]] = [()]
    failures += [(n,) for n in names]
    failures += list(itertools.combinations(names, 2))

    print(f"{'jammed':>34}  {'margin U/U_f':>13}  {'growth @1.1':>12}  verdict")
    for jammed in failures:
        margin = airspeed_margin(model, controller, jammed=jammed)
        at_design = closed_loop_growth(
            model, controller, flutter_speed * DESIGN_RATIO, jammed=jammed
        )
        label = "none" if not jammed else " + ".join(jammed)
        ratio = "inf" if np.isinf(margin) else f"{margin / flutter_speed:.2f}"
        print(
            f"{label:>34}  {ratio:>13}  {at_design.growth_rate:12.3f}"
            f"  {'stable' if at_design.stable else 'UNSTABLE'}"
        )
    print()


def _study_finite_authority(model, flutter_speed: float) -> dict:
    print("=" * 74)
    print("3. FINITE AUTHORITY  (time domain: travel + rate saturation)")
    print("   Jam angle matters here -- healthy surfaces spend travel trimming it.")
    print("=" * 74)

    speed = flutter_speed * DESIGN_RATIO
    results = {
        "effort": EFFORT_WEIGHTS,
        "angles": JAM_ANGLES_DEG,
        "growth": [],
        "diverged": [],
    }

    print(f"{'effort':>9}  " + "  ".join(f"{a:>5.0f}deg" for a in JAM_ANGLES_DEG))
    for effort in EFFORT_WEIGHTS:
        controller = CentralizedLQR(model, speed, effort_weight=effort)
        row_growth, row_diverged = [], []
        cells = []
        for angle in JAM_ANGLES_DEG:
            growth, diverged, peak = saturated_growth(
                model,
                controller,
                speed,
                jam_angle=np.deg2rad(angle),
                jammed_surface="tab_outboard" if angle > 0 else None,
            )
            row_growth.append(growth)
            row_diverged.append(diverged)
            cells.append("DIVERGED" if diverged else f"{growth:+8.3f}")
            del peak
        results["growth"].append(row_growth)
        results["diverged"].append(row_diverged)
        print(f"{effort:9.0e}  " + "  ".join(f"{c:>8}" for c in cells))

    print("\n(numbers are energy growth rate in 1/s; negative = suppressed)")
    print()
    return results


def _report_margin(label: str, margin: float, flutter_speed: float) -> None:
    if np.isinf(margin):
        print(f"  airspeed margin ({label}): stable past 4x the flutter speed")
    else:
        print(
            f"  airspeed margin ({label}): {margin:.1f} m/s "
            f"= {margin / flutter_speed:.2f} U_f"
        )


def _plot(results: dict, path: Path, flutter_speed: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz.palette import THEME

    growth = np.asarray(results["growth"])
    diverged = np.asarray(results["diverged"])
    display = np.where(diverged, np.nan, growth)

    figure, axes = plt.subplots(figsize=(8.2, 4.6), dpi=110)
    figure.patch.set_facecolor(THEME["background"])
    axes.set_facecolor(THEME["panel"])

    limit = float(np.nanmax(np.abs(display))) if np.isfinite(display).any() else 1.0
    image = axes.imshow(
        display, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower"
    )
    for i, j in itertools.product(range(growth.shape[0]), range(growth.shape[1])):
        text = "div" if diverged[i, j] else f"{growth[i, j]:+.2f}"
        axes.text(
            j, i, text, ha="center", va="center", fontsize=8.5,
            color="#111111" if not diverged[i, j] else THEME["danger"],
            fontweight="bold" if diverged[i, j] else "normal",
        )

    axes.set_xticks(range(len(results["angles"])))
    axes.set_xticklabels([f"{a:.0f}°" for a in results["angles"]])
    axes.set_yticks(range(len(results["effort"])))
    axes.set_yticklabels([f"{e:.0e}" for e in results["effort"]])
    axes.set_xlabel("outboard tab jam angle", color=THEME["text_dim"])
    axes.set_ylabel("LQR control-effort weight", color=THEME["text_dim"])
    axes.set_title(
        "Centralised LQR energy growth rate [1/s] under jam and control cost\n"
        f"U = {DESIGN_RATIO:.2f} U_f = {flutter_speed * DESIGN_RATIO:.0f} m/s,"
        " time domain with travel and rate limits",
        color=THEME["text"], fontsize=10.5, loc="left",
    )
    axes.tick_params(colors=THEME["text_dim"])
    bar = figure.colorbar(image, ax=axes)
    bar.ax.tick_params(colors=THEME["text_dim"])

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"wrote {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="media", help="where to write the figure")
    return parser.parse_args()


if __name__ == "__main__":
    main()
