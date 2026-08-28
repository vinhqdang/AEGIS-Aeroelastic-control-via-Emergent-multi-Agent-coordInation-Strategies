"""Assemble the manuscript's results: tables, figures, and a machine-readable dump.

Runs the classical baseline ladder, loads the trained variants from a run
directory, establishes which evaluation cells are feasible at all, and writes
everything to ``paper/figures`` and ``paper/results.json``.

The feasibility pass matters. ``BatchedJamAwareLQR`` gets the full plant state
*and* oracle knowledge of which surface failed; a cell it cannot hold is
infeasible for any controller with these actuators, and reporting it as a failure
of a learned policy would be misleading. Such cells are marked, not scored.

Usage::

    python scripts/results.py --runs runs/sweep1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis.control.batched_baselines import (
    BatchedDecentralizedLQG,
    BatchedJamAwareLQR,
    BatchedLocalFeedback,
    BatchedLQG,
    BatchedLQR,
    BatchedRobustDecentralizedLQG,
    BatchedScheduledLQR,
)
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import evaluate, standard_conditions
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

EVAL_EPISODES = 48
SPEED_RANGE_RATIOS = (1.0, 1.5)

BASELINE_LABELS = {
    "local_fb": "Local rate feedback",
    "lqr_fixed": "LQR, fixed gain",
    "lqr_sched": "LQR, gain-scheduled",
    "lqg_local": "LQG, centralised, local sensing",
    "dlqg_local": "LQG, fully decentralised",
    "dlqg_robust": "LQG, decentralised, effort-reweighted for margin",
    "lqr_oracle": "LQR, fault-aware oracle",
}


@dataclass
class Row:
    """One controller's scores across the evaluation grid."""

    key: str
    label: str
    kind: str  # "classical", "oracle", or "learned"
    divergence: list[float]
    suppression: list[float]
    rms_deflection: list[float]

    def stabilised(self, feasible: list[bool]) -> tuple[int, int]:
        """Cells fully stabilised, counted only over feasible cells."""
        total = sum(feasible)
        good = sum(
            1
            for index, ok in enumerate(feasible)
            if ok and self.divergence[index] == 0.0
        )
        return good, total

    def healthy_suppression(self, conditions) -> float:
        values = [
            self.suppression[i]
            for i, c in enumerate(conditions)
            if c.jam_surface < 0 and self.divergence[i] == 0.0
        ]
        return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    config = EnvConfig(episode_duration=2.0, speed_ratio_range=SPEED_RANGE_RATIOS)
    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point().airspeed
    speed_range = (flutter * SPEED_RANGE_RATIOS[0], flutter * SPEED_RANGE_RATIOS[1])
    conditions = standard_conditions()

    print(f"flutter speed {flutter:.2f} m/s\nevaluating {len(conditions)} conditions "
          f"x {EVAL_EPISODES} episodes\n")

    rows: list[Row] = []
    controllers = {
        "local_fb": BatchedLocalFeedback(),
        "lqr_fixed": BatchedLQR(model, flutter * 1.1, config.control_dt),
        "lqr_sched": BatchedScheduledLQR(model, speed_range, config.control_dt),
        "lqg_local": BatchedLQG(model, speed_range, config.control_dt),
        "dlqg_local": BatchedDecentralizedLQG(model, speed_range, config.control_dt),
        "dlqg_robust": BatchedRobustDecentralizedLQG(model, speed_range, config.control_dt),
        "lqr_oracle": BatchedJamAwareLQR(model, speed_range, config.control_dt),
    }
    for key, controller in controllers.items():
        results = evaluate(controller, config, conditions, n_episodes=EVAL_EPISODES)
        kind = "oracle" if key == "lqr_oracle" else "classical"
        rows.append(
            Row(
                key,
                BASELINE_LABELS[key],
                kind,
                [r.divergence_rate for r in results],
                [r.suppression_rate for r in results],
                [r.rms_deflection for r in results],
            )
        )
        print(f"  {key:12s} done")

    # Feasibility from the fault-aware oracle.
    oracle = next(r for r in rows if r.key == "lqr_oracle")
    feasible = [d == 0.0 for d in oracle.divergence]
    print(f"\nfeasible cells: {sum(feasible)}/{len(feasible)}")
    for index, condition in enumerate(conditions):
        if not feasible[index]:
            print(f"  INFEASIBLE (oracle diverges {oracle.divergence[index]*100:.0f}%): "
                  f"{condition.describe()}")

    learned, curves = _load_learned(Path(args.runs), conditions)
    rows.extend(learned)

    print()
    print(_text_table(rows, conditions, feasible))

    _write_json(outdir / "results.json", rows, conditions, feasible, curves, flutter)
    _write_latex(outdir / "results_table.tex", rows, conditions, feasible)
    _figures(outdir / "figures", rows, conditions, feasible, curves)
    print(f"\nwrote {outdir/'results.json'}, {outdir/'results_table.tex'}, "
          f"and figures in {outdir/'figures'}")


def _load_learned(runs: Path, conditions) -> tuple[list[Row], dict]:
    """Load trained variants, averaging over seeds where several exist."""
    if not runs.exists():
        print(f"(no run directory {runs}; classical baselines only)")
        return [], {}

    grouped: dict[str, list[dict]] = {}
    for path in sorted(runs.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(payload["variant"]["name"], []).append(payload)

    rows, curves = [], {}
    for name, payloads in grouped.items():
        divergence = np.mean(
            [[r["divergence_rate"] for r in p["results"]] for p in payloads], axis=0
        )
        suppression = np.mean(
            [[r["suppression_rate"] for r in p["results"]] for p in payloads], axis=0
        )
        rms = np.mean(
            [[r["rms_deflection"] for r in p["results"]] for p in payloads], axis=0
        )
        rows.append(
            Row(name, name, "learned", list(divergence), list(suppression), list(rms))
        )
        curves[name] = {
            "steps": payloads[0]["log"]["steps"],
            "mean_energy": np.mean(
                [p["log"]["mean_energy"] for p in payloads], axis=0
            ).tolist(),
            "divergence_rate": np.mean(
                [p["log"]["divergence_rate"] for p in payloads], axis=0
            ).tolist(),
            "n_seeds": len(payloads),
            "note": payloads[0]["variant"]["note"],
        }
    del conditions
    return rows, curves


def _cell(row: Row, index: int, feasible: bool) -> str:
    if not feasible:
        return "  --  "
    if row.divergence[index] > 0.0:
        return f"D{row.divergence[index]*100:3.0f}%"
    return f"{row.suppression[index]:6.2f}"


def _text_table(rows: list[Row], conditions, feasible: list[bool]) -> str:
    width = 8
    header = f"{'condition':>26}" + "".join(f"{r.key[:width]:>{width+1}}" for r in rows)
    lines = [header, "-" * len(header)]
    for index, condition in enumerate(conditions):
        cells = "".join(f"{_cell(r, index, feasible[index]):>{width+1}}" for r in rows)
        lines.append(f"{condition.describe():>26}{cells}")
    lines.append("-" * len(header))
    summary = f"{'stabilised / feasible':>26}"
    for row in rows:
        good, total = row.stabilised(feasible)
        summary += f"{f'{good}/{total}':>{width+1}}"
    lines.append(summary)
    healthy = f"{'healthy suppression':>26}"
    for row in rows:
        healthy += f"{row.healthy_suppression(conditions):>{width+1}.2f}"
    lines.append(healthy)
    lines.append("")
    lines.append("cells: energy decay rate [1/s], negative = suppressed;")
    lines.append(
        "       Dnn% = diverged in nn% of episodes;"
        "  --  = infeasible for any controller"
    )
    return "\n".join(lines)


def _write_json(path: Path, rows, conditions, feasible, curves, flutter_speed) -> None:
    path.write_text(
        json.dumps(
            {
                "flutter_speed": flutter_speed,
                "conditions": [
                    {
                        "label": c.describe(),
                        "speed_ratio": c.speed_ratio,
                        "jam_surface": c.jam_surface,
                        "jam_angle_deg": float(np.rad2deg(c.jam_angle)),
                        "feasible": feasible[i],
                    }
                    for i, c in enumerate(conditions)
                ],
                "controllers": [
                    {
                        "key": r.key,
                        "label": r.label,
                        "kind": r.kind,
                        "divergence": r.divergence,
                        "suppression": r.suppression,
                        "rms_deflection": r.rms_deflection,
                        "stabilised": r.stabilised(feasible)[0],
                        "feasible_cells": r.stabilised(feasible)[1],
                        "healthy_suppression": r.healthy_suppression(conditions),
                    }
                    for r in rows
                ],
                "curves": curves,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_latex(path: Path, rows, conditions, feasible) -> None:
    lines = [
        r"\begin{tabular}{l" + "r" * len(rows) + "}",
        r"\toprule",
        "Condition & " + " & ".join(r.key.replace("_", r"\_") for r in rows) + r" \\",
        r"\midrule",
    ]
    for index, condition in enumerate(conditions):
        cells = []
        for row in rows:
            if not feasible[index]:
                cells.append("--")
            elif row.divergence[index] > 0.0:
                cells.append(f"D{row.divergence[index]*100:.0f}\\%")
            else:
                cells.append(f"{row.suppression[index]:.2f}")
        lines.append(
            condition.describe().replace("#", r"\#") + " & " + " & ".join(cells) + r" \\"
        )
    lines.append(r"\midrule")
    counts = []
    for row in rows:
        good, total = row.stabilised(feasible)
        counts.append(f"{good}/{total}")
    lines.append("Stabilised & " + " & ".join(counts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _figures(outdir: Path, rows, conditions, feasible, curves) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz.paper import (
        GRID,
        INK,
        INK_MUTED,
        SUPPRESSION_CMAP,
        controller_color,
        symmetric_limit,
        use_paper_style,
    )

    use_paper_style()
    _comparison_figure(
        outdir / "fig_comparison.pdf", rows, conditions, feasible,
        plt, SUPPRESSION_CMAP, symmetric_limit, INK, INK_MUTED,
    )
    if curves:
        _learning_figure(outdir / "fig_learning.pdf", curves, plt, controller_color, INK_MUTED)
        _ablation_figure(
            outdir / "fig_ablation.pdf", rows, conditions, feasible, plt,
            controller_color, INK, INK_MUTED, GRID,
        )


def _comparison_figure(
    path, rows, conditions, feasible, plt, cmap, symmetric_limit, ink, ink_muted
) -> None:
    """Controller x condition grid of the energy decay rate.

    Conditions that no controller can hold are omitted from the plot; they remain
    in the results table and are named in the caption.
    """
    shown = [j for j in range(len(conditions)) if feasible[j]]
    conditions = [conditions[j] for j in shown]
    feasible = [True] * len(shown)
    rows = [
        Row(
            r.key, r.label, r.kind,
            [r.divergence[j] for j in shown],
            [r.suppression[j] for j in shown],
            [r.rms_deflection[j] for j in shown],
        )
        for r in rows
    ]

    matrix = np.full((len(rows), len(conditions)), np.nan)
    for i, row in enumerate(rows):
        for j in range(len(conditions)):
            if row.divergence[j] == 0.0:
                matrix[i, j] = row.suppression[j]

    limit = symmetric_limit(matrix)
    figure, axes = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 2.1))
    axes.grid(False)
    image = axes.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")

    for i, row in enumerate(rows):
        for j in range(len(conditions)):
            if not feasible[j]:
                axes.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f0",
                                  edgecolor="white", linewidth=1.4, zorder=2)
                )
                axes.text(j, i, "n/a", ha="center", va="center", color=ink_muted,
                          fontsize=7, zorder=3)
            elif row.divergence[j] > 0.0:
                axes.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f7eceb",
                                  edgecolor="white", linewidth=1.4, hatch="////",
                                  zorder=2)
                )
                axes.text(j, i, f"{row.divergence[j]*100:.0f}%", ha="center",
                          va="center", color="#b03a2e", fontsize=7, zorder=3)
            else:
                axes.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          color=ink, fontsize=7, zorder=3)

    axes.set_xticks(range(len(conditions)))
    axes.set_xticklabels(
        [
            c.describe().replace("Uf ", "$U_f$\n").replace("jam#2@", "jam ")
            for c in conditions
        ],
        fontsize=7,
    )
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([r.label for r in rows], fontsize=8)
    axes.set_title(
        "Energy decay rate [1/s] by controller and flight condition\n"
        "negative = suppressed; hatched = diverged in the stated fraction of episodes",
        loc="left", fontsize=9,
    )
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("decay rate [1/s]", fontsize=8)
    figure.savefig(path)
    plt.close(figure)


def _learning_figure(path, curves, plt, controller_color, ink_muted) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for index, (name, curve) in enumerate(sorted(curves.items())):
        color = controller_color(index % 6)
        left.plot(np.asarray(curve["steps"]) / 1e6, curve["mean_energy"],
                  color=color, label=name)
        right.plot(np.asarray(curve["steps"]) / 1e6,
                   100.0 * np.asarray(curve["divergence_rate"]), color=color, label=name)
    left.set_yscale("log")
    left.set_xlabel("environment steps [millions]")
    left.set_ylabel("mean structural energy [J]")
    left.set_title("Training: energy", loc="left")
    right.set_xlabel("environment steps [millions]")
    right.set_ylabel("divergence rate [%]")
    right.set_title("Training: divergence", loc="left")
    right.legend(loc="upper right", ncol=1, fontsize=7)
    figure.savefig(path)
    plt.close(figure)


def _ablation_figure(
    path, rows, conditions, feasible, plt, controller_color, ink, ink_muted, grid
) -> None:
    learned = [r for r in rows if r.kind == "learned"]
    classical = [r for r in rows if r.kind != "learned"]
    ordered = classical + learned
    if not ordered:
        return

    counts = [r.stabilised(feasible)[0] for r in ordered]
    total = ordered[0].stabilised(feasible)[1]
    healthy = [r.healthy_suppression(conditions) for r in ordered]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True)
    positions = np.arange(len(ordered))
    colors = ["#9aa4ae" if r.kind != "learned" else controller_color(0) for r in ordered]
    for index, row in enumerate(ordered):
        if row.kind == "oracle":
            colors[index] = controller_color(1)

    top.bar(positions, counts, width=0.62, color=colors)
    top.axhline(total, color=grid, linestyle="--", linewidth=0.8)
    top.text(len(ordered) - 0.4, total, f"  all {total} feasible cells",
             va="center", color=ink_muted, fontsize=7)
    for index, value in enumerate(counts):
        top.text(index, value + 0.08, str(value), ha="center", color=ink, fontsize=7.5)
    top.set_ylabel("cells stabilised")
    top.set_ylim(0, total + 0.9)
    top.set_title("Robustness: feasible cells held without divergence", loc="left")

    bottom.bar(positions, healthy, width=0.62, color=colors)
    for index, value in enumerate(healthy):
        if np.isfinite(value):
            bottom.text(index, value - 0.18, f"{value:.2f}", ha="center",
                        color=ink, fontsize=7.5, va="top")
    bottom.set_ylabel("decay rate [1/s]")
    bottom.set_title(
        "Nominal performance: mean decay rate on healthy conditions (lower is better)",
        loc="left",
    )
    bottom.set_xticks(positions)
    bottom.set_xticklabels([r.label if r.kind != "learned" else r.key for r in ordered],
                          rotation=30, ha="right", fontsize=7.5)
    figure.savefig(path)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/sweep1")
    parser.add_argument("--outdir", default="paper")
    return parser.parse_args()


if __name__ == "__main__":
    main()
