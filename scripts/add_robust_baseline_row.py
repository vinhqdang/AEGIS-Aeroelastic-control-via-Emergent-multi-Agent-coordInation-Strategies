"""Splice the dlqg_robust classical baseline into the existing results files.

The original ``runs/`` training-run archive that produced ``paper/results.json``
is not present in this checkout (it is gitignored), so ``scripts/results.py``
cannot be re-run end to end without retraining every learned variant. The new
classical baseline added in response to review (``dlqg_robust``, see
``aegis.control.batched_baselines.BatchedRobustDecentralizedLQG``) needs no
training data at all -- it is a deterministic classical design -- so this
script evaluates it with the exact same methodology as ``scripts/results.py``
and splices the one new row into the existing ``results.json`` and
``results_table.tex`` in place, leaving every other (learned) row untouched.

Usage::

    python scripts/add_robust_baseline_row.py
"""

from __future__ import annotations

import json
from pathlib import Path

from aegis.control.batched_baselines import BatchedRobustDecentralizedLQG
from aegis.envs.flutter_marl import EnvConfig
from aegis.learning.evaluate import evaluate, standard_conditions
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING

EVAL_EPISODES = 48  # matches scripts/results.py
SPEED_RANGE_RATIOS = (1.0, 1.5)
KEY = "dlqg_robust"
LABEL = "LQG, decentralised, effort-reweighted for margin"
AFTER_KEY = "dlqg_local"  # insert immediately after this row, both places


def main() -> None:
    results_path = Path("paper/results.json")
    table_path = Path("paper/results_table.tex")
    data = json.loads(results_path.read_text(encoding="utf-8"))

    if any(c["key"] == KEY for c in data["controllers"]):
        print(f"{KEY} already present in {results_path}; nothing to do")
        return

    config = EnvConfig(episode_duration=2.0, speed_ratio_range=SPEED_RANGE_RATIOS)
    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point().airspeed
    speed_range = (flutter * SPEED_RANGE_RATIOS[0], flutter * SPEED_RANGE_RATIOS[1])
    conditions = standard_conditions()
    feasible = [c["feasible"] for c in data["conditions"]]

    controller = BatchedRobustDecentralizedLQG(model, speed_range, config.control_dt)
    results = evaluate(controller, config, conditions, n_episodes=EVAL_EPISODES)
    divergence = [r.divergence_rate for r in results]
    suppression = [r.suppression_rate for r in results]
    rms = [r.rms_deflection for r in results]

    stabilised = sum(
        1 for i, ok in enumerate(feasible) if ok and divergence[i] == 0.0
    )
    feasible_cells = sum(feasible)
    healthy_values = [
        suppression[i]
        for i, c in enumerate(data["conditions"])
        if c["jam_surface"] < 0 and divergence[i] == 0.0
    ]
    healthy_suppression = (
        sum(healthy_values) / len(healthy_values) if healthy_values else float("nan")
    )

    new_entry = {
        "key": KEY,
        "label": LABEL,
        "kind": "classical",
        "divergence": divergence,
        "suppression": suppression,
        "rms_deflection": rms,
        "stabilised": stabilised,
        "feasible_cells": feasible_cells,
        "healthy_suppression": healthy_suppression,
    }

    keys = [c["key"] for c in data["controllers"]]
    insert_at = keys.index(AFTER_KEY) + 1
    data["controllers"].insert(insert_at, new_entry)
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"inserted {KEY} into {results_path} after {AFTER_KEY}")
    print(
        f"  stabilised {stabilised}/{feasible_cells}, "
        f"healthy suppression {healthy_suppression:+.3f} 1/s"
    )

    _update_table(table_path, data, insert_at)
    print(f"inserted {KEY} column into {table_path}")


def _update_table(table_path: Path, data: dict, insert_at: int) -> None:
    lines = table_path.read_text(encoding="utf-8").splitlines()
    conditions = data["conditions"]
    controllers = data["controllers"]
    new = controllers[insert_at]

    out = []
    condition_index = 0
    for line in lines:
        if line.startswith(r"\begin{tabular}{"):
            out.append(line[:-1] + "r}" if line.endswith("}") else line + "r")
        elif line.startswith("Condition & "):
            parts = line.rstrip(r" \\").split(" & ")
            parts.insert(insert_at + 1, new["key"].replace("_", r"\_"))
            out.append(" & ".join(parts) + r" \\")
        elif line.startswith("Stabilised & "):
            parts = line.rstrip(r" \\").split(" & ")
            parts.insert(insert_at + 1, f"{new['stabilised']}/{new['feasible_cells']}")
            out.append(" & ".join(parts) + r" \\")
        elif r" \\" in line and " & " in line and not line.startswith(
            (r"\toprule", r"\midrule", r"\bottomrule")
        ):
            condition = conditions[condition_index]
            if not condition["feasible"]:
                cell = "--"
            elif new["divergence"][condition_index] > 0.0:
                cell = f"D{new['divergence'][condition_index] * 100:.0f}\\%"
            else:
                cell = f"{new['suppression'][condition_index]:.2f}"
            parts = line.rstrip(r" \\").split(" & ")
            parts.insert(insert_at + 1, cell)
            out.append(" & ".join(parts) + r" \\")
            condition_index += 1
        else:
            out.append(line)
    table_path.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
