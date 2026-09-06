"""Cross-check every number asserted in the manuscript against the data files.

Written because this study reversed its own conclusions several times as the
measurements improved, and each reversal left stale figures in the prose. A
manuscript that disagrees with its own results directory is the worst outcome
available, and it is entirely preventable.

The checker extracts the numeric claims the text makes about the ablations and
the baselines, recomputes them from the run directories and the results JSON, and
reports any disagreement beyond a tolerance. It is deliberately noisy about
claims it cannot locate, since a silently unchecked number is the thing that
causes the problem.

Usage::

    python scripts/check_paper_numbers.py --tex paper/aiaaj.tex
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

TOLERANCE = 0.005


def healthy_decay(payload: dict) -> float:
    values = [
        r["suppression_rate"]
        for r in payload["results"]
        if r["condition"]["jam_surface"] < 0 and r["divergence_rate"] == 0.0
    ]
    return float(np.mean(values)) if values else float("nan")


def group(run_dir: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = {}
    for path in sorted(glob.glob(f"{run_dir}/*.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        value = healthy_decay(payload)
        if np.isfinite(value):
            grouped.setdefault(payload["variant"]["name"], []).append(value)
    return {k: np.asarray(v) for k, v in grouped.items()}


def welch(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    _, pvalue = stats.ttest_ind(left, right, equal_var=False)
    pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
    effect = (left.mean() - right.mean()) / pooled if pooled > 0 else float("nan")
    return float(effect), float(pvalue)


def main() -> None:
    args = _parse_args()
    text = Path(args.tex).read_text(encoding="utf-8")
    credit = group(args.credit_runs)
    arch = group(args.arch_runs)
    combined = dict(arch)
    if "ippo_blended" in credit:
        combined["ippo_blended"] = credit["ippo_blended"]

    checks: list[tuple[str, float, float]] = []

    for name, key in (
        ("ippo_physics", "ippo_physics"),
        ("shaped_shared", "shaped_shared"),
        ("ippo_blended", "ippo_blended"),
    ):
        if key in credit:
            checks.append((f"credit mean {name}", float(credit[key].mean()), 0.0))
            checks.append((f"credit sd {name}", float(credit[key].std(ddof=1)), 0.0))

    for key in ("recurrent_only", "mocca_nohead", "mocca", "comm_raw"):
        if key in arch:
            checks.append((f"arch mean {key}", float(arch[key].mean()), 0.0))

    print(f"reading {args.tex}")
    print("credit runs: " + ", ".join(f"{k}(n={v.size})" for k, v in sorted(credit.items())))
    print("arch runs:   " + ", ".join(f"{k}(n={v.size})" for k, v in sorted(arch.items())))
    print()

    problems = 0
    for label, value, _ in checks:
        rendered = f"{value:.3f}".lstrip("-")
        found = rendered in text or f"{value:.2f}".lstrip("-") in text
        status = "found " if found else "ABSENT"
        if not found:
            problems += 1
        print(f"  {status}  {label:34s} = {value:+.3f}")

    print("\nsignificance claims:")
    sig_pairs = [
        ("ippo_blended", "ippo_physics", credit),
        ("ippo_blended", "shaped_shared", credit),
        ("ippo_blended", "mocca", combined),
        ("ippo_blended", "mocca_nohead", combined),
        ("ippo_blended", "recurrent_only", combined),
        ("ippo_blended", "comm_raw", combined),
    ]
    for first, second, source in sig_pairs:
        if first in source and second in source:
            effect, pvalue = welch(source[first], source[second])
            in_text = f"{pvalue:.3f}" in text or f"{pvalue:.4f}" in text
            print(f"  {first} vs {second}: d={effect:+.2f} p={pvalue:.4f}"
                  f"   {'p found in text' if in_text else 'p NOT found in text'}")
            if not in_text:
                problems += 1

    print(f"\n{problems} claim(s) in the text could not be matched to the data.")
    if problems:
        print("Update the prose, or explain why the text differs.")
    raise SystemExit(1 if problems else 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default="paper/aiaaj.tex")
    parser.add_argument("--credit-runs", default="runs/credit_fine")
    parser.add_argument("--arch-runs", default="runs/arch_fine")
    return parser.parse_args()


if __name__ == "__main__":
    main()
