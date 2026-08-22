"""Ablation statistics with significance tests, not eyeballed means.

Written because eyeballing cost this study three reversals. The same comparison
read positive, then null, then positive again across earlier sweeps, and each
reading was taken from two seeds whose spread was of the same order as the
difference being claimed. Any ablation on this task needs a seed count and a
test, or it is noise interpretation.

Reports mean, standard deviation, Welch t-tests (unequal variances, since the
variants have visibly different spread) and Cohen's d for every pair, plus a
LaTeX table for the manuscript.

Usage::

    python scripts/ablation_stats.py --runs runs/credit_fine --out paper/ablation.tex
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> None:
    args = _parse_args()
    groups = _load(Path(args.runs))
    if not groups:
        raise SystemExit(f"no runs found in {args.runs}")

    print(f"{'variant':18s} {'n':>2}  {'healthy decay':>22}  {'stabilised':>11}")
    for name in sorted(groups):
        decay, stabilised = groups[name]
        print(
            f"{name:18s} {decay.size:2d}  {decay.mean():+9.3f} +- {decay.std(ddof=1):6.3f}"
            f"        {stabilised.mean():4.1f}/9"
        )

    print("\npairwise Welch t-tests on healthy decay rate (more negative is better):")
    rows = []
    for first, second in itertools.combinations(sorted(groups), 2):
        left, right = groups[first][0], groups[second][0]
        statistic, pvalue = stats.ttest_ind(left, right, equal_var=False)
        difference = left.mean() - right.mean()
        pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
        effect = difference / pooled if pooled > 0 else np.nan
        rows.append((first, second, difference, effect, pvalue))
        marker = "  SIGNIFICANT" if pvalue < 0.05 else ""
        print(
            f"  {first:18s} vs {second:18s}  diff {difference:+6.3f}  "
            f"d {effect:+5.2f}  p = {pvalue:.4f}{marker}"
        )
        del statistic

    if args.out:
        _write_latex(Path(args.out), groups, rows)
        print(f"\nwrote {args.out}")


def _load(runs: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(runs.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(payload["variant"]["name"], []).append(payload)

    result = {}
    for name, payloads in grouped.items():
        decay, stabilised = [], []
        for payload in payloads:
            healthy = [
                r["suppression_rate"]
                for r in payload["results"]
                if r["condition"]["jam_surface"] < 0 and r["divergence_rate"] == 0.0
            ]
            if healthy:
                decay.append(float(np.mean(healthy)))
            stabilised.append(
                sum(1 for r in payload["results"] if r["divergence_rate"] == 0.0)
            )
        if decay:
            result[name] = (np.asarray(decay), np.asarray(stabilised, dtype=float))
    return result


def _write_latex(path: Path, groups, rows) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Seeds & Decay rate [1/s] & Conditions held \\",
        r"\midrule",
    ]
    for name in sorted(groups):
        decay, stabilised = groups[name]
        lines.append(
            f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {decay.size} & "
            f"${decay.mean():.3f} \\pm {decay.std(ddof=1):.3f}$ & "
            f"{stabilised.mean():.1f}/9 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.5em}", ""]
    lines += [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Comparison & & Cohen's $d$ & $p$ \\",
        r"\midrule",
    ]
    for first, second, _, effect, pvalue in rows:
        star = r"$^{*}$" if pvalue < 0.05 else ""
        lines.append(
            f"\\texttt{{{first.replace('_', chr(92) + '_')}}} & "
            f"vs \\texttt{{{second.replace('_', chr(92) + '_')}}} & "
            f"${effect:.2f}$ & ${pvalue:.4f}${star} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/credit_fine")
    parser.add_argument("--out", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
