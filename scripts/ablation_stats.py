"""Ablation statistics with significance tests, not eyeballed means.

Written because eyeballing cost this study three reversals. The same comparison
read positive, then null, then positive again across earlier sweeps, and each
reading was taken from two seeds whose spread was of the same order as the
difference being claimed. Any ablation on this task needs a seed count and a
test, or it is noise interpretation.

Reports mean, standard deviation, Welch t-tests (unequal variances, since the
variants have visibly different spread), Cohen's d, and Hedges' g (the
small-sample-bias-corrected effect size -- d has a known positive bias of
order 1/(4n) at n=6 per group) for every pair, plus a Holm-Bonferroni-corrected
p-value across every pairwise test actually run in one invocation (a review
flagged that quoting a dozen-plus uncorrected pairwise p-values without any
multiple-comparison correction overstates significance), plus a LaTeX table for
the manuscript.

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
        n1, n2 = left.size, right.size
        hedges_correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0) if n1 + n2 > 2 else np.nan
        hedges_g = effect * hedges_correction
        rows.append([first, second, difference, effect, hedges_g, pvalue, None])
        del statistic

    _holm_bonferroni(rows)

    for first, second, difference, effect, hedges_g, pvalue, adjusted in rows:
        marker = "  SIGNIFICANT (Holm)" if adjusted < 0.05 else ""
        print(
            f"  {first:18s} vs {second:18s}  diff {difference:+6.3f}  "
            f"d {effect:+5.2f}  g {hedges_g:+5.2f}  p = {pvalue:.4f}  "
            f"p_holm = {adjusted:.4f}{marker}"
        )

    if args.out:
        _write_latex(Path(args.out), groups, rows)
        print(f"\nwrote {args.out}")


def _holm_bonferroni(rows: list[list]) -> None:
    """Step-down Holm-Bonferroni correction, in place, over rows[i][5] (pvalue)
    into rows[i][6] (adjusted). Standard, monotone-enforced Holm procedure:
    the k-th smallest of m p-values is inflated by (m - k + 1), then adjusted
    upward as needed so adjusted p-values are non-decreasing in sorted order.
    """
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][5])
    running_max = 0.0
    for rank, index in enumerate(order):
        inflated = (m - rank) * rows[index][5]
        running_max = max(running_max, inflated)
        rows[index][6] = min(1.0, running_max)


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
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparison & & Cohen's $d$ & Hedges' $g$ & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for first, second, _, effect, hedges_g, pvalue, adjusted in rows:
        star = r"$^{*}$" if adjusted < 0.05 else ""
        lines.append(
            f"\\texttt{{{first.replace('_', chr(92) + '_')}}} & "
            f"vs \\texttt{{{second.replace('_', chr(92) + '_')}}} & "
            f"${effect:.2f}$ & ${hedges_g:.2f}$ & ${pvalue:.4f}$ & "
            f"${adjusted:.4f}${star} \\\\"
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
