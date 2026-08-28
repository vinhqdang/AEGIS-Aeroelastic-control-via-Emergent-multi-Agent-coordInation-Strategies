"""Rebuild the credit-assignment ablation table with the two new 6-seed groups.

Adds ``vdn_shared`` (learned value-decomposition credit, see
scripts/train_agents.py) and the now-6-seed ``ippo_shared`` (previously a
1-seed probe) to the three published groups (``ippo_physics``,
``ippo_blended``, ``shaped_shared``, from paper/ablation_credit.tex, whose
original per-seed run archive is not present in this checkout) and produces
a fresh Holm-Bonferroni-corrected, Hedges'-g-reported comparison across all
five, i.e. across the full C(5,2)=10-test family rather than two disconnected
sub-families.

Usage::

    python scripts/rebuild_credit_ablation.py --out paper/ablation_credit.tex
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

# From paper/ablation_credit.tex as it stood before this study's revision.
PUBLISHED = {
    "ippo_physics": (-0.566, 0.123, 6),
    "ippo_blended": (-1.085, 0.243, 6),
    "shaped_shared": (-0.690, 0.145, 6),
}
FRESH_RUN_DIRS = {
    "ippo_shared": "runs/ippo_shared_6seed",
    "vdn_shared": "runs/vdn_shared",
}
STABILISED_OF = {
    # Held-out from paper/ablation_credit.tex's "Conditions held" column.
    "ippo_physics": 5.0,
    "ippo_blended": 5.0,
    "shaped_shared": 4.3,
}


def _load_healthy_decay(runs: Path) -> np.ndarray:
    values = []
    for path in sorted(runs.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        healthy = [
            r["suppression_rate"]
            for r in payload["results"]
            if r["condition"]["jam_surface"] < 0 and r["divergence_rate"] == 0.0
        ]
        if healthy:
            values.append(float(np.mean(healthy)))
    return np.asarray(values)


def _stabilised_mean(runs: Path) -> float:
    counts = []
    for path in sorted(runs.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts.append(sum(1 for r in payload["results"] if r["divergence_rate"] == 0.0))
    return float(np.mean(counts))


def main() -> None:
    args = _parse_args()
    groups = dict(PUBLISHED)
    stabilised = dict(STABILISED_OF)
    for name, directory in FRESH_RUN_DIRS.items():
        values = _load_healthy_decay(Path(directory))
        groups[name] = (float(values.mean()), float(values.std(ddof=1)), values.size)
        stabilised[name] = _stabilised_mean(Path(directory))

    print(f"{'variant':16s} {'n':>2}  {'healthy decay':>18}  {'stabilised':>11}")
    for name in sorted(groups):
        mean, std, n = groups[name]
        print(f"{name:16s} {n:2d}  {mean:+7.3f} +- {std:5.3f}        {stabilised[name]:4.1f}/9")

    rows = []
    for first, second in itertools.combinations(sorted(groups), 2):
        m1, s1, n1 = groups[first]
        m2, s2, n2 = groups[second]
        statistic, pvalue = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        effect = (m1 - m2) / pooled if pooled > 0 else np.nan
        correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0) if n1 + n2 > 2 else np.nan
        hedges_g = effect * correction
        rows.append([first, second, m1 - m2, effect, hedges_g, pvalue, None])
        del statistic

    _holm_bonferroni(rows)
    print("\npairwise comparisons (Welch, from summary statistics where the "
          "original per-seed data is not present in this checkout):")
    for first, second, diff, effect, hedges_g, pvalue, adjusted in rows:
        marker = "  SIGNIFICANT (Holm)" if adjusted < 0.05 else ""
        print(
            f"  {first:14s} vs {second:14s}  diff {diff:+6.3f}  d {effect:+5.2f}  "
            f"g {hedges_g:+5.2f}  p={pvalue:.4f}  p_holm={adjusted:.4f}{marker}"
        )

    if args.out:
        _write_latex(Path(args.out), groups, stabilised, rows)
        print(f"\nwrote {args.out}")


def _holm_bonferroni(rows: list[list]) -> None:
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][5])
    running_max = 0.0
    for rank, index in enumerate(order):
        inflated = (m - rank) * rows[index][5]
        running_max = max(running_max, inflated)
        rows[index][6] = min(1.0, running_max)


def _write_latex(path: Path, groups: dict, stabilised: dict, rows: list) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Seeds & Decay rate [1/s] & Conditions held \\",
        r"\midrule",
    ]
    for name in sorted(groups):
        mean, std, n = groups[name]
        lines.append(
            f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {n} & "
            f"${mean:.3f} \\pm {std:.3f}$ & {stabilised[name]:.1f}/9 \\\\"
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
    parser.add_argument("--out", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
