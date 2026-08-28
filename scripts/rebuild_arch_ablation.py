"""Recompute the architecture ablation table with Hedges' g and Holm-Bonferroni
correction across the full comparison family (a review flagged the original
table's dozen-plus uncorrected pairwise p-values). The original per-seed run
archive is not present in this checkout, so this uses the published summary
statistics (mean, std, n=6) from the pre-revision paper/ablation_arch.tex via
scipy.stats.ttest_ind_from_stats, which needs only those summary statistics --
statistically equivalent to the two-sample Welch's t-test used originally.

Usage::

    python scripts/rebuild_arch_ablation.py --out paper/ablation_arch.tex
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

# From paper/ablation_arch.tex and paper/ablation_credit.tex as they stood
# before this study's revision.
PUBLISHED = {
    "ippo_blended": (-1.085, 0.243, 6),
    "comm_raw": (-0.596, 0.227, 6),
    "mocca": (-0.389, 0.219, 6),
    "mocca_nohead": (-0.308, 0.169, 6),
    "recurrent_only": (-0.474, 0.249, 6),
}
STABILISED_OF = {
    "ippo_blended": 5.0,
    "comm_raw": 5.0,
    "mocca": 4.2,
    "mocca_nohead": 5.0,
    "recurrent_only": 5.0,
}
# Only these four are reported in the architecture table itself; ippo_blended
# is included in the comparison family (it is the reference every architecture
# variant is measured against) but shown in Table tab:ablation-credit instead.
ARCH_VARIANTS = ["comm_raw", "mocca", "mocca_nohead", "recurrent_only"]


def main() -> None:
    args = _parse_args()
    rows = []
    for first, second in itertools.combinations(sorted(PUBLISHED), 2):
        m1, s1, n1 = PUBLISHED[first]
        m2, s2, n2 = PUBLISHED[second]
        statistic, pvalue = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
        pooled = np.sqrt((s1**2 + s2**2) / 2.0)
        effect = (m1 - m2) / pooled if pooled > 0 else np.nan
        correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0) if n1 + n2 > 2 else np.nan
        hedges_g = effect * correction
        rows.append([first, second, m1 - m2, effect, hedges_g, pvalue, None])
        del statistic

    _holm_bonferroni(rows)
    for first, second, diff, effect, hedges_g, pvalue, adjusted in rows:
        marker = "  SIGNIFICANT (Holm)" if adjusted < 0.05 else ""
        print(
            f"  {first:16s} vs {second:16s}  diff {diff:+6.3f}  d {effect:+5.2f}  "
            f"g {hedges_g:+5.2f}  p={pvalue:.4f}  p_holm={adjusted:.4f}{marker}"
        )

    if args.out:
        _write_latex(Path(args.out), rows)
        print(f"\nwrote {args.out}")


def _holm_bonferroni(rows: list[list]) -> None:
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][5])
    running_max = 0.0
    for rank, index in enumerate(order):
        inflated = (m - rank) * rows[index][5]
        running_max = max(running_max, inflated)
        rows[index][6] = min(1.0, running_max)


def _write_latex(path: Path, rows: list) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Seeds & Decay rate [1/s] & Conditions held \\",
        r"\midrule",
    ]
    for name in ARCH_VARIANTS:
        mean, std, n = PUBLISHED[name]
        lines.append(
            f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {n} & "
            f"${mean:.3f} \\pm {std:.3f}$ & {STABILISED_OF[name]:.1f}/9 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{0.5em}", ""]
    lines += [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparison & & Cohen's $d$ & Hedges' $g$ & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    # Only comparisons among the four architecture variants themselves; the
    # vs-ippo_blended comparisons are reported in prose (Section on
    # Ablations), since ippo_blended's own row lives in the other table.
    shown = [r for r in rows if "ippo_blended" not in (r[0], r[1])]
    for first, second, _, effect, hedges_g, pvalue, adjusted in shown:
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
