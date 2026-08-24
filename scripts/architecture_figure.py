"""Block diagram of the full MoCCA policy: encoder, consensus, and action head.

The ablation tables name four architectural variants (comm_raw, mocca,
mocca_nohead, recurrent_only) and the text explains what each does to
performance, but nothing in the manuscript shows what any of them actually
*is* as a network. This draws the full pipeline once, in one figure, with the
three optional mechanisms (the encoder, the consensus rounds, the phase-locked
head) labelled so that removing one from the diagram is a literal reading of
which ablation row it corresponds to.

Usage::

    python scripts/architecture_figure.py --out paper/figures/fig_architecture.pdf
"""

from __future__ import annotations

import argparse


def main() -> None:
    args = _parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from aegis.viz.paper import INK, INK_MUTED, controller_color, use_paper_style

    use_paper_style()

    figure, axes = plt.subplots(figsize=(9.2, 4.3))
    axes.set_xlim(0, 15.6)
    axes.set_ylim(0, 8.0)
    axes.axis("off")

    def box(x, y, w, h, text, face, edge=INK, fontsize=8.0, textcolor="black"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=face, edgecolor=edge, lw=1.1, alpha=0.92, zorder=2,
        )
        axes.add_patch(patch)
        axes.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=3, wrap=True,
        )
        return (x, y, w, h)

    def arrow(src, dst, color=INK_MUTED, style="-|>", label="", label_dy=0.18):
        x0 = src[0] + src[2]
        y0 = src[1] + src[3] / 2
        x1 = dst[0]
        y1 = dst[1] + dst[3] / 2
        patch = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, color=color, lw=1.2,
            mutation_scale=11, zorder=1,
        )
        axes.add_patch(patch)
        if label:
            axes.text(
                0.5 * (x0 + x1), 0.5 * (y0 + y1) + label_dy, label,
                ha="center", va="bottom", fontsize=6.8, color=color,
            )

    neutral = "#eef1f4"
    optional_edge = "#8d4f9e"

    obs = box(0.2, 4.4, 1.9, 1.4, "local\nobservation\n$o_k^t$", neutral)
    trunk_in = box(2.9, 4.4, 1.7, 1.4, "concat\n$[o_k, e_k]$", neutral)
    trunk = box(5.4, 4.4, 2.0, 1.4, "shared trunk\nMLP, 2×128,\ntanh", controller_color(3), textcolor="white")
    action = box(8.2, 5.6, 2.1, 1.2, "plain head\n$\\tanh(\\mathrm{MLP})$", neutral)
    plhead = box(8.2, 3.2, 2.5, 1.4, "phase-locked head\n$\\tanh(u+\\sum_r A_r\\,\\hat\\phi_r{\\cdot}R_r)$", "#fbeff7", edge=optional_edge)
    out = box(11.4, 4.6, 1.5, 1.0, "action\n$a_k\\in[-1,1]$", controller_color(1), textcolor="white")

    encoder = box(0.2, 1.1, 2.4, 1.4, "PhasorEncoder\nGRU$_{64}$, per agent\n(optional)", "#fbeff7", edge=optional_edge)
    consensus = box(3.3, 1.1, 2.6, 1.4, "PhasorConsensus\n2 rounds, line graph\n(optional)", "#fbeff7", edge=optional_edge)

    arrow(obs, trunk_in)
    arrow(trunk_in, trunk)
    arrow(trunk, action, label="latent $\\ell_k$")
    arrow(trunk, plhead, label="latent $\\ell_k$", label_dy=-0.32)
    arrow(action, out)
    arrow(plhead, out)
    arrow(obs, encoder, color=optional_edge, label="history")
    arrow(encoder, consensus, color=optional_edge, label="$\\hat\\phi_k$, $2R$ nums")
    arrow(consensus, trunk_in, color=optional_edge, label="$e_k=$ consensus phasor")
    arrow(consensus, plhead, color=optional_edge, label="phasor $\\hat\\phi$")

    axes.text(
        0.2, 6.35, "always present", fontsize=8.5, color=INK, fontweight="bold",
    )
    axes.text(
        0.2, 0.35, "optional (ablated in Table 4)", fontsize=8.5, color=optional_edge,
        fontweight="bold",
    )
    axes.annotate(
        "", xy=(0.05, 6.15), xytext=(0.05, 0.55),
        arrowprops=dict(arrowstyle="-", color="#c9c9c9", lw=8, alpha=0.35),
    )

    figure.tight_layout()
    figure.savefig(args.out)
    print(f"wrote {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures/fig_architecture.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
