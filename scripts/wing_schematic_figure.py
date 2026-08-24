"""Static schematic of the testbed: planform, typical section, and Dec-POMDP loop.

The manuscript described the wing, its control surfaces and sensors entirely in
prose. A reader has to reconstruct the geometry mentally before the rest of the
paper makes sense, and every aeroelasticity paper in the target venue leads with
exactly this kind of figure. This produces it from the same wing/surface objects
the plant and the environment use, so the picture cannot silently drift from the
model it is meant to depict.

Three panels:
  (a) planform view -- span, chord, the three control-surface bands in their
      agent colours, the elastic axis and centre-of-gravity lines, and the
      sensor station at the midpoint of each surface;
  (b) typical section at one span station -- plunge and twist degrees of
      freedom, elastic axis, centre of gravity, and the flap hinge line, which
      is the geometry the thin-airfoil derivative in Section 5 refers to;
  (c) the decentralised control loop -- one agent per surface, local
      observation only, closing the loop through the shared physical plant.

Usage::

    python scripts/wing_schematic_figure.py --out paper/figures/fig_schematic.pdf
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    args = _parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.physics.wing import GOLAND_WING
    from aegis.viz.paper import INK, INK_MUTED, controller_color, use_paper_style

    use_paper_style()
    wing = GOLAND_WING

    # Sized so panel (a), which is wide and flat, does not leave the dead space
    # that a naive equal height allocation produces under an equal-aspect axis.
    figure = plt.figure(figsize=(8.6, 4.6))
    grid = figure.add_gridspec(
        2, 2, height_ratios=[0.62, 1.0], width_ratios=[1.05, 1.0],
        hspace=0.62, wspace=0.32, left=0.06, right=0.98, top=0.90, bottom=0.07,
    )
    planform = figure.add_subplot(grid[0, :])
    section = figure.add_subplot(grid[1, 0])
    loop = figure.add_subplot(grid[1, 1])

    _draw_planform(planform, wing, controller_color, INK, INK_MUTED)
    _draw_typical_section(section, wing, INK, INK_MUTED)
    _draw_control_loop(loop, wing, controller_color, INK, INK_MUTED)

    figure.savefig(args.out)
    print(f"wrote {args.out}")


def _draw_planform(axes, wing, colour_of, ink, ink_muted) -> None:
    from matplotlib.patches import Rectangle

    semispan, chord = wing.semispan, wing.chord

    outline_y = np.array([0.0, semispan, semispan, 0.0, 0.0])
    outline_x = np.array([0.0, 0.0, chord, chord, 0.0])
    for side in (1.0, -1.0):
        axes.plot(side * outline_y, outline_x, color=ink, lw=1.1)
        axes.fill(side * outline_y, outline_x, color="#eef1f4", zorder=0)

    # Staggered label depths, because the three stations sit close enough in
    # span that same-depth labels collide.
    label_depth = (-0.50, -0.70, -0.50)
    for k, surface in enumerate(wing.surfaces):
        y0, y1 = surface.y_start_frac * semispan, surface.y_end_frac * semispan
        hinge_x = surface.hinge_frac * chord
        colour = colour_of(k)
        for side in (1.0, -1.0):
            origin_y = side * y1 if side < 0 else y0
            axes.add_patch(
                Rectangle(
                    (origin_y, hinge_x), (y1 - y0), chord - hinge_x,
                    facecolor=colour, edgecolor=ink, lw=0.6, alpha=0.85, zorder=2,
                )
            )
            station = 0.5 * (y0 + y1)
            axes.plot([side * station], [0.35 * chord], "o", color=ink, ms=4,
                       zorder=3, markerfacecolor="white", markeredgewidth=1.0)
        axes.annotate(
            surface.name.replace("_", " "),
            xy=(0.5 * (y0 + y1), hinge_x - 0.05 * chord),
            xytext=(0.5 * (y0 + y1), label_depth[k] * chord),
            ha="center", va="top", fontsize=6.9, color=colour,
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.7),
        )

    axes.plot([-semispan, semispan], [wing.ea_frac * chord] * 2, "--",
               color=ink_muted, lw=0.9)
    axes.plot([-semispan, semispan], [wing.cg_frac * chord] * 2, ":",
               color=ink_muted, lw=0.9)
    axes.text(semispan * 1.03, wing.ea_frac * chord, "EA", fontsize=7.2,
               color=ink_muted, va="center")
    axes.text(semispan * 1.03, wing.cg_frac * chord, "CG", fontsize=7.2,
               color=ink_muted, va="center")
    # Callout routed upward, into the empty space above the wing: below and to
    # the right are both already crowded with surface labels and the EA/CG key.
    middle = wing.surfaces[len(wing.surfaces) // 2]
    axes.annotate(
        "sensor station\n(accel. pair, fwd/aft of EA)",
        xy=(0.5 * (middle.y_start_frac + middle.y_end_frac) * semispan, 0.35 * chord),
        xytext=(semispan * 0.30, 0.95 * chord),
        fontsize=6.6, color=ink_muted, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=ink_muted, lw=0.6,
                         connectionstyle="arc3,rad=-0.3"),
    )

    axes.set_xlim(-semispan * 1.16, semispan * 1.16)
    axes.set_ylim(-0.95 * chord, 1.30 * chord)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title(
        "(a) Planform: three control surfaces, one agent each, "
        "with a local accelerometer pair at each station",
        fontsize=9, loc="left",
    )


def _draw_typical_section(axes, wing, ink, ink_muted) -> None:
    from matplotlib.patches import FancyArrowPatch

    chord = wing.chord
    hinge = wing.surfaces[1].hinge_frac * chord
    ea_x, cg_x = wing.ea_frac * chord, wing.cg_frac * chord

    axes.plot([0, chord], [0, 0], color=ink, lw=1.4, zorder=2)
    axes.plot([hinge, chord], [0, 0], color="#c9427d", lw=3.4,
               solid_capstyle="butt", zorder=3)
    axes.plot([hinge], [0], "|", color=ink, ms=12, mew=1.6, zorder=4)

    axes.plot([ea_x], [0], "o", color=ink, ms=7, zorder=5)
    # CG sits BELOW the chord line on purpose, even though it is physically on
    # the same line: the plunge arrow and the twist arc both live above the
    # line, and keeping CG below is what stops the three annotations colliding.
    axes.plot([cg_x], [-0.20 * chord], "x", color=ink, ms=8, mew=1.8, zorder=5)
    axes.plot([cg_x, cg_x], [0, -0.20 * chord], ":", color=ink_muted, lw=0.8, zorder=1)
    axes.text(ea_x, -0.38 * chord, "EA", fontsize=7.5, ha="center", va="top")
    axes.text(cg_x, -0.30 * chord, "CG", fontsize=7.5, ha="center", va="top")
    axes.text(hinge, -0.38 * chord, "hinge\n(75% $c$)", fontsize=6.6, ha="center",
               va="top", color="#c9427d")

    # Plunge: a straight double-ended arrow above the elastic axis.
    axes.annotate(
        "", xy=(ea_x, 0.62 * chord), xytext=(ea_x, 0.30 * chord),
        arrowprops=dict(arrowstyle="-|>", color=ink_muted, lw=1.1),
    )
    axes.text(ea_x - 0.03 * chord, 0.46 * chord, "$h$\n(plunge,\n+down)",
               fontsize=6.6, color=ink_muted, va="center", ha="right")

    # Twist: a genuinely curved arrow about the elastic axis, well clear of CG
    # (which is now below the line) and of the plunge arrow (to its left).
    twist_arc = FancyArrowPatch(
        (ea_x + 0.10 * chord, 0.34 * chord), (ea_x + 0.32 * chord, 0.22 * chord),
        connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=10,
        color=ink_muted, lw=1.1, zorder=5,
    )
    axes.add_patch(twist_arc)
    axes.text(ea_x + 0.38 * chord, 0.30 * chord, r"$\alpha$" + "\n(twist,\n+nose-up)",
               fontsize=6.6, color=ink_muted, va="center", ha="left")

    axes.set_xlim(-0.10 * chord, 1.10 * chord)
    axes.set_ylim(-0.55 * chord, 0.75 * chord)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title("(b) Typical section: DOFs and the flap hinge", fontsize=9, loc="left")


def _draw_control_loop(axes, wing, colour_of, ink, ink_muted) -> None:
    from matplotlib.patches import FancyBboxPatch

    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis("off")
    axes.set_title("(c) Decentralised loop: local observation, no shared state",
                    fontsize=9, loc="left")

    plant_box = FancyBboxPatch((2.7, 0.3), 4.6, 1.7, boxstyle="round,pad=0.08",
                                facecolor="#eef1f4", edgecolor=ink, lw=1.0)
    axes.add_patch(plant_box)
    axes.text(5.0, 1.15, "shared plant\n" + r"$M\ddot q + C\dot q + Kq = Lx_{\mathrm{w}} + B\delta$",
               ha="center", va="center", fontsize=6.6)

    # Positions computed from the box width and a fixed margin, rather than
    # hard-coded offsets, so three boxes always fit inside the axes with even
    # gaps instead of the last one silently overflowing the right edge.
    n = wing.n_surfaces
    box_w, margin = 2.5, 0.5
    gap = (10.0 - 2 * margin - n * box_w) / (n - 1) if n > 1 else 0.0
    for k, surface in enumerate(wing.surfaces):
        x = margin + k * (box_w + gap)
        colour = colour_of(k)
        box = FancyBboxPatch((x, 6.5), box_w, 1.7, boxstyle="round,pad=0.08",
                              facecolor=colour, edgecolor=ink, lw=1.0, alpha=0.88)
        axes.add_patch(box)
        axes.text(x + box_w / 2, 7.35, f"agent $k{'=' + str(k + 1)}$\n"
                                        f"{surface.name.replace('_', ' ')}",
                   ha="center", va="center", fontsize=6.5, color="black")
        delta_x = x + box_w * 0.62
        obs_x = x + box_w * 0.30
        axes.annotate(
            "", xy=(delta_x, 2.0), xytext=(delta_x, 6.5),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3, shrinkA=0, shrinkB=0),
        )
        axes.text(delta_x + 0.18, 4.2, "$\\delta_k$", fontsize=7.0, color=colour)
        axes.annotate(
            "", xy=(obs_x, 6.5), xytext=(obs_x, 2.0),
            arrowprops=dict(arrowstyle="-|>", color=ink_muted, lw=1.0, shrinkA=0, shrinkB=0),
        )
        axes.text(obs_x - 0.55, 4.2, "$o_k$", fontsize=7.0, color=ink_muted)

    axes.text(5.0, 9.35, "no inter-agent channel in the plain / physics-credit policy",
               ha="center", fontsize=6.6, color=ink_muted, style="italic")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures/fig_schematic.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
