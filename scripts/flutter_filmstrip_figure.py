"""Static filmstrip of the wing actually flapping -- for a medium that cannot play video.

The manuscript's only view of the wing was the undeformed schematic
(fig_schematic.pdf). A reader has no way to see what flutter, or its
suppression, actually looks like as motion. A PDF cannot embed the project's
animations (media/*.mp4, from scripts/animate_flutter.py), so this produces the
next best thing at publication quality: the same two rollouts -- open loop and
centralised LQR -- sampled at six shared instants and rendered as two rows of
wing snapshots on one shared, honest deformation scale. Reading left to right
across the top row is watching flutter diverge; the same instants on the bottom
row are the same initial pluck, suppressed.

Both rows share one deformation scale (chosen from the open-loop trajectory,
the larger of the two) and one pair of axis limits, so a flat-looking bottom
row is not an axis-scaling artefact -- it is the point being made.

Usage::

    python scripts/flutter_filmstrip_figure.py --out paper/figures/fig_filmstrip.pdf
"""

from __future__ import annotations

import argparse

import numpy as np

from aegis.control.lqr import CentralizedLQR
from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.wing import GOLAND_WING
from aegis.simulator import WingSimulation

INITIAL_TIP_PLUNGE = 0.05   # m, matches scripts/animate_flutter.py
DURATION = 2.4              # s
DIVERGENCE_LIMIT = 0.9      # m of tip travel
N_FRAMES = 6


def main() -> None:
    args = _parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aegis.viz import animation as animation_module
    from aegis.viz import wing_render as wing_render_module
    from aegis.viz.palette import THEME_LIGHT
    from aegis.viz.paper import INK, INK_MUTED, use_paper_style
    from aegis.viz.wing_render import WingRenderer, project

    use_paper_style()
    # WingRenderer reads the module-level THEME at call time, so swapping it
    # here repaints every renderer this script creates without touching the
    # animation code path, which still wants the dark theme for on-screen use.
    wing_render_module.THEME = THEME_LIGHT

    model = AeroelasticModel(GOLAND_WING)
    flutter = model.flutter_point()
    airspeed = flutter.airspeed * 1.10

    open_loop = _rollout(model, airspeed, controller=None)
    controlled = _rollout(model, airspeed, controller=CentralizedLQR(model, airspeed))

    times = np.linspace(0.0, open_loop.time[-1], N_FRAMES)
    scale = animation_module.choose_deformation_scale(model, open_loop)
    peak_plunge = float(np.abs(open_loop.tip_plunge).max())
    peak_twist = float(np.abs(open_loop.tip_twist).max())

    figure, grid = plt.subplots(
        2, N_FRAMES, figsize=(10.5, 4.4),
        gridspec_kw=dict(hspace=0.05, wspace=0.05, left=0.075, right=0.99, top=0.85, bottom=0.09),
    )

    # Short labels only: at this figure height a full sentence rotated 90 deg
    # is taller than either row and sprawls into its neighbour regardless of
    # how precisely the two label centres are separated. The explanation goes
    # in the caption instead.
    rows = (
        ("open loop", open_loop, "#b03a2e"),
        ("LQR", controlled, "#0f9b8e"),
    )
    for row_index, (row_title, trajectory, row_color) in enumerate(rows):
        for col_index, t in enumerate(times):
            axes = grid[row_index, col_index]
            renderer = WingRenderer(
                model, axes, deformation_scale=scale, show_reference=True,
            )
            frame = int(np.argmin(np.abs(trajectory.time - t)))
            renderer.update(
                trajectory.modal_position[frame], trajectory.deflection[frame]
            )
            _fit_axes(axes, model, project, peak_plunge, peak_twist, scale)
            if row_index == 0:
                axes.set_title(f"$t={t:0.2f}$ s", fontsize=8, color=INK_MUTED, pad=3)

    # Row labels are placed from the axes' FINAL position (post aspect-equal
    # box adjustment), not transAxes at creation time -- an equal-aspect box on
    # data much taller than its cell gets re-anchored by matplotlib, and a
    # transAxes-relative label drifts away from the row it names.
    figure.canvas.draw()
    for row_index, (row_title, _, row_color) in enumerate(rows):
        left_axes = grid[row_index, 0]
        box = left_axes.get_position()
        figure.text(
            0.012, 0.5 * (box.y0 + box.y1), row_title, rotation=90,
            ha="center", va="center", fontsize=10, color=row_color, fontweight="bold",
        )

    figure.suptitle(
        f"Same {INITIAL_TIP_PLUNGE*100:.0f} cm pluck at "
        f"$U/U_\\mathrm{{f}}={airspeed/flutter.airspeed:.2f}$, same deformation scale "
        f"(×{scale:.0f}) in both rows",
        fontsize=9, color=INK, y=0.975,
    )

    figure.savefig(args.out)
    print(f"wrote {args.out}")
    print(f"  open loop diverged at t={open_loop.time[-1]:.2f} s"
          f" (peak tip {peak_plunge*100:.1f} cm)")
    print(f"  LQR final tip plunge {abs(controlled.tip_plunge[-1])*1000:.2f} mm")


def _rollout(model, airspeed, controller):
    sim = WingSimulation(model, airspeed, divergence_limit=DIVERGENCE_LIMIT)
    return sim.rollout(
        controller,
        duration=DURATION,
        initial_tip_plunge=INITIAL_TIP_PLUNGE,
        stop_on_divergence=True,
    )


def _fit_axes(axes, model, project, peak_plunge, peak_twist, scale) -> None:
    """Identical axis limits for every panel, sized from the larger trajectory."""
    wing = model.wing
    corners_span = np.asarray([0.0, wing.semispan, wing.semispan, 0.0])
    corners_chord = np.asarray([0.0, 0.0, wing.chord, wing.chord])
    base_x, base_y = project(corners_span, corners_chord, np.zeros(4))
    motion = scale * (peak_plunge + 0.7 * wing.chord * peak_twist)

    x_center = 0.5 * (base_x.min() + base_x.max())
    x_half = 0.5 * (base_x.max() - base_x.min()) * 1.06
    y_center = 0.5 * (base_y.min() + base_y.max())
    y_half = 0.5 * (base_y.max() - base_y.min()) + motion * 1.10 + 0.10

    axes.set_xlim(x_center - x_half, x_center + x_half)
    axes.set_ylim(y_center - y_half, y_center + y_half)
    axes.set_aspect("equal")
    axes.axis("off")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper/figures/fig_filmstrip.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    main()
