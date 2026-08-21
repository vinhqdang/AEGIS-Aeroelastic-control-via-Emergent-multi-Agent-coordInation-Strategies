"""Animated rollout view: deforming wing plus the telemetry that explains it.

The layout is deliberate. A wing view alone shows *that* the wing flutters; it
does not show *why* a controller works. So each frame pairs the geometry with:

* tip plunge -- the quantity a flight-test engineer would watch;
* per-surface deflection -- what each agent actually did, in its own colour;
* structural energy on a log axis -- growth versus decay, the real objective;
* a bending-torsion phase portrait -- flutter *is* the two modes locking into a
  fixed phase relationship, and an orbit that opens out says it better than any
  eigenvalue plot;
* per-agent control power -- the closed-form difference reward of Proposition 1,
  drawn as signed bars so it is immediately visible who is extracting energy from
  the structure and who is pumping it.

That last panel is the point of the whole figure: it makes credit assignment
something you can watch rather than something you infer from a learning curve.

Structural deformation is exaggerated by a factor chosen automatically so the
motion is visible, and the factor is printed on every frame. Silently scaling
geometry in an explanatory animation is how people end up with the wrong mental
model of how much a wing actually moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from aegis.physics.aeroelastic import AeroelasticModel
from aegis.simulator import TrajectoryArrays
from aegis.viz.palette import THEME, agent_color
from aegis.viz.wing_render import WingRenderer, project

DEFAULT_FPS = 25
_FIGURE_SIZE = (14.4, 7.8)
_DPI = 100

# Target peak tip travel on screen, as a fraction of the semispan.
_TARGET_TIP_TRAVEL_FRACTION = 0.24
_MAX_EXAGGERATION = 400.0
_PHASE_TAIL = 90  # trajectory steps kept bright in the phase portrait


@dataclass(frozen=True)
class AnimationSpec:
    """Everything about an animation that is not the trajectory itself."""

    title: str
    output_path: Path
    stride: int = 2
    fps: int = DEFAULT_FPS
    deformation_scale: float | None = None  # None selects it automatically
    flutter_speed: float | None = None
    subtitle: str = ""


def render_animation(
    model: AeroelasticModel, trajectory: TrajectoryArrays, spec: AnimationSpec
) -> Path:
    """Write an animation of ``trajectory`` and return the path written."""
    if trajectory.n_steps < 2:
        raise ValueError("trajectory too short to animate")

    panels = _build_figure(model, trajectory, spec)
    writer = _open_writer(spec)
    try:
        for index in range(0, trajectory.n_steps, max(1, spec.stride)):
            panels.update(index)
            writer.append_data(_grab_frame(panels.figure))
    finally:
        writer.close()
        plt.close(panels.figure)
    return spec.output_path


def choose_deformation_scale(
    model: AeroelasticModel, trajectory: TrajectoryArrays
) -> float:
    """Exaggeration factor that brings peak tip travel to a visible size."""
    peak = float(np.abs(trajectory.tip_plunge).max())
    if peak <= 1e-9:
        return 1.0
    target = _TARGET_TIP_TRAVEL_FRACTION * model.wing.semispan
    return float(np.clip(target / peak, 1.0, _MAX_EXAGGERATION))


class _Panels:
    """Mutable handles for every artist that changes between frames."""

    def __init__(self, figure, renderer, trajectory, artists):
        self.figure = figure
        self.renderer = renderer
        self.trajectory = trajectory
        self.artists = artists

    def update(self, index: int) -> None:
        traj = self.trajectory
        self.renderer.update(traj.modal_position[index], traj.deflection[index])

        history = slice(0, index + 1)
        elapsed = traj.time[history]
        self.artists["tip"].set_data(elapsed, traj.tip_plunge[history])
        for k, line in enumerate(self.artists["deflection"]):
            line.set_data(elapsed, np.rad2deg(traj.deflection[history, k]))
        self.artists["energy"].set_data(elapsed, np.maximum(traj.energy[history], 1e-12))
        for cursor in self.artists["cursors"]:
            cursor.set_xdata([traj.time[index], traj.time[index]])

        tail = slice(max(0, index - _PHASE_TAIL), index + 1)
        self.artists["phase_tail"].set_data(
            traj.modal_position[tail, 0], np.rad2deg(traj.modal_position[tail, self._torsion_index])
        )
        self.artists["phase_head"].set_data(
            [traj.modal_position[index, 0]],
            [np.rad2deg(traj.modal_position[index, self._torsion_index])],
        )

        # Sign flip so that "extracting energy from the structure" points up.
        extraction = -traj.control_power[index]
        for k, bar in enumerate(self.artists["bars"]):
            bar.set_height(extraction[k])
            bar.set_color(agent_color(k) if extraction[k] >= 0.0 else THEME["danger"])
        self.artists["clock"].set_text(f"t = {traj.time[index]:5.3f} s")

    _torsion_index: int = 0


def _build_figure(
    model: AeroelasticModel, trajectory: TrajectoryArrays, spec: AnimationSpec
) -> _Panels:
    scale = (
        spec.deformation_scale
        if spec.deformation_scale is not None
        else choose_deformation_scale(model, trajectory)
    )

    figure = plt.figure(figsize=_FIGURE_SIZE, dpi=_DPI)
    figure.patch.set_facecolor(THEME["background"])
    grid = GridSpec(
        3, 3,
        figure=figure,
        width_ratios=[2.35, 1.30, 1.55],
        height_ratios=[1.30, 1.0, 1.0],
        hspace=0.46, wspace=0.28,
        left=0.04, right=0.972, top=0.855, bottom=0.085,
    )

    wing_axes = figure.add_subplot(grid[0:2, 0:2])
    renderer = WingRenderer(model, wing_axes, deformation_scale=scale)
    _frame_wing_axes(wing_axes, model, trajectory, scale)
    _label_surfaces(wing_axes, model)

    artists: dict = {"cursors": []}
    artists["bars"] = _power_panel(figure.add_subplot(grid[2, 0]), trajectory, model)
    artists["phase_tail"], artists["phase_head"] = _phase_panel(
        figure.add_subplot(grid[2, 1]), trajectory, model
    )
    artists["tip"] = _trace_panel(
        figure.add_subplot(grid[0, 2]), trajectory.time, trajectory.tip_plunge,
        "tip plunge   [m, down +]", THEME["accent"], artists["cursors"],
    )
    artists["deflection"] = _deflection_panel(
        figure.add_subplot(grid[1, 2]), trajectory, model, artists["cursors"]
    )
    artists["energy"] = _energy_panel(
        figure.add_subplot(grid[2, 2]), trajectory, artists["cursors"]
    )

    _write_header(figure, model, trajectory, spec, scale)
    artists["clock"] = figure.text(
        0.972, 0.902, "", ha="right", va="center",
        color=THEME["text"], fontsize=11.5, family="monospace",
    )

    panels = _Panels(figure, renderer, trajectory, artists)
    panels._torsion_index = model.wing.n_bending
    return panels


def _frame_wing_axes(axes, model, trajectory: TrajectoryArrays, scale: float) -> None:
    """Fit the wing view to its cell, once, at a true 1:1 screen aspect.

    Two things make this fiddly. First, the undeformed planform already occupies
    a wide vertical band on screen -- the axonometric projection maps the 6 m
    span onto screen-vertical as well -- so sizing the view from the deformation
    alone underestimates the extent badly. Second, matplotlib's
    ``set_aspect("equal")`` either shrinks the axes box (leaving dead space in
    the cell) or inflates the limits (letterboxing the wing). Neither fills the
    cell, so the aspect is enforced by hand against the measured box.
    """
    wing = model.wing
    corners_span = np.asarray([0.0, wing.semispan, wing.semispan, 0.0])
    corners_chord = np.asarray([0.0, 0.0, wing.chord, wing.chord])
    base_x, base_y = project(corners_span, corners_chord, np.zeros(4))

    peak_plunge = float(np.abs(trajectory.tip_plunge).max())
    peak_twist = float(np.abs(trajectory.tip_twist).max())
    motion = scale * (peak_plunge + 0.7 * wing.chord * peak_twist)

    x_center = 0.5 * (base_x.min() + base_x.max())
    x_half = 0.5 * (base_x.max() - base_x.min()) * 1.06
    y_center = 0.5 * (base_y.min() + base_y.max())
    y_half = 0.5 * (base_y.max() - base_y.min()) + motion * 1.10 + 0.10

    # Grow whichever axis is short so that one metre reads the same either way.
    axes.figure.canvas.draw()
    box = axes.get_window_extent()
    box_aspect = box.width / max(box.height, 1.0)
    if x_half / y_half < box_aspect:
        x_half = y_half * box_aspect
    else:
        y_half = x_half / box_aspect

    axes.set_aspect("auto")
    axes.set_xlim(x_center - x_half, x_center + x_half)
    axes.set_ylim(y_center - y_half, y_center + y_half)


def _trace_panel(axes, time, series, label, color, cursors):
    _style_axes(axes, label)
    axes.plot(time, series, color=color, lw=0.8, alpha=0.20)
    (line,) = axes.plot([], [], color=color, lw=1.6)
    axes.set_xlim(time[0], time[-1])
    limit = max(float(np.abs(series).max()) * 1.15, 1e-9)
    axes.set_ylim(-limit, limit)
    axes.axhline(0.0, color=THEME["grid"], lw=0.8)
    cursors.append(axes.axvline(time[0], color=THEME["text_dim"], lw=0.8, ls=":"))
    return line


def _deflection_panel(axes, trajectory, model, cursors):
    _style_axes(axes, "surface deflection   [deg, TE down +]")
    degrees = np.rad2deg(trajectory.deflection)
    lines = []
    for k, surface in enumerate(model.wing.surfaces):
        color = agent_color(k)
        axes.plot(trajectory.time, degrees[:, k], color=color, lw=0.7, alpha=0.18)
        (line,) = axes.plot([], [], color=color, lw=1.5, label=surface.name)
        lines.append(line)

    # Scale to the deflections actually used; pinning the axis to the travel
    # limit squashes a 4-degree signal into a sliver and hides the phasing.
    travel = np.rad2deg(max(s.max_deflection for s in model.wing.surfaces))
    limit = max(float(np.abs(degrees).max()) * 1.35, travel * 0.2)
    if limit >= travel * 0.75:
        limit = travel * 1.15
        for sign in (-1.0, 1.0):
            axes.axhline(
                sign * travel, color=THEME["danger"], lw=0.7, ls="--", alpha=0.6
            )
        axes.text(
            0.99, 0.04, "dashed = travel limit", transform=axes.transAxes,
            ha="right", color=THEME["danger"], fontsize=7, alpha=0.8,
        )
    axes.set_xlim(trajectory.time[0], trajectory.time[-1])
    axes.set_ylim(-limit, limit)
    axes.axhline(0.0, color=THEME["grid"], lw=0.8)
    cursors.append(axes.axvline(trajectory.time[0], color=THEME["text_dim"], lw=0.8, ls=":"))
    axes.legend(
        loc="lower left", fontsize=7, frameon=False, ncol=3,
        labelcolor=THEME["text_dim"], handlelength=1.2, columnspacing=0.9,
        borderaxespad=0.2,
    )
    return lines


def _energy_panel(axes, trajectory, cursors):
    _style_axes(axes, "structural energy   [J]")
    energy = np.maximum(trajectory.energy, 1e-12)
    ceiling = float(energy.max())
    floor = max(ceiling * 1e-7, float(energy.min()) * 0.4, 1e-12)
    axes.plot(trajectory.time, energy, color=THEME["accent"], lw=0.8, alpha=0.20)
    (line,) = axes.plot([], [], color=THEME["accent"], lw=1.6)
    axes.set_yscale("log")
    axes.set_xlim(trajectory.time[0], trajectory.time[-1])
    axes.set_ylim(floor, ceiling * 3.0)
    axes.set_xlabel("time   [s]", color=THEME["text_dim"], fontsize=8)
    cursors.append(axes.axvline(trajectory.time[0], color=THEME["text_dim"], lw=0.8, ls=":"))
    return line


def _phase_panel(axes, trajectory, model):
    """Bending versus torsion amplitude -- the flutter mechanism itself."""
    _style_axes(axes, "bending  vs  torsion   (mode 1)")
    torsion_index = model.wing.n_bending
    bending = trajectory.modal_position[:, 0]
    torsion = np.rad2deg(trajectory.modal_position[:, torsion_index])

    axes.plot(bending, torsion, color=THEME["text_dim"], lw=0.6, alpha=0.30)
    (tail,) = axes.plot([], [], color=THEME["safe"], lw=1.4)
    (head,) = axes.plot([], [], "o", color=THEME["accent"], ms=4.5)

    x_limit = max(float(np.abs(bending).max()) * 1.2, 1e-6)
    y_limit = max(float(np.abs(torsion).max()) * 1.2, 1e-6)
    axes.set_xlim(-x_limit, x_limit)
    axes.set_ylim(-y_limit, y_limit)
    axes.axhline(0.0, color=THEME["grid"], lw=0.8)
    axes.axvline(0.0, color=THEME["grid"], lw=0.8)
    axes.set_xlabel("bending  [m]", color=THEME["text_dim"], fontsize=7.5, labelpad=1)
    axes.set_ylabel("torsion  [deg]", color=THEME["text_dim"], fontsize=7.5, labelpad=1)
    axes.text(
        0.03, 0.03, "an opening orbit is flutter", transform=axes.transAxes,
        ha="left", va="bottom", color=THEME["text_dim"], fontsize=7, style="italic",
    )
    return tail, head


def _power_panel(axes, trajectory, model):
    _style_axes(
        axes,
        "energy extraction per agent   $-P_k = -(\\dot{q}^{\\mathsf{T}} b_k)\\,\\delta_k$   [W]",
    )
    names = [surface.name.replace("_", "\n") for surface in model.wing.surfaces]
    positions = np.arange(len(names))
    bars = axes.bar(positions, np.zeros(len(names)), width=0.5)
    for k, bar in enumerate(bars):
        bar.set_color(agent_color(k))

    # A single startup spike would otherwise flatten every later frame.
    magnitude = np.abs(trajectory.control_power)
    peak = float(magnitude.max())
    if peak < 1e-3:
        # Open loop: no control input at all. Autoscaling to numerical noise
        # would draw a meaningless 1e-6 axis, so say what is happening instead.
        axes.set_ylim(-1.0, 1.0)
        axes.text(
            0.5, 0.5, "no control input", transform=axes.transAxes,
            ha="center", va="center", color=THEME["text_dim"],
            fontsize=11, style="italic",
        )
    else:
        limit = float(np.percentile(magnitude, 99.0)) * 1.3
        axes.set_ylim(-limit, limit)
    axes.set_xlim(-0.62, len(names) - 0.38)
    axes.set_xticks(positions)
    axes.set_xticklabels(names, color=THEME["text_dim"], fontsize=7.5)
    axes.axhline(0.0, color=THEME["text_dim"], lw=1.0)
    axes.text(
        0.985, 0.90, "damping", transform=axes.transAxes, ha="right",
        color=THEME["safe"], fontsize=8,
    )
    axes.text(
        0.985, 0.06, "exciting", transform=axes.transAxes, ha="right",
        color=THEME["danger"], fontsize=8,
    )
    axes.text(
        0.015, 0.90, "closed-form difference reward", transform=axes.transAxes,
        ha="left", color=THEME["text_dim"], fontsize=7, style="italic",
    )
    return list(bars)


def _label_surfaces(axes, model) -> None:
    """Surface names pinned below the view, clear of the deforming mesh.

    Anchoring them to the trailing edge puts them underneath the wing as soon as
    it deflects, so they sit on a fixed baseline instead.
    """
    wing = model.wing
    baseline = axes.get_ylim()[0]
    for k, surface in enumerate(wing.surfaces):
        mid_span = 0.5 * (surface.y_start_frac + surface.y_end_frac) * wing.semispan
        x, _ = project(
            np.asarray([mid_span]), np.asarray([wing.chord]), np.asarray([0.0])
        )
        axes.text(
            float(x[0]), baseline + 0.06 * (axes.get_ylim()[1] - baseline),
            surface.name.replace("_", " "),
            color=agent_color(k), fontsize=8.5, ha="center", va="bottom", zorder=6,
        )


def _write_header(figure, model, trajectory, spec: AnimationSpec, scale: float) -> None:
    figure.text(
        0.04, 0.955, spec.title, color=THEME["text"], fontsize=16.5,
        va="center", fontweight="semibold",
    )
    speed_ratio = ""
    if spec.flutter_speed:
        speed_ratio = f"   ·   U/U_f = {trajectory.airspeed / spec.flutter_speed:.3f}"
    detail = (
        f"{model.wing.name} wing   ·   U = {trajectory.airspeed:.1f} m/s{speed_ratio}"
        f"   ·   {model.wing.n_surfaces} control surfaces   ·   {model.n_states} plant states"
    )
    if spec.subtitle:
        detail = f"{detail}\n{spec.subtitle}"
    figure.text(0.04, 0.893, detail, color=THEME["text_dim"], fontsize=9.5, va="center")

    interval = float(trajectory.time[1] - trajectory.time[0])
    slow_motion = 1.0 / (spec.stride * spec.fps * interval)
    exaggeration = (
        "true scale" if scale < 1.05 else f"deformation × {scale:.0f}"
    )
    figure.text(
        0.972, 0.955,
        f"{exaggeration}   ·   {slow_motion:.0f}× slow motion",
        ha="right", va="center", color=THEME["text_dim"], fontsize=9,
    )


def _style_axes(axes, label: str) -> None:
    axes.set_facecolor(THEME["panel"])
    axes.set_title(label, color=THEME["text_dim"], fontsize=8.5, loc="left", pad=6)
    axes.tick_params(colors=THEME["text_dim"], labelsize=7.5, length=2)
    axes.grid(True, color=THEME["grid"], lw=0.5, alpha=0.7)
    for spine in axes.spines.values():
        spine.set_color(THEME["grid"])


def _open_writer(spec: AnimationSpec):
    import imageio.v2 as imageio

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.output_path.suffix.lower() == ".gif":
        return imageio.get_writer(spec.output_path, mode="I", fps=spec.fps, loop=0)
    return imageio.get_writer(
        spec.output_path, fps=spec.fps, codec="libx264", quality=8,
        macro_block_size=None,
    )


def _grab_frame(figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
