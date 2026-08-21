"""Axonometric render of the deforming wing with its control surfaces.

The wing is drawn as a quad mesh projected through a fixed axonometric map into
an ordinary 2D axes. A hand-rolled projection is used rather than matplotlib's
3D toolkit because the surface has to be redrawn a few hundred times per
animation, and updating one PolyCollection's vertices is roughly an order of
magnitude cheaper than rebuilding a 3D surface per frame.

Geometry follows the sign conventions in :mod:`aegis.physics.wing`: the modal
solution gives plunge positive **down** and twist positive **nose-up**, so the
screen-vertical coordinate is the negation of the physical displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.collections import PolyCollection

from aegis.physics.aeroelastic import AeroelasticModel
from aegis.physics.thin_airfoil import flap_derivatives  # noqa: F401  (doc cross-ref)
from aegis.viz.palette import THEME, agent_color

# Axonometric view angles: a shallow yaw plus a modest downward tilt reads as
# "looking at the wing from ahead, above, and slightly outboard".
_YAW = np.deg2rad(22.0)
_TILT = np.deg2rad(26.0)


def project(
    span: np.ndarray, chord: np.ndarray, vertical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Axonometric projection of wing coordinates onto screen coordinates.

    ``span`` runs root to tip, ``chord`` runs leading to trailing edge, and
    ``vertical`` is positive up.
    """
    screen_x = span * np.cos(_YAW) + chord * np.sin(_YAW)
    depth = -span * np.sin(_YAW) + chord * np.cos(_YAW)
    screen_y = vertical + depth * np.sin(_TILT)
    return screen_x, screen_y


@dataclass(frozen=True)
class MeshLayout:
    """Precomputed, deformation-independent mesh bookkeeping."""

    span_nodes: np.ndarray       # (n_span + 1,)
    chord_nodes: np.ndarray      # (n_chord + 1,)
    plunge_rows: np.ndarray      # (n_span + 1, n_modes)
    twist_rows: np.ndarray       # (n_span + 1, n_modes)
    surface_of_cell: np.ndarray  # (n_span, n_chord) surface index or -1
    hinge_position: np.ndarray   # (n_span,) chordwise hinge station, or nan


class WingRenderer:
    """Draws and updates one deforming-wing view inside a matplotlib axes."""

    def __init__(
        self,
        model: AeroelasticModel,
        axes,
        n_span: int = 22,
        n_chord: int = 7,
        deformation_scale: float = 1.0,
        show_reference: bool = True,
    ):
        if n_span < 4 or n_chord < 3:
            raise ValueError("mesh needs at least 4 spanwise and 3 chordwise cells")
        self.model = model
        self.axes = axes
        self.deformation_scale = deformation_scale
        self.layout = _build_layout(model, n_span, n_chord)

        self._n_span = n_span
        self._n_chord = n_chord
        self._elastic_axis = model.wing.ea_frac * model.wing.chord

        if show_reference:
            self._draw_reference()
        self._collection = PolyCollection(
            self._vertices(np.zeros(model.n_modes), np.zeros(model.wing.n_surfaces)),
            linewidths=0.6,
        )
        self._collection.set_edgecolor(THEME["wing_edge"])
        axes.add_collection(self._collection)
        self._face_colors = self._base_face_colors()
        self._configure_axes()

    # ---------------------------------------------------------------- drawing
    def update(self, modal_position: np.ndarray, deflection: np.ndarray) -> None:
        """Redraw the surface for one instant of the rollout."""
        self._collection.set_verts(self._vertices(modal_position, deflection))
        self._collection.set_facecolor(self._shaded_colors(modal_position))

    def _vertices(
        self, modal_position: np.ndarray, deflection: np.ndarray
    ) -> list[np.ndarray]:
        vertical = self._node_heights(modal_position, deflection)
        span_grid, chord_grid = np.meshgrid(
            self.layout.span_nodes, self.layout.chord_nodes, indexing="ij"
        )
        screen_x, screen_y = project(span_grid, chord_grid, vertical)

        quads = []
        for i in range(self._n_span):
            for j in range(self._n_chord):
                quads.append(
                    np.column_stack(
                        (
                            [
                                screen_x[i, j], screen_x[i + 1, j],
                                screen_x[i + 1, j + 1], screen_x[i, j + 1],
                            ],
                            [
                                screen_y[i, j], screen_y[i + 1, j],
                                screen_y[i + 1, j + 1], screen_y[i, j + 1],
                            ],
                        )
                    )
                )
        return quads

    def _node_heights(
        self, modal_position: np.ndarray, deflection: np.ndarray
    ) -> np.ndarray:
        """Screen-vertical position of every mesh node, shape (n_span+1, n_chord+1)."""
        scale = self.deformation_scale
        plunge = scale * (self.layout.plunge_rows @ modal_position)
        twist = scale * (self.layout.twist_rows @ modal_position)

        chord = self.layout.chord_nodes[None, :]
        lever = chord - self._elastic_axis
        # Physical displacement is downward-positive; screen vertical is up.
        vertical = -(plunge[:, None] + lever * twist[:, None])

        # Flap rotation about the hinge, applied only aft of the hinge line.
        for k, surface in enumerate(self.model.wing.surfaces):
            hinge = surface.hinge_frac * self.model.wing.chord
            band = _node_band_mask(self.layout.span_nodes, surface)
            aft = np.clip(self.layout.chord_nodes - hinge, 0.0, None)
            vertical[band] -= np.outer(np.ones(band.sum()), aft) * deflection[k]
        return vertical

    def _base_face_colors(self) -> np.ndarray:
        """RGB per cell: neutral wing skin, agent colour on each control surface."""
        from matplotlib.colors import to_rgb

        colors = np.empty((self._n_span * self._n_chord, 3))
        wing_rgb = to_rgb(THEME["wing"])
        for i in range(self._n_span):
            for j in range(self._n_chord):
                surface_index = self.layout.surface_of_cell[i, j]
                colors[i * self._n_chord + j] = (
                    wing_rgb if surface_index < 0 else to_rgb(agent_color(surface_index))
                )
        return colors

    def _shaded_colors(self, modal_position: np.ndarray) -> np.ndarray:
        """Modulate cell brightness by local twist as a cheap surface-normal cue."""
        twist = self.layout.twist_rows @ modal_position
        cell_twist = 0.5 * (twist[:-1] + twist[1:])
        span = max(float(np.abs(cell_twist).max()), 1e-6)
        shade = 0.72 + 0.38 * (cell_twist / span)
        shade = np.repeat(shade, self._n_chord)[:, None]
        return np.clip(self._face_colors * shade, 0.0, 1.0)

    def _draw_reference(self) -> None:
        """Faint outline of the undeformed planform for scale."""
        wing = self.model.wing
        corners_span = np.array([0.0, wing.semispan, wing.semispan, 0.0, 0.0])
        corners_chord = np.array([0.0, 0.0, wing.chord, wing.chord, 0.0])
        x, y = project(corners_span, corners_chord, np.zeros_like(corners_span))
        self.axes.plot(x, y, color=THEME["reference"], lw=1.0, ls="--", zorder=0)

    def _configure_axes(self) -> None:
        """Cosmetics only. Limits and aspect are owned by the caller.

        The renderer cannot size the view sensibly on its own -- that needs the
        trajectory extent and the axes box -- so it deliberately does not try.
        """
        self.axes.set_facecolor(THEME["panel"])
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        self.axes.set_xticks([])
        self.axes.set_yticks([])


def _build_layout(model: AeroelasticModel, n_span: int, n_chord: int) -> MeshLayout:
    wing = model.wing
    span_nodes = np.linspace(0.0, wing.semispan, n_span + 1)
    chord_nodes = np.linspace(0.0, wing.chord, n_chord + 1)

    plunge_rows = np.empty((span_nodes.size, model.n_modes))
    twist_rows = np.empty((span_nodes.size, model.n_modes))
    for index, y in enumerate(span_nodes):
        plunge_rows[index], twist_rows[index] = model.station_rows(y / wing.semispan)

    span_centers = 0.5 * (span_nodes[:-1] + span_nodes[1:])
    chord_centers = 0.5 * (chord_nodes[:-1] + chord_nodes[1:])
    surface_of_cell = np.full((n_span, n_chord), -1, dtype=int)
    hinge_position = np.full(n_span, np.nan)

    for k, surface in enumerate(wing.surfaces):
        inside_span = (span_centers >= surface.y_start_frac * wing.semispan) & (
            span_centers <= surface.y_end_frac * wing.semispan
        )
        hinge = surface.hinge_frac * wing.chord
        aft = chord_centers > hinge
        surface_of_cell[np.ix_(inside_span, aft)] = k
        hinge_position[inside_span] = hinge

    return MeshLayout(
        span_nodes=span_nodes,
        chord_nodes=chord_nodes,
        plunge_rows=plunge_rows,
        twist_rows=twist_rows,
        surface_of_cell=surface_of_cell,
        hinge_position=hinge_position,
    )


def _node_band_mask(span_nodes: np.ndarray, surface) -> np.ndarray:
    """Nodes lying inside a surface's spanwise band."""
    return np.ones_like(span_nodes, dtype=bool) & (
        span_nodes >= surface.y_start_frac * span_nodes[-1]
    ) & (span_nodes <= surface.y_end_frac * span_nodes[-1])
