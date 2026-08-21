"""Publication figure style for the manuscript.

The categorical palette is not a taste choice -- it was validated rather than
eyeballed. The order below passes the full six-check battery on a light surface:
lightness band, chroma floor, colour-vision-deficiency separation on every
adjacent pair, the normal-vision floor, and contrast against the surface. The
order matters, because the checks run on *adjacent* pairs; reordering can break
it. Assign these in fixed order and never cycle them.

The diverging map for signed quantities uses two hues with a **neutral grey**
midpoint. A rainbow or a yellow-midpoint map (``RdYlGn``) implies a meaningful
middle colour where there is only "zero", and it fails under colour-vision
deficiency.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#: Fixed categorical order, validated on a light surface. Do not reorder.
CONTROLLER_COLORS = (
    "#8d4f9e",  # purple
    "#e6820e",  # orange
    "#0f9b8e",  # teal
    "#3465a4",  # blue
    "#c9427d",  # magenta
    "#7f7f2f",  # olive
)

INK = "#1a1f26"
INK_MUTED = "#5b6672"
GRID = "#d8dee6"
SURFACE = "#ffffff"

#: Signed-quantity map: cool = suppressed (good), warm = growing (bad), grey at zero.
SUPPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "aegis_diverging",
    ["#1b4f8a", "#6f9bc4", "#dcdcda", "#dd8f76", "#b03a2e"],
)

#: Single-hue ramp for magnitudes.
MAGNITUDE_CMAP = LinearSegmentedColormap.from_list(
    "aegis_sequential", ["#eef3f8", "#9dbdd8", "#3465a4", "#16305c"]
)


def controller_color(index: int) -> str:
    """Colour for series ``index`` in fixed order."""
    if index >= len(CONTROLLER_COLORS):
        raise ValueError(
            f"only {len(CONTROLLER_COLORS)} validated categorical slots exist; "
            "fold extra series into 'other' or use small multiples rather than "
            "generating a new hue"
        )
    return CONTROLLER_COLORS[index]


def use_paper_style() -> None:
    """Apply publication rcParams: thin marks, recessive grid, serif text."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "text.color": INK,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
        }
    )


def symmetric_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    """Symmetric colour limit for a signed field, ignoring non-finite entries."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(max(np.abs(finite).max(), 1e-9))
