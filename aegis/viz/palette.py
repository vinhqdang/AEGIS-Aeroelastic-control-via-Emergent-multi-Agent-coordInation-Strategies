"""Single source of colour truth for every AEGIS figure and animation.

Agent colours are separated in hue *and* lightness so they survive greyscale
printing and the common forms of colour-vision deficiency -- a figure whose
agents are only distinguishable by hue is unusable in a printed paper.
"""

from __future__ import annotations

# Dark technical theme. Animations are watched on a screen; the paper figures
# use the same hues on a light ground via THEME_LIGHT.
THEME = {
    "background": "#0e1116",
    "panel": "#161b22",
    "grid": "#242c37",
    "text": "#d7dde5",
    "text_dim": "#7d8896",
    "wing": "#46586b",
    "wing_edge": "#71879e",
    "reference": "#3a4552",
    "accent": "#e8eaed",
    "danger": "#ff6b6b",
    "safe": "#5fd68a",
}

THEME_LIGHT = {
    "background": "#ffffff",
    "panel": "#f5f7fa",
    "grid": "#dde3ea",
    "text": "#1a1f26",
    "text_dim": "#5b6672",
    "wing": "#9fb1c4",
    "wing_edge": "#5a6b7d",
    "reference": "#c4ccd5",
    "accent": "#1a1f26",
    "danger": "#c92a2a",
    "safe": "#2f9e44",
}

#: Ordered inboard to outboard, matching the default Goland surface layout.
AGENT_COLORS = ("#f2a93b", "#35c4b5", "#c86bfa", "#ff7a9c", "#8fd14f", "#4d9dff")


def agent_color(index: int) -> str:
    """Colour for agent ``index``, cycling if there are more agents than colours."""
    return AGENT_COLORS[index % len(AGENT_COLORS)]
