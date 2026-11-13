"""Shared matplotlib text sizing for website figures.

Figures are saved at ~7-11 in x 160 dpi and shown on the website in a column
roughly 700-850 px wide, so they are scaled down to about 60% of their pixel
size. At matplotlib's default 10 pt text, axis labels and ticks end up at an
effective 6-7 pt and are hard to read. These settings raise text sizes only;
they do not change data, colors, layout logic, or any computed result.

Usage (after importing pyplot):
    from plot_style import apply_web_style
    apply_web_style()
"""
from __future__ import annotations

import matplotlib as mpl

WEB_RC = {
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11.5,
    'legend.title_fontsize': 12,
    'figure.titlesize': 16,
}

# Size for small in-plot value labels (bar annotations), previously 8 pt.
ANNOTATION_SIZE = 11


def apply_web_style() -> None:
    """Apply web-readable text sizes to all subsequent matplotlib figures."""
    mpl.rcParams.update(WEB_RC)
