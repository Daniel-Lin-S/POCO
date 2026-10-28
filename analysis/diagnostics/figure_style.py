"""Shared readable styling and measured layout for diagnostic PDFs.

Inputs are Matplotlib figures whose axes are already populated. The saver
places a wrapped title exactly TITLE_GAP_POINTS above the highest axis
annotation and an optional legend below all axes. Output is one cropped PDF.
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np

TITLE_FONT = 26
PANEL_FONT = 22
LABEL_FONT = 21
TICK_FONT = 18
LEGEND_FONT = 20
TITLE_GAP_POINTS = 16
LEGEND_GAP_POINTS = 14
TITLE_WRAP = 66
POINTS_PER_INCH = 72
SESSION_COLORS = {3: "#80aed1", 9: "#397aaa", 18: "#073b65"}
SESSION_WIDTHS = {3: 2.4, 9: 3.2, 18: 4.0}


def save_diagnostic(
    figure: Figure, path: Path, title: str,
    handles: list | None = None, labels: list[str] | None = None,
) -> None:
    """Save a figure with measured title and legend clearance.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure with fully labeled axes and no figure-level title yet.
    path : pathlib.Path
        Absolute PDF destination, including a pending PDF when replacing.
    title : str
        Readable title; explicit newlines are preserved when wrapping.
    handles : list or None, optional
        Legend artists, default None for no legend.
    labels : list[str] or None, optional
        Corresponding labels, default None for no legend.
    """
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute figure path: {path}.")
    if (handles is None) != (labels is None):
        raise ValueError("Legend handles and labels must be supplied together.")
    for text in figure.findobj(Text):
        if text.get_fontsize() < TICK_FONT:
            text.set_fontsize(TICK_FONT)
    figure.tight_layout(pad=0.65, h_pad=1.1, w_pad=0.9)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = Bbox.union([
        axis.get_tightbbox(renderer) for axis in figure.axes
    ])
    center = (bounds.x0 + bounds.x1) / (2 * figure.bbox.width)
    points_to_pixels = figure.dpi / POINTS_PER_INCH
    title_bottom = bounds.y1 + TITLE_GAP_POINTS * points_to_pixels
    heading = figure.suptitle(
        "\n".join(fill(line, TITLE_WRAP) for line in title.splitlines()),
        x=center, y=title_bottom / figure.bbox.height,
        va="bottom", fontsize=TITLE_FONT,
    )
    legend = None
    if handles is not None:
        if len(handles) != len(labels) or not handles:
            raise ValueError("Expected matching nonempty legend entries.")
        legend_top = bounds.y0 - LEGEND_GAP_POINTS * points_to_pixels
        legend = figure.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(center, legend_top / figure.bbox.height),
            ncol=len(labels), frameon=False, fontsize=LEGEND_FONT,
            borderaxespad=0, borderpad=0, columnspacing=1,
            handlelength=2.2,
        )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = (heading.get_window_extent(renderer).y0 - bounds.y1)
    gap /= points_to_pixels
    if not np.isfinite(gap) or abs(gap - TITLE_GAP_POINTS) > 0.5:
        raise ValueError(f"Expected a 16-point title gap, got {gap:.2f}.")
    if legend is not None:
        if legend.get_window_extent(renderer).y1 >= bounds.y0:
            raise ValueError("Figure legend overlaps the axes annotations.")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
