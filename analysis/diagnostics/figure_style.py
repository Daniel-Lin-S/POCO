"""Shared readable styling and measured layout for figures.

Inputs are Matplotlib figures whose axes are already populated. The saver
places a wrapped title exactly TITLE_GAP_POINTS above the highest axis
annotation and an optional legend below all axes. Output follows the
tracked figure format and PNG resolution in configs/plotting.
"""

import json
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
FIGURE_CONFIG = (
    Path(__file__).resolve().parents[2] /
    'configs' / 'plotting' / 'figure_output.json'
)
SUPPORTED_FORMATS = {'png', 'pdf'}


def load_figure_output() -> tuple[str, int]:
    """Read the project-wide output format and PNG resolution.

    Returns
    -------
    tuple[str, int]
        Figure suffix and PNG dots per inch.
    """
    with FIGURE_CONFIG.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    if (not isinstance(settings, dict) or
            set(settings) != {'format', 'png_dpi'}):
        raise ValueError(
            f'Expected format and png_dpi in {FIGURE_CONFIG}.'
        )
    kind = settings['format']
    dpi = settings['png_dpi']
    if kind not in SUPPORTED_FORMATS:
        raise ValueError(
            f'Figure format must be png or pdf, got {kind!r}.'
        )
    if type(dpi) is not int or dpi < 1:
        raise ValueError(
            f'PNG DPI must be a positive integer, got {dpi!r}.'
        )
    return f'.{kind}', dpi


FIGURE_SUFFIX, PNG_DPI = load_figure_output()


def save_plot(
    figure: Figure, path: Path | str, **kwargs: object,
) -> None:
    """Save one figure using the configured format and resolution.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to serialize.
    path : pathlib.Path or str
        Output path with the configured suffix.
    **kwargs : object
        Additional Matplotlib save options.
    """
    destination = Path(path)
    if destination.suffix.lstrip('.') not in SUPPORTED_FORMATS:
        raise ValueError(
            f'Figure suffix must be png or pdf, got {destination}.'
        )
    figure.savefig(
        destination,
        dpi=PNG_DPI if destination.suffix == '.png' else None,
        **kwargs,
    )


SESSION_COLORS = {3: "#80aed1", 9: "#397aaa", 18: "#073b65"}
SESSION_WIDTHS = {3: 2.4, 9: 3.2, 18: 4.0}


def save_diagnostic(
    figure: Figure, path: Path, title: str,
    handles: list | None = None, labels: list[str] | None = None,
    legend_columns: int | None = None,
) -> None:
    """Save a figure with measured title and legend clearance.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure with fully labeled axes and no figure-level title yet.
    path : pathlib.Path
        Absolute configured-format destination, including pending files.
    title : str
        Readable title; explicit newlines are preserved when wrapping.
    handles : list or None, optional
        Legend artists, default None for no legend.
    labels : list[str] or None, optional
        Corresponding labels, default None for no legend.
    legend_columns : int or None, optional
        Number of legend columns, default None for one row.
    """
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute figure path: {path}.")
    if (handles is None) != (labels is None):
        raise ValueError("Legend handles and labels must be supplied together.")
    if legend_columns is not None and legend_columns < 1:
        raise ValueError(
            f"Expected positive legend columns, got {legend_columns}."
        )
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
            ncol=legend_columns or len(labels),
            frameon=False, fontsize=LEGEND_FONT,
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
    save_plot(
        figure, path, bbox_inches="tight", pad_inches=0.12,
    )
    plt.close(figure)
