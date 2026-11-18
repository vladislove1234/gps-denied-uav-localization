"""The thesis chart look: the matplotlib default of the bachelor thesis, at the master's text width.

The bachelor charts are plain matplotlib: DejaVu Sans, a full black frame, a solid grey grid, the ``tab10``
colours, and a framed legend. This module reproduces that manner. The only decided differences are a white
background always, Ukrainian text, and a native width of 17.0 cm (or half of it), so the text keeps its size
when the PNG is placed in Word without scaling.

    fig, ax = new_figure(width="half", aspect=0.75)
    ax.plot(x, y, color=SPLIT_COLOURS["val"])
    save_thesis_figure(fig, FIGURES_DIR / "loss.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

__all__ = ["FULL_WIDTH_CM", "HALF_WIDTH_CM", "DEFAULT_ASPECT", "CM_PER_INCH", "DPI", "SERIES_COLOURS",
           "SPLIT_COLOURS", "SPLIT_LABELS", "METHOD_COLOURS", "series_colours", "use_thesis_style", "new_figure",
           "figure_legend", "save_thesis_figure", "figure_size_px"]

FigureWidth = Literal["full", "half"]

# Template "Шаблон магістерської роботи.dotm": A4 11906 twips wide, margins left 1417 and right 850 (1 cm = 567 twips)
TWIPS_PER_CM = 567.0
TEXT_WIDTH_CM = (11906 - 1417 - 850) / TWIPS_PER_CM            # 17.0
SIDE_BY_SIDE_GAP_CM = 0.4                                      # gap between two half-width figures in one Word row
FULL_WIDTH_CM = TEXT_WIDTH_CM
HALF_WIDTH_CM = (TEXT_WIDTH_CM - SIDE_BY_SIDE_GAP_CM) / 2
DEFAULT_ASPECT = 0.62                                          # height / width
CM_PER_INCH = 2.54
DPI = 300

# The bachelor charts are 10 x 5 in at font 10 and shrink to about 63 % in Word. A figure here is placed unscaled
# at 17 cm (6.7 in), so its text is set at 8-10 pt, which prints as legibly as the original.
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_PT = 10.0
TITLE_FONT_SIZE_PT = 11.0
TICK_FONT_SIZE_PT = 9.0
LEGEND_FONT_SIZE_PT = 9.0

# Marks, in points: matplotlib defaults
LINE_WIDTH_PT = 1.5
MARKER_SIZE_PT = 6.0
GRID_WIDTH_PT = 0.8
SPINE_WIDTH_PT = 0.8
LAYOUT_PAD_IN = 0.05                                           # outer padding around the figure, same for every figure

# Ink and surface: matplotlib defaults on a white page
SURFACE = "#ffffff"
INK = "#000000"
GRID = "#b0b0b0"

# matplotlib ``tab10``, in its fixed order. A colour follows the entity, never its rank.
SERIES_COLOURS: tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                                   "#7f7f7f", "#bcbd22", "#17becf")

# The bachelor training curves: train blue, validation orange.
SPLIT_COLOURS = {"train": SERIES_COLOURS[0], "val": SERIES_COLOURS[1]}
SPLIT_LABELS = {"train": "Навчальна вибірка", "val": "Валідаційна вибірка"}

# Entity -> colour for series that recur across figures. Extend here, never in a notebook.
# Loss configurations (tab10 order), and the positioning methods: INS red and the system blue as on the route maps.
METHOD_COLOURS: dict[str, str] = {
    "базова": SERIES_COLOURS[0],
    "з доданками для будівель": SERIES_COLOURS[1],
    "з доданками для доріг": SERIES_COLOURS[2],
    "ІНС": SERIES_COLOURS[3],
    "фільтр + NCC": SERIES_COLOURS[1],
    "фільтр + мережа зіставлення": SERIES_COLOURS[0],
    "система": SERIES_COLOURS[0],
}


def series_colours(names: Sequence[str]) -> dict[str, str]:
    """Colour per series name: fixed ones from ``METHOD_COLOURS``, the rest in free slot order."""
    free = [c for c in SERIES_COLOURS if c not in METHOD_COLOURS.values()]
    colours: dict[str, str] = {}
    for name in names:
        if name in METHOD_COLOURS:
            colours[name] = METHOD_COLOURS[name]
        elif free:
            colours[name] = free.pop(0)
        else:
            raise ValueError(f"No free palette slot for {name!r}: fold series into 'other' or use small multiples")
    return colours


def _rc_params() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
        "font.size": FONT_SIZE_PT,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": TITLE_FONT_SIZE_PT,
        "xtick.labelsize": TICK_FONT_SIZE_PT,
        "ytick.labelsize": TICK_FONT_SIZE_PT,
        "legend.fontsize": LEGEND_FONT_SIZE_PT,
        "legend.title_fontsize": LEGEND_FONT_SIZE_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK,
        "axes.linewidth": SPINE_WIDTH_PT,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": GRID_WIDTH_PT,
        "lines.linewidth": LINE_WIDTH_PT,
        "lines.markersize": MARKER_SIZE_PT,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES_COLOURS)),
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "axes.formatter.use_locale": False,      # decimal point, whatever the machine locale
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": False,
        "axes.unicode_minus": True,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": LAYOUT_PAD_IN,
        "figure.constrained_layout.h_pad": LAYOUT_PAD_IN,
    }


def use_thesis_style() -> None:
    """Make the thesis look the matplotlib default. Idempotent; ``new_figure`` calls it."""
    mpl.rcParams.update(_rc_params())


def figure_size_px(width: FigureWidth = "full", aspect: float = DEFAULT_ASPECT) -> tuple[int, int]:
    """Exact PNG size in pixels at ``DPI`` for the figure width and ``aspect`` = height / width."""
    width_cm = FULL_WIDTH_CM if width == "full" else HALF_WIDTH_CM
    width_px = round(width_cm / CM_PER_INCH * DPI)
    return width_px, round(width_px * aspect)


def new_figure(width: FigureWidth = "full", aspect: float = DEFAULT_ASPECT, nrows: int = 1, ncols: int = 1,
               **subplots_kwargs: Any) -> tuple[Figure, Any]:
    """Figure of exact printed width, with its axes from ``plt.subplots``.

    The size is snapped to whole pixels at 300 dpi, so the PNG is 17.0 cm (full) or 8.3 cm (half) wide to within
    0.01 cm, and ``constrained_layout`` fits everything inside it instead of resizing the canvas.
    """
    use_thesis_style()
    width_px, height_px = figure_size_px(width, aspect)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_px / DPI, height_px / DPI), dpi=DPI, **subplots_kwargs)
    return fig, axes


def figure_legend(fig: Figure, ax: Any, ncol: int | None = None, handles: Sequence[Any] | None = None,
                  title: str | None = None) -> None:
    """One legend above the plots, outside them so it never covers data.

    Entries come from ``ax`` unless ``handles`` (labelled artists) are given.
    """
    if handles is None:
        handles = ax.get_legend_handles_labels()[0]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol or len(handles), title=title)


def save_thesis_figure(fig: Figure, path: Path) -> tuple[int, int]:
    """Write ``fig`` as a 300 dpi PNG at the exact size it was created with; returns ``(width_px, height_px)``.

    ``bbox_inches="tight"`` is not used: it crops to the artists, so the width would change with the data (a long
    tick label, a legend) and a figure would no longer fit the text width or line up with its neighbour. The
    layout engine already keeps every artist inside the canvas with the same outer padding.
    """
    width_px, height_px = (round(side * DPI) for side in fig.get_size_inches())
    if not np.isclose(fig.get_size_inches() * DPI, (width_px, height_px)).all():
        raise ValueError("Figure size is not a whole number of pixels at 300 dpi: create it with new_figure")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, format="png")
    return width_px, height_px
