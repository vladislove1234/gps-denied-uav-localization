"""Grouped bars with the value written over each bar, in the thesis style."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes

BAR_GROUP_WIDTH = 0.72
VALUE_FORMAT = "{:.2f}"
VALUE_FONT_SIZE_PT = 8.0
VALUE_PAD_PT = 2.0
HEADROOM = 1.18


def plot_grouped_bars(ax: Axes, groups: Sequence[str], series: Mapping[str, Sequence[float]],
                      colours: Mapping[str, str]) -> None:
    """One cluster per group, one bar per series in ``series`` (label -> one value per group)."""
    bar_width = BAR_GROUP_WIDTH / len(series)
    positions = np.arange(len(groups))
    for i, (label, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * bar_width
        bars = ax.bar(positions + offset, values, bar_width, color=colours[label], label=label)
        ax.bar_label(bars, labels=[VALUE_FORMAT.format(v) for v in values], padding=VALUE_PAD_PT,
                     fontsize=VALUE_FONT_SIZE_PT)
    ax.set_xticks(positions, groups)
    ax.set_ylim(0, HEADROOM * max(max(values) for values in series.values()))
    ax.grid(axis="x", visible=False)
