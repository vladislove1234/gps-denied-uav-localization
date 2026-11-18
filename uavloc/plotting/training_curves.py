"""Training-curve axes in the thesis style, drawn from the long scalar table of ``uavloc.track.load_scalars``."""

from __future__ import annotations

from typing import Mapping

import pandas as pd
from matplotlib.axes import Axes

from uavloc.plotting.thesis_style import SPLIT_COLOURS, SPLIT_LABELS

STEP_LABEL = "Крок навчання"
TRAIN_LINESTYLE = "--"
VAL_LINESTYLE = "-"


def scalar_series(table: pd.DataFrame, run: str, tag: str) -> pd.DataFrame:
    """Rows of one run and tag, ordered by step; raises if there are none."""
    series = table[(table["run"] == run) & (table["tag"] == tag)].sort_values("step")
    if series.empty:
        raise KeyError(f"No scalars for tag {tag!r} in run {run!r}")
    return series


def plot_curve(ax: Axes, series: pd.DataFrame, colour: str, label: str, linestyle: str = "-") -> None:
    ax.plot(series["step"], series["value"], color=colour, label=label, linestyle=linestyle)


def plot_train_val(ax: Axes, table: pd.DataFrame, run: str, metric: str, ylabel: str) -> None:
    """``train/<metric>`` and ``val/<metric>`` of one run, in the fixed split colours."""
    for split in ("train", "val"):
        plot_curve(ax, scalar_series(table, run, f"{split}/{metric}"), SPLIT_COLOURS[split], SPLIT_LABELS[split])
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(ylabel)


def plot_config_train_val(ax: Axes, table: pd.DataFrame, run_labels: Mapping[str, str], colours: Mapping[str, str],
                          metric: str) -> None:
    """Train (dashed) and validation (solid) ``metric`` of every run, coloured by its ``run_labels`` entry."""
    for run, label in run_labels.items():
        plot_curve(ax, scalar_series(table, run, f"train/{metric}"), colours[label], label, TRAIN_LINESTYLE)
        plot_curve(ax, scalar_series(table, run, f"val/{metric}"), colours[label], label, VAL_LINESTYLE)
    ax.set_xlabel(STEP_LABEL)


def plot_run_comparison(ax: Axes, table: pd.DataFrame, tag: str, run_labels: Mapping[str, str],
                        colours: Mapping[str, str]) -> None:
    """One curve of ``tag`` per run; ``run_labels`` maps run to legend text, ``colours`` legend text to colour."""
    for run, label in run_labels.items():
        plot_curve(ax, scalar_series(table, run, tag), colours[label], label)
    ax.set_xlabel(STEP_LABEL)
