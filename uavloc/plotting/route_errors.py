"""Position-error charts over simulated flights: INS drift, end-error distribution, error over time, one long flight."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

FLIGHT_KEY = ["village", "route_index"]
IQR_BAND_ALPHA = 0.2
INS_LINE_ALPHA = 0.7
INS_LINE_WIDTH_PT = 1.0
QUARTILES = (0.25, 0.75)
CDF_Y_MAX = 1.02
DISTANCE_LABEL = "Пройдена відстань, м"
TIME_LABEL = "Час польоту, с"
ERROR_LABEL = "Похибка позиції, м"
ROUTE_DISTANCE_LABEL = "Пройдена відстань, км"


def evenly_ranked_flights(flights: pd.DataFrame, count: int, by: str = "end_error_m") -> pd.DataFrame:
    """``count`` flights spread evenly over the ranking by ``by``, so the draw is deterministic and covers the range."""
    ranked = flights.sort_values(by).reset_index(drop=True)
    return ranked.iloc[np.linspace(0, len(ranked) - 1, count).round().astype(int)]


def plot_ins_drift(ax: Axes, fixes: pd.DataFrame, flights: pd.DataFrame, colour: str) -> None:
    """INS error against distance flown, one thin line per flight in ``flights``."""
    for _, key in flights[FLIGHT_KEY].iterrows():
        flight = fixes[(fixes["village"] == key["village"]) & (fixes["route_index"] == key["route_index"])]
        ax.plot(flight["distance_m"], flight["error_after_m"], color=colour, alpha=INS_LINE_ALPHA,
                linewidth=INS_LINE_WIDTH_PT)
    ax.set_xlabel(DISTANCE_LABEL)
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def plot_end_error_cdf(ax: Axes, end_errors: Mapping[str, pd.Series], colours: Mapping[str, str]) -> None:
    """Share of flights whose end error is at most the abscissa, one step curve per method."""
    x_max = max(errors.max() for errors in end_errors.values())
    for label, errors in end_errors.items():
        values = np.sort(errors.to_numpy())
        share = np.arange(1, len(values) + 1) / len(values)
        ax.step(np.concatenate([[0.0], values, [x_max]]), np.concatenate([[0.0], share, [1.0]]), where="post",
                color=colours[label], label=label)
    ax.set_xlabel("Кінцева похибка, м")
    ax.set_ylabel("Частка маршрутів")
    ax.set_xlim(left=0)
    ax.set_ylim(0, CDF_Y_MAX)


def plot_error_over_time(ax: Axes, fixes_by_label: Mapping[str, pd.DataFrame], colours: Mapping[str, str],
                         max_time_s: float) -> None:
    """Median error over flights against flight time, with the interquartile band, up to ``max_time_s``."""
    for label, fixes in fixes_by_label.items():
        in_range = fixes[fixes["t_s"] <= max_time_s]
        by_time = in_range.groupby("t_s")["error_after_m"]
        median, low, high = by_time.median(), by_time.quantile(QUARTILES[0]), by_time.quantile(QUARTILES[1])
        ax.fill_between(median.index, low, high, color=colours[label], alpha=IQR_BAND_ALPHA, linewidth=0)
        ax.plot(median.index, median, color=colours[label], label=label)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xlim(0, max_time_s)
    ax.set_ylim(bottom=0)


def plot_long_route(ax: Axes, distance_km: np.ndarray, errors_m: Mapping[str, np.ndarray],
                    colours: Mapping[str, str]) -> None:
    """Position error of each method against distance flown along one route."""
    for label, error in errors_m.items():
        ax.plot(distance_km, error, color=colours[label], label=label)
    ax.set_xlabel(ROUTE_DISTANCE_LABEL)
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xlim(0, distance_km[-1])
    ax.set_ylim(bottom=0)
