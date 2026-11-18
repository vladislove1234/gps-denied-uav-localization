"""Static pictures of one simulated flight: the tracks over the map, and error over time."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, east_north_m, east_north_m_array
from uavloc.simulation.flight import FlightResult

RGB_MAX = 255.0
TRACK_COLUMNS = {"truth": ("true_lat", "true_lon"), "ins": ("ins_lat", "ins_lon"), "estimate": ("est_lat", "est_lon")}
TRACK_RGB = {"truth": (44, 160, 44), "ins": (214, 39, 40), "estimate": (31, 119, 180)}
TRACK_LABELS = {"truth": "true", "ins": "INS only", "estimate": "corrected"}
TRACK_ERROR_COLUMNS = {"ins": "ins_error_m", "estimate": "est_error_m"}
DECISION_MARKERS = {"accept": "o", "across": "D", "reject": "x", "out-of-raster": "s"}
UNFILLED_MARKERS = ("x",)
FILTER_DECISION_MARKERS = {"update": "o", "blind": "x", "out-of-raster": "s"}   # filter estimators' fix decisions
MAP_MARGIN_M = 40.0


def track_color(track: str) -> tuple[float, float, float]:
    return tuple(c / RGB_MAX for c in TRACK_RGB[track])


def track_xy_m(result: FlightResult, track: str, origin: LatLon) -> tuple[np.ndarray, np.ndarray]:
    """`(east_m, north_m)` of every pose of one track, relative to `origin`."""
    lat_column, lon_column = TRACK_COLUMNS[track]
    return east_north_m_array(origin, result.track[lat_column].to_numpy(), result.track[lon_column].to_numpy())


def plot_tracks(result: FlightResult, basemap: GeoImage | None = None, ax: Axes | None = None,
                tracks: tuple[str, ...] = ("truth", "ins", "estimate"), margin_m: float = MAP_MARGIN_M) -> Axes:
    """Tracks in metres from the start, over the map. Legend entries carry each track's end-point error."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    origin = result.poses[0].position
    if basemap is not None:
        (left, top), (right, bottom) = east_north_m(origin, basemap.top_left), east_north_m(origin, basemap.bottom_right)
        ax.imshow(basemap.pixels, extent=(left, right, bottom, top))
    all_x, all_y = [], []
    for track in tracks:
        x, y = track_xy_m(result, track, origin)
        error = result.track[TRACK_ERROR_COLUMNS[track]].iloc[-1] if track in TRACK_ERROR_COLUMNS else None
        label = TRACK_LABELS[track] + (f" (end error {error:.1f} m)" if error is not None else "")
        ax.plot(x, y, color=track_color(track), lw=2, ls="--" if track == "ins" else "-", label=label)
        ax.plot(x[-1], y[-1], "o", color=track_color(track), ms=8, mec="white")
        all_x.append(x), all_y.append(y)
    plot_fixes(result, ax, origin)
    ax.plot(0, 0, "^", color="white", ms=10, mec="black", label="start")
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    ax.set_xlim(x.min() - margin_m, x.max() + margin_m)
    ax.set_ylim(y.min() - margin_m, y.max() + margin_m)
    ax.set_aspect("equal")
    ax.set_xlabel("east, m"), ax.set_ylabel("north, m")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.85)
    return ax


def plot_fixes(result: FlightResult, ax: Axes, origin: LatLon) -> None:
    """Where each attempted fix put the drone, one marker per decision."""
    for decision, marker in DECISION_MARKERS.items():
        points = [east_north_m(origin, fix.position) for fix in result.fixes.values() if fix.decision == decision]
        if points:
            style = {"color": "black"} if marker in UNFILLED_MARKERS else {"color": "white", "edgecolor": "black"}
            ax.scatter(*np.array(points).T, marker=marker, s=28, lw=0.8, zorder=5,
                       label=f"fix: {decision} ({len(points)})", **style)


def plot_errors(result: FlightResult, ax: Axes | None = None) -> Axes:
    """Distance to the true position over time, with the correction attempts marked."""
    ax = ax or plt.subplots(figsize=(10, 3.5))[1]
    track = result.track
    for name, column in TRACK_ERROR_COLUMNS.items():
        ax.plot(track["t_s"], track[column], color=track_color(name), lw=1.8, ls="--" if name == "ins" else "-",
                label=f"{TRACK_LABELS[name]} (end {track[column].iloc[-1]:.1f} m)")
    attempts = track[track["fix_attempted"]]
    for decision, marker in DECISION_MARKERS.items():
        hits = attempts[attempts["decision"] == decision]
        if len(hits):
            ax.scatter(hits["t_s"], hits["est_error_m"], marker=marker, s=24, color="black", zorder=5, label=decision)
    ax.set_xlabel("time since GNSS loss, s"), ax.set_ylabel("position error, m")
    ax.grid(alpha=0.3), ax.legend(loc="upper left")
    return ax


def plot_end_error_cdf(end_errors_m: dict[str, np.ndarray], ax: Axes) -> None:
    """Empirical CDF of per-flight end-point errors, one line per estimator."""
    for name, errors in end_errors_m.items():
        values = np.sort(np.asarray(errors, dtype=float))
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{name} (mean {values.mean():.1f} m)")
    ax.set(xscale="log", xlabel="end-point error, m", ylabel="share of flights")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_arm_tracks(tracks: Mapping[str, pd.DataFrame], colours: Mapping[str, str], basemap: GeoImage | None = None,
                    ax: Axes | None = None, margin_m: float = MAP_MARGIN_M) -> Axes:
    """One route flown by several estimators: truth and INS once, then each estimator's track (`fly` track frames)."""
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    first = next(iter(tracks.values()))
    origin = LatLon(first["true_lat"].iloc[0], first["true_lon"].iloc[0])
    if basemap is not None:
        (left, top), (right, bottom) = east_north_m(origin, basemap.top_left), east_north_m(origin, basemap.bottom_right)
        ax.imshow(basemap.pixels, extent=(left, right, bottom, top))
    lines = {"true": (first, TRACK_COLUMNS["truth"], track_color("truth"), "-", None),
             "INS only": (first, TRACK_COLUMNS["ins"], track_color("ins"), "--", "ins_error_m")}
    lines.update({name: (track, TRACK_COLUMNS["estimate"], colours[name], "-", "est_error_m")
                  for name, track in tracks.items()})
    all_x, all_y = [], []
    for label, (track, (lat_column, lon_column), colour, style, error_column) in lines.items():
        x, y = east_north_m_array(origin, track[lat_column].to_numpy(), track[lon_column].to_numpy())
        suffix = f" (end {track[error_column].iloc[-1]:.1f} m)" if error_column else ""
        ax.plot(x, y, color=colour, lw=1.8, ls=style, label=label + suffix)
        ax.plot(x[-1], y[-1], "o", color=colour, ms=6, mec="white")
        all_x.append(x), all_y.append(y)
    ax.plot(0, 0, "^", color="white", ms=10, mec="black", label="start")
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    ax.set_xlim(x.min() - margin_m, x.max() + margin_m)
    ax.set_ylim(y.min() - margin_m, y.max() + margin_m)
    ax.set_aspect("equal")
    ax.set_xlabel("east, m"), ax.set_ylabel("north, m")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.85, fontsize=8)
    return ax


def plot_arm_errors(tracks: Mapping[str, pd.DataFrame], fixes: Mapping[str, pd.DataFrame],
                    colours: Mapping[str, str], ax: Axes | None = None) -> Axes:
    """Distance to the truth over time per estimator; each fix marked by its decision (`x` = blind, skipped)."""
    ax = ax or plt.subplots(figsize=(10, 3.5))[1]
    first = next(iter(tracks.values()))
    ax.plot(first["t_s"], first["ins_error_m"], color=track_color("ins"), lw=1.5, ls="--", label="INS only")
    for name, track in tracks.items():
        ax.plot(track["t_s"], track["est_error_m"], color=colours[name], lw=1.6, label=name)
        for decision, marker in FILTER_DECISION_MARKERS.items():
            hits = fixes[name][fixes[name]["decision"] == decision]
            if len(hits):
                ax.scatter(hits["t_s"], hits["error_after_m"], marker=marker, s=14, color=colours[name], zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("time since GNSS loss, s"), ax.set_ylabel("position error, m")
    ax.grid(alpha=0.3), ax.legend(loc="upper left", fontsize=8)
    return ax
