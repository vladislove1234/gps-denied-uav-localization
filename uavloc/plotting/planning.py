"""Route-planner figures: rasters in lon/lat with routes and their blind stretches on top."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.planning.grid import CostGrid
from uavloc.planning.routes import BLIND_THRESHOLD, Route, blind_profile, sample_polyline

ROUTE_COLOURS = ("#1f77b4", "#9467bd", "#17becf", "#e377c2", "#8c564b")
BLIND_COLOUR = "#d62728"
ROUTE_WIDTH = 2.0
BLIND_WIDTH = 4.0
MARKER_SIZE = 60


def show_raster(ax: plt.Axes, raster: GeoImage, **imshow_kwargs: object) -> None:
    """`raster` in lon/lat axes, with the aspect that makes metres square."""
    tl, br = raster.top_left, raster.bottom_right
    ax.imshow(raster.pixels.squeeze(), extent=(tl.lon, br.lon, br.lat, tl.lat), **imshow_kwargs)
    ax.set_aspect(1 / math.cos(math.radians((tl.lat + br.lat) / 2)))
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")


def plot_route(ax: plt.Axes, grid: CostGrid, route: Route, colour: str, label: str) -> None:
    """The route as a line, its blind samples overdrawn in red."""
    points = [tuple(reversed(grid.raster.to_px(p))) for p in route.waypoints]
    _, blind = blind_profile(grid, points)
    _, coords = sample_polyline(grid, points)   # the profile's own samples
    lons = np.array([grid.latlon_of(r, c).lon for r, c in coords.T])
    lats = np.array([grid.latlon_of(r, c).lat for r, c in coords.T])
    ax.plot(lons, lats, color=colour, lw=ROUTE_WIDTH, label=label)
    ax.plot(np.where(blind >= BLIND_THRESHOLD, lons, np.nan), np.where(blind >= BLIND_THRESHOLD, lats, np.nan),
            color=BLIND_COLOUR, lw=BLIND_WIDTH, solid_capstyle="butt")
    corners = route.waypoints
    ax.scatter([p.lon for p in corners], [p.lat for p in corners], s=MARKER_SIZE / 4, color=colour, zorder=3)


def plot_routes(ax: plt.Axes, grid: CostGrid, routes: list[Route]) -> None:
    for i, route in enumerate(routes):
        flag = " ⚠ blind limit" if route.exceeds_blind_limit else ""
        label = (f"w={route.weight:g}: {route.length_m / 1000:.2f} km, blind {route.blind_share:.0%}, "
                 f"longest {route.longest_blind_m:.0f} m{flag}")
        plot_route(ax, grid, route, ROUTE_COLOURS[i % len(ROUTE_COLOURS)], label)
    start, end = routes[0].waypoints[0], routes[0].waypoints[-1]
    ax.scatter([start.lon], [start.lat], s=MARKER_SIZE, marker="o", color="white", edgecolor="black", zorder=4)
    ax.scatter([end.lon], [end.lat], s=MARKER_SIZE, marker="*", color="white", edgecolor="black", zorder=4)
