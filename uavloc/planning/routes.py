"""Route alternatives: A* at several weights, straightened into legs, checked, near-duplicates dropped."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from uavloc.geo import LatLon
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.grid import Cell, CostGrid, astar

BLIND_THRESHOLD = 0.5            # a sample counts as blind when its interpolated blind share reaches this
SAMPLES_PER_CELL = 2             # profile sampling density along a leg
ARDUCOPTER_CLIMB_MPS = 2.5       # WPNAV_SPEED_UP default
ARDUCOPTER_DESCENT_MPS = 1.5     # WPNAV_SPEED_DN default
DUPLICATE_SAMPLE_M = 10.0

Point = tuple[float, float]      # fractional (row, col); (r + 0.5, c + 0.5) is a cell centre


@dataclass(frozen=True)
class BlindStretch:
    start_m: float               # along the one-way route
    end_m: float

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass(frozen=True)
class Route:
    weight: float
    waypoints: tuple[LatLon, ...]        # start, legs' corners, end
    length_m: float                      # one way
    blind_share: float
    blind_stretches: tuple[BlindStretch, ...]
    flight_time_s: float                 # out and back, with climb, hover and descent
    blind_limit_m: float
    flight_time_limit_s: float

    @property
    def longest_blind_m(self) -> float:
        return max((s.length_m for s in self.blind_stretches), default=0.0)

    @property
    def exceeds_blind_limit(self) -> bool:
        return self.longest_blind_m > self.blind_limit_m

    @property
    def exceeds_flight_time(self) -> bool:
        return self.flight_time_s > self.flight_time_limit_s


def cell_centre(cell: Cell) -> Point:
    return cell[0] + 0.5, cell[1] + 0.5


def sample_polyline(grid: CostGrid, points: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """`(distance_m, (2, N) row/col)` samples along `points`, about `SAMPLES_PER_CELL` per cell."""
    distances, coords, travelled = [0.0], [points[0]], 0.0
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        n = max(int(np.ceil(max(abs(r1 - r0), abs(c1 - c0)) * SAMPLES_PER_CELL)), 1)
        leg_m = grid.step_m(r1 - r0, c1 - c0)
        for k in range(1, n + 1):
            coords.append((r0 + (r1 - r0) * k / n, c0 + (c1 - c0) * k / n))
            distances.append(travelled + leg_m * k / n)
        travelled += leg_m
    return np.asarray(distances), np.asarray(coords).T


def blind_profile(grid: CostGrid, points: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """`(distance_m, blind share)` along `points`, bilinear on cell centres."""
    distances, coords = sample_polyline(grid, points)
    return distances, map_coordinates(grid.blind, coords - 0.5, order=1, mode="nearest")


def blind_stretches(distances_m: np.ndarray, blind: np.ndarray) -> tuple[BlindStretch, ...]:
    flags = blind >= BLIND_THRESHOLD
    stretches, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = distances_m[max(i - 1, 0)]   # the stretch begins after the last good sample
        if not flag and start is not None:
            stretches.append(BlindStretch(float(start), float(distances_m[i])))
            start = None
    if start is not None:
        stretches.append(BlindStretch(float(start), float(distances_m[-1])))
    return tuple(stretches)


def longest_blind_m(grid: CostGrid, points: list[Point]) -> float:
    return max((s.length_m for s in blind_stretches(*blind_profile(grid, points))), default=0.0)


def polyline_cost(grid: CostGrid, points: list[Point], weight: float) -> float:
    """The A* cost of flying straight along `points`; inf if it crosses an impassable cell."""
    distances, coords = sample_polyline(grid, points)
    cells = np.clip(np.floor(coords).astype(int), 0, np.array(grid.shape)[:, None] - 1)
    if grid.blocked[cells[0], cells[1]].any():
        return float("inf")
    blind = map_coordinates(grid.blind, coords - 0.5, order=1, mode="nearest")
    steps = np.diff(distances)
    return float((steps * (1 + weight * (blind[1:] + blind[:-1]) / 2)).sum())


def straighten(grid: CostGrid, cells: list[Cell], weight: float, tolerance: float) -> list[Point]:
    """Fewest legs whose shortcuts cost at most `tolerance` more than the cell path they replace and whose
    longest blind stretch is no longer than that path's (plus one cell). Greedy, farthest reach by bisection."""
    points = [cell_centre(c) for c in cells]
    cell_m = max(grid.raster.mpp_xy)

    def acceptable(i: int, j: int) -> bool:
        shortcut = [points[i], points[j]]
        budget = (1 + tolerance) * polyline_cost(grid, points[i:j + 1], weight)
        return (polyline_cost(grid, shortcut, weight) <= budget
                and longest_blind_m(grid, shortcut) <= longest_blind_m(grid, points[i:j + 1]) + cell_m)

    kept, i = [points[0]], 0
    while i < len(points) - 1:
        low, high = i + 1, len(points) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if acceptable(i, mid):
                low = mid
            else:
                high = mid - 1
        kept.append(points[low])
        i = low
    return kept


def out_and_back_time_s(length_m: float, inputs: MissionInputs) -> float:
    return (inputs.altitude_m / ARDUCOPTER_CLIMB_MPS + 2 * length_m / inputs.speed_mps + inputs.hover_s
            + inputs.altitude_m / ARDUCOPTER_DESCENT_MPS)


def mean_separation_m(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest distance between two `(N, 2)` metre point sets."""
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)


def route_points_m(grid: CostGrid, points: list[Point]) -> np.ndarray:
    _, coords = sample_polyline(grid, points)
    mpp_x, mpp_y = grid.raster.mpp_xy
    stride = max(int(DUPLICATE_SAMPLE_M / min(mpp_x, mpp_y) * SAMPLES_PER_CELL), 1)
    return np.stack([coords[0, ::stride] * mpp_y, coords[1, ::stride] * mpp_x], axis=1)


def plan_routes(grid: CostGrid, inputs: MissionInputs, config: PlannerConfig, blind_limit_m: float) -> list[Route]:
    """One route per weight in `config.route_weights`, shortest first, near-duplicates dropped."""
    start, goal = grid.cell_of(inputs.start), grid.cell_of(inputs.end)
    time_limit_s = inputs.max_flight_time_s * (1 - config.flight_reserve)
    routes, kept_points = [], []
    for weight in config.route_weights:
        points = straighten(grid, astar(grid, start, goal, weight), weight, config.straighten_tolerance)
        sampled = route_points_m(grid, points)
        if any(mean_separation_m(sampled, other) < config.duplicate_distance_m for other in kept_points):
            continue
        kept_points.append(sampled)
        distances, blind = blind_profile(grid, points)
        # legs run between cell centres; the ends are pinned to the exact clicked points
        waypoints = (inputs.start, *(grid.latlon_of(*p) for p in points[1:-1]), inputs.end)
        length_m = float(distances[-1])
        routes.append(Route(weight, waypoints, length_m, float((blind >= BLIND_THRESHOLD).mean()),
                            blind_stretches(distances, blind), out_and_back_time_s(length_m, inputs),
                            blind_limit_m, time_limit_s))
    return routes
