"""The routing grid: blind share and impassable cells on ~10 m cells, with A* over it."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.features.classes import CLASSES
from uavloc.geo import LatLon

Cell = tuple[int, int]   # (row, col)
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True)
class CostGrid:
    raster: GeoImage           # `(rows, cols, 1)` float32 blind share per cell, 0..1; carries the georeference
    blocked: np.ndarray        # `(rows, cols)` bool, impassable

    @property
    def blind(self) -> np.ndarray:
        return self.raster.pixels[..., 0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.blind.shape

    def cell_of(self, point: LatLon) -> Cell:
        x, y = self.raster.to_px(point)
        rows, cols = self.shape
        if not (0 <= x < cols and 0 <= y < rows):
            raise ValueError(f"{point} is outside the planning area")
        return int(y), int(x)

    def latlon_of(self, row: float, col: float) -> LatLon:
        """Lat/lon of a fractional cell position; `(r + 0.5, c + 0.5)` is the centre of cell `(r, c)`."""
        return self.raster.to_latlon(col, row)

    def step_m(self, d_row: float, d_col: float) -> float:
        mpp_x, mpp_y = self.raster.mpp_xy
        return math.hypot(d_row * mpp_y, d_col * mpp_x)


def cost_grid(blind_map: GeoImage, class_map: GeoImage, cell_m: float, blocked_classes: tuple[str, ...],
              blocked_share: float) -> CostGrid:
    """Blind share bilinearly resampled to `cell_m` cells; cells where blocked classes dominate are impassable."""
    width_m, height_m = blind_map.size_m
    size = (max(round(width_m / cell_m), 1), max(round(height_m / cell_m), 1))
    blind = cv2.resize(blind_map.pixels[..., 0], size, interpolation=cv2.INTER_LINEAR)
    blocked = np.zeros((size[1], size[0]), bool)
    if blocked_classes:
        ids = [CLASSES.index(name) for name in blocked_classes]
        argmax = class_map.pixels.argmax(-1)
        share = cv2.resize(np.isin(argmax, ids).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        blocked = share >= blocked_share
    return CostGrid(blind_map.with_pixels(np.clip(blind, 0, 1).astype(np.float32)[..., None]), blocked)


def astar(grid: CostGrid, start: Cell, goal: Cell, weight: float) -> list[Cell]:
    """Cheapest 8-connected cell path. Step cost = length x (1 + weight x mean blind share of its two cells).

    The heuristic is the straight-line distance, admissible because no step costs less than its length.
    """
    if grid.blocked[start] or grid.blocked[goal]:
        raise ValueError("start or end lies in an impassable cell")
    rows, cols = grid.shape
    blind, blocked = grid.blind, grid.blocked
    step_lengths = [grid.step_m(dr, dc) for dr, dc in NEIGHBOURS]
    mpp_x, mpp_y = grid.raster.mpp_xy
    best = np.full((rows, cols), np.inf)
    parent = np.full((rows, cols), -1, np.int64)
    best[start] = 0.0
    frontier = [(0.0, 0.0, start)]
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell == goal:
            break
        if cost > best[cell]:
            continue
        r, c = cell
        for (dr, dc), length in zip(NEIGHBOURS, step_lengths):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                continue
            new_cost = cost + length * (1 + weight * (blind[r, c] + blind[nr, nc]) / 2)
            if new_cost < best[nr, nc]:
                best[nr, nc] = new_cost
                parent[nr, nc] = r * cols + c
                heuristic = math.hypot((goal[0] - nr) * mpp_y, (goal[1] - nc) * mpp_x)
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, (nr, nc)))
    if not np.isfinite(best[goal]):
        raise ValueError("no path between start and end")
    path = [goal]
    while path[-1] != start:
        index = parent[path[-1]]
        path.append((int(index // cols), int(index % cols)))
    return path[::-1]
