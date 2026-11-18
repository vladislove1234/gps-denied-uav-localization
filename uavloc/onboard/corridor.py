"""The strip of map around a route that a mission package carries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, east_north_m, east_north_m_array, footprint_m, haversine_m, offset_latlon
from uavloc.matching.correlator import half_range_for_footprint_m
from uavloc.planning.config import MissionInputs, PlannerConfig
from uavloc.planning.drift import drift_sigma_m

MASK_CELL_M = 4.0   # the distance to the route is evaluated on this grid, so the edge is placed within about this


@dataclass(frozen=True)
class CorridorWidth:
    """Half-width of the carried strip, as the sum of three things the matcher needs map for."""

    drift_m: float                  # `drift_sigmas` x INS drift after the longest allowed blind stretch
    footprint_half_diagonal_m: float
    search_half_range_m: float      # the matcher's larger half-range, the reach of its search window

    @property
    def half_width_m(self) -> float:
        return self.drift_m + self.footprint_half_diagonal_m + self.search_half_range_m


def corridor_width(blind_limit_m: float, inputs: MissionInputs, config: PlannerConfig) -> CorridorWidth:
    """Drift at the blind limit puts the search window off the route by up to `drift_m`; the window and the
    footprint then reach out from there, so the map must hold all three."""
    drift_m = config.drift_sigmas * drift_sigma_m(blind_limit_m, inputs.speed_mps, config.ins)
    footprint = footprint_m(inputs.altitude_m, config.matcher.fov_w_deg, config.matcher.aspect_w_over_h)
    return CorridorWidth(drift_m, math.hypot(*footprint) / 2, max(half_range_for_footprint_m(config.matcher, footprint)))


def polyline_length_m(points: tuple[LatLon, ...]) -> float:
    return sum(haversine_m(a, b) for a, b in zip(points, points[1:]))


def point_at_distance(points: tuple[LatLon, ...], distance_m: float) -> LatLon:
    """The point `distance_m` along the polyline, clamped to its ends."""
    remaining = max(distance_m, 0.0)
    for a, b in zip(points, points[1:]):
        leg_m = haversine_m(a, b)
        if remaining <= leg_m and leg_m > 0:
            east_m, north_m = east_north_m(a, b)
            return offset_latlon(a, east_m * remaining / leg_m, north_m * remaining / leg_m)
        remaining -= leg_m
    return points[-1]


def distance_to_polyline_m(origin: LatLon, points: tuple[LatLon, ...], lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Distance of each `(lat, lon)` to the polyline, in the tangent plane at `origin`."""
    east, north = east_north_m_array(origin, lats, lons)
    nodes = np.array([east_north_m(origin, p) for p in points])
    nearest = np.full(east.shape, np.inf)
    for (e0, n0), (e1, n1) in zip(nodes, nodes[1:]):
        seg_e, seg_n = e1 - e0, n1 - n0
        seg_sq = max(seg_e ** 2 + seg_n ** 2, np.finfo(float).tiny)
        along = np.clip(((east - e0) * seg_e + (north - n0) * seg_n) / seg_sq, 0.0, 1.0)
        nearest = np.minimum(nearest, np.hypot(east - (e0 + along * seg_e), north - (n0 + along * seg_n)))
    return nearest


def corridor_mask(image: GeoImage, waypoints: tuple[LatLon, ...], half_width_m: float) -> np.ndarray:
    """`(H, W)` bool, True within `half_width_m` of the route. Computed on a coarse grid, then repeated per pixel."""
    height, width = image.pixels.shape[:2]
    width_m, height_m = image.size_m
    coarse_w, coarse_h = max(math.ceil(width_m / MASK_CELL_M), 1), max(math.ceil(height_m / MASK_CELL_M), 1)
    cols, rows = np.meshgrid(np.arange(coarse_w) + 0.5, np.arange(coarse_h) + 0.5)
    lats = image.top_left.lat + (image.bottom_right.lat - image.top_left.lat) * rows / coarse_h
    lons = image.top_left.lon + (image.bottom_right.lon - image.top_left.lon) * cols / coarse_w
    inside = distance_to_polyline_m(waypoints[0], waypoints, lats, lons) <= half_width_m
    return cv2.resize(inside.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
