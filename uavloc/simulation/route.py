"""Flight routes: waypoints on a raster, sampled into timed poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, bearing_deg, east_north_m, haversine_m, offset_latlon

MIN_LEG_M = 1e-6


@dataclass(frozen=True)
class Pose:
    t_s: float
    position: LatLon
    yaw_deg: float       # compass heading: 0 = north, clockwise
    altitude_m: float


def waypoints_from_fractions(raster: GeoImage, fractions: Sequence[tuple[float, float]]) -> list[LatLon]:
    """`(x, y)` as shares of the raster's width and height, measured from its top-left corner."""
    h, w = raster.pixels.shape[:2]
    return [raster.to_latlon(x * w, y * h) for x, y in fractions]


def random_waypoints(raster: GeoImage, n_waypoints: int, margin_m: float, rng: np.random.Generator) -> list[LatLon]:
    """Uniform waypoints at least `margin_m` inside the raster's edges."""
    width_m, height_m = raster.size_m
    margin_x, margin_y = margin_m / width_m, margin_m / height_m
    return waypoints_from_fractions(raster, [(rng.uniform(margin_x, 1 - margin_x), rng.uniform(margin_y, 1 - margin_y))
                                             for _ in range(n_waypoints)])


def sample_route(waypoints: Sequence[LatLon], speed_mps: float, step_s: float, altitude_m: float) -> list[Pose]:
    """Constant-speed poses every `step_s` along the polyline; yaw follows the current leg."""
    origin = waypoints[0]
    points = np.array([east_north_m(origin, p) for p in waypoints])
    legs = np.diff(points, axis=0)
    keep = np.hypot(*legs.T) > MIN_LEG_M
    starts, legs = points[:-1][keep], legs[keep]
    lengths = np.hypot(*legs.T)
    leg_start_m = np.concatenate([[0.0], np.cumsum(lengths)])
    poses = []
    for k, along_m in enumerate(np.arange(0.0, leg_start_m[-1], speed_mps * step_s)):
        leg = int(np.searchsorted(leg_start_m, along_m, side="right")) - 1
        east_m, north_m = starts[leg] + legs[leg] * (along_m - leg_start_m[leg]) / lengths[leg]
        poses.append(Pose(k * step_s, offset_latlon(origin, east_m, north_m), bearing_deg(*legs[leg]), altitude_m))
    return poses


def route_length_m(poses: Sequence[Pose]) -> float:
    return float(sum(haversine_m(a.position, b.position) for a, b in zip(poses, poses[1:])))


def check_inside(poses: Sequence[Pose], raster: GeoImage, margin_m: float, name: str = "raster") -> None:
    """Raise if any pose comes closer than `margin_m` to the raster's edge (search patches would leave it)."""
    h, w = raster.pixels.shape[:2]
    margin_x, margin_y = margin_m / raster.mpp_xy[0], margin_m / raster.mpp_xy[1]
    for pose in poses:
        x, y = raster.to_px(pose.position)
        if not (margin_x <= x <= w - margin_x and margin_y <= y <= h - margin_y):
            raise ValueError(f"pose at t={pose.t_s:.1f} s is within {margin_m} m of the {name} edge; "
                             "move the waypoints inward")
