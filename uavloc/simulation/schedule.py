"""When a simulated flight attempts a fix: on a clock, or where the route planner expects one to work."""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, haversine_m
from uavloc.simulation.flight import TIME_EPS_S
from uavloc.simulation.route import Pose

OFF_MAP_BLIND_SHARE = 1.0     # outside every planner map nothing is known, so no fix is expected


class FixSchedule(Protocol):
    def due(self, pose: Pose, estimate: LatLon) -> bool:
        """Called once per pose; True means attempt a fix at this pose."""
        ...


class EveryInterval:
    """A fix every `interval_s` of flight time, wherever the drone is (027-034)."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.next_s = interval_s

    def due(self, pose: Pose, estimate: LatLon) -> bool:
        if pose.t_s + TIME_EPS_S < self.next_s:
            return False
        self.next_s += self.interval_s
        return True


class WhereLocalizable:
    """A fix where the planner's blind map, read at the estimate, expects one, and only once the estimate has
    moved `min_spacing_m` from the previous attempt (the flight's start counts as one)."""

    def __init__(self, blind_maps: Sequence[GeoImage], blind_threshold: float, min_spacing_m: float) -> None:
        self.blind_maps = list(blind_maps)
        self.blind_threshold = blind_threshold
        self.min_spacing_m = min_spacing_m
        self.last: LatLon | None = None

    def blind_share(self, point: LatLon) -> float:
        """Bilinear on cell centres of the first map that contains `point`."""
        for blind_map in self.blind_maps:
            x, y = blind_map.to_px(point)
            rows, cols = blind_map.pixels.shape[:2]
            if 0 <= x < cols and 0 <= y < rows:
                return float(map_coordinates(blind_map.pixels[..., 0], np.array([[y - 0.5], [x - 0.5]]),
                                             order=1, mode="nearest")[0])
        return OFF_MAP_BLIND_SHARE

    def due(self, pose: Pose, estimate: LatLon) -> bool:
        if self.last is None:
            self.last = estimate
        if haversine_m(self.last, estimate) < self.min_spacing_m:
            return False
        share = self.blind_share(estimate)
        if math.isnan(share) or share >= self.blind_threshold:
            return False
        self.last = estimate
        return True
