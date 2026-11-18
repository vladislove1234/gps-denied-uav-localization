"""Ground elevation along a route, checked against a mission flown at constant height above takeoff."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from uavloc.geo import LatLon, haversine_m

OPEN_METEO_URL = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_MAX_POINTS = 100
REQUEST_TIMEOUT_S = 30
PROFILE_STEP_M = 100.0           # the Copernicus DEM behind Open-Meteo is ~90 m
MIN_CLEARANCE_M = 20.0
MAX_FOOTPRINT_CHANGE = 0.15      # height above ground may differ from the planned altitude by this share


class ElevationSource(Protocol):
    def elevations_m(self, points: Sequence[LatLon]) -> np.ndarray: ...


class OpenMeteoElevation:
    """Open-Meteo's free elevation API (Copernicus DEM GLO-90)."""

    def elevations_m(self, points: Sequence[LatLon]) -> np.ndarray:
        out = []
        for i in range(0, len(points), OPEN_METEO_MAX_POINTS):
            chunk = points[i:i + OPEN_METEO_MAX_POINTS]
            query = urllib.parse.urlencode({"latitude": ",".join(f"{p.lat:.6f}" for p in chunk),
                                            "longitude": ",".join(f"{p.lon:.6f}" for p in chunk)})
            with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=REQUEST_TIMEOUT_S) as response:
                out.extend(json.load(response)["elevation"])
        return np.asarray(out, float)


@dataclass(frozen=True)
class ElevationCheck:
    distances_m: np.ndarray      # along the one-way route
    ground_m: np.ndarray         # relative to the takeoff point's ground
    altitude_m: float
    min_clearance_m: float
    max_footprint_change: float

    @property
    def height_above_ground_m(self) -> np.ndarray:
        return self.altitude_m - self.ground_m

    @property
    def warnings(self) -> tuple[str, ...]:
        agl = self.height_above_ground_m
        out = []
        if agl.min() < self.min_clearance_m:
            i = int(agl.argmin())
            out.append(f"clearance {agl[i]:.0f} m at {self.distances_m[i]:.0f} m along the route "
                       f"(minimum {self.min_clearance_m:.0f} m)")
        change = np.abs(agl / self.altitude_m - 1)
        if change.max() > self.max_footprint_change:
            i = int(change.argmax())
            out.append(f"height above ground {agl[i]:.0f} m at {self.distances_m[i]:.0f} m changes the camera "
                       f"footprint by {change[i]:.0%} (limit {self.max_footprint_change:.0%})")
        return tuple(out)


def profile_points(waypoints: Sequence[LatLon], step_m: float) -> tuple[np.ndarray, list[LatLon]]:
    """Points every ~`step_m` along the legs, with their distance along the route."""
    distances, points, travelled = [0.0], [waypoints[0]], 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        leg_m = haversine_m(a, b)
        n = max(int(np.ceil(leg_m / step_m)), 1)
        for k in range(1, n + 1):
            points.append(LatLon(a.lat + (b.lat - a.lat) * k / n, a.lon + (b.lon - a.lon) * k / n))
            distances.append(travelled + leg_m * k / n)
        travelled += leg_m
    return np.asarray(distances), points


def check_elevation(waypoints: Sequence[LatLon], altitude_m: float, source: ElevationSource,
                    step_m: float = PROFILE_STEP_M, min_clearance_m: float = MIN_CLEARANCE_M,
                    max_footprint_change: float = MAX_FOOTPRINT_CHANGE) -> ElevationCheck:
    distances, points = profile_points(waypoints, step_m)
    elevations = source.elevations_m(points)
    return ElevationCheck(distances, elevations - elevations[0], altitude_m, min_clearance_m, max_footprint_change)
