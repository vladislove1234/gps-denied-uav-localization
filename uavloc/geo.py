"""Shared geodesy: lat/lon, metres and bearings. Every module converts through here."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

FULL_TURN_DEG = 360.0

# WGS84 semi-major axis. The 2025 code's 111,320 m/deg and 40,075 km equator
# both imply this sphere, so one radius keeps its planar maths and our
# haversine consistent to 5e-6.
EARTH_RADIUS_M = 6_378_137.0
METERS_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0


class LatLon(NamedTuple):
    lat: float
    lon: float


class GeoBounds(NamedTuple):
    """A lat/lon box, north-up."""

    north: float
    west: float
    south: float
    east: float

    @classmethod
    def around(cls, points: list[LatLon], margin_m: float) -> GeoBounds:
        """Smallest box holding `points`, grown by `margin_m` on every side."""
        lats, lons = [p.lat for p in points], [p.lon for p in points]
        mid_lat = (max(lats) + min(lats)) / 2
        dlat, dlon = margin_m / METERS_PER_DEG, margin_m / meters_per_deg_lon(mid_lat)
        return cls(max(lats) + dlat, min(lons) - dlon, min(lats) - dlat, max(lons) + dlon)

    @property
    def top_left(self) -> LatLon:
        return LatLon(self.north, self.west)

    @property
    def bottom_right(self) -> LatLon:
        return LatLon(self.south, self.east)

    @property
    def size_m(self) -> tuple[float, float]:
        """`(width_m, height_m)` at the middle latitude."""
        mid_lat = (self.north + self.south) / 2
        return (self.east - self.west) * meters_per_deg_lon(mid_lat), (self.north - self.south) * METERS_PER_DEG

    def contains(self, point: LatLon) -> bool:
        return self.south <= point.lat <= self.north and self.west <= point.lon <= self.east


def haversine_m(a: LatLon, b: LatLon) -> float:
    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def meters_per_deg_lon(lat: float) -> float:
    return METERS_PER_DEG * math.cos(math.radians(lat))


def offset_latlon(origin: LatLon, east_m: float, north_m: float) -> LatLon:
    """Local tangent-plane offset; exact to mm over the tens of metres used here."""
    return LatLon(origin.lat + north_m / METERS_PER_DEG,
                  origin.lon + east_m / meters_per_deg_lon(origin.lat))


def east_north_m(origin: LatLon, point: LatLon) -> tuple[float, float]:
    """Inverse of `offset_latlon`: `(east_m, north_m)` of `point` relative to `origin`."""
    return ((point.lon - origin.lon) * meters_per_deg_lon(origin.lat),
            (point.lat - origin.lat) * METERS_PER_DEG)


def east_north_m_array(origin: LatLon, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `east_north_m`."""
    return ((np.asarray(lons) - origin.lon) * meters_per_deg_lon(origin.lat),
            (np.asarray(lats) - origin.lat) * METERS_PER_DEG)


def bearing_deg(east_m: float, north_m: float) -> float:
    """Compass bearing of a ground vector: 0 = north, clockwise, in [0, 360)."""
    return math.degrees(math.atan2(east_m, north_m)) % FULL_TURN_DEG


def rotate_clockwise(east_m: float, north_m: float, angle_deg: float) -> tuple[float, float]:
    """A ground vector turned clockwise by `angle_deg`, the way a compass heading turns."""
    angle = math.radians(angle_deg)
    return (east_m * math.cos(angle) + north_m * math.sin(angle),
            -east_m * math.sin(angle) + north_m * math.cos(angle))


def footprint_m(altitude_m: float, fov_w_deg: float, aspect_w_over_h: float) -> tuple[float, float]:
    """Nadir camera ground footprint `(width_m, height_m)`."""
    width_m = 2 * altitude_m * math.tan(math.radians(fov_w_deg / 2))
    return width_m, width_m / aspect_w_over_h


def rotate_to_north_up(du: float, dv: float, yaw_deg: float) -> tuple[float, float]:
    """Heading-up image offset `(right, down)` → north-up image offset `(right, down)`.

    Same convention as the 2025 `PositionCorrelator.offset_pixels_to_gps`.
    """
    yaw = math.radians(yaw_deg)
    return (math.cos(yaw) * du - math.sin(yaw) * dv,
            math.sin(yaw) * du + math.cos(yaw) * dv)


def rotate_to_heading_up(dx: float, dy: float, yaw_deg: float) -> tuple[float, float]:
    """Inverse of `rotate_to_north_up`."""
    yaw = math.radians(yaw_deg)
    return (math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy)


def rotate_to_heading_up_array(dx: np.ndarray, dy: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """`rotate_to_heading_up` for arrays of offsets."""
    yaw = math.radians(yaw_deg)
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def rotate_to_north_up_array(du: np.ndarray, dv: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """`rotate_to_north_up` for arrays of offsets."""
    yaw = math.radians(yaw_deg)
    return math.cos(yaw) * du - math.sin(yaw) * dv, math.sin(yaw) * du + math.cos(yaw) * dv
