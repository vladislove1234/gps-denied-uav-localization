"""Bounding box <-> pixel conversion for a north-up georeferenced image.

The archive carries three separate implementations of this (bug-008) and they
disagree. This is a local PoC copy, deliberately: nothing earns a place in
``uavloc/`` until the PoC returns a verdict. If it graduates, this is the file
that settles which convention wins.

Convention here matches the 2025 pipeline: a ``Bounds`` is the lat/lon of the
top-left and bottom-right corners, the image is north-up with no rotation, and
the projection is treated as locally linear -- fine over a few hundred metres,
wrong over a county.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

M_PER_DEG_LAT = 110_574.0


def m_per_deg_lon(lat: float) -> float:
    """Metres per degree of longitude at this latitude."""
    return 111_320.0 * math.cos(math.radians(lat))


@dataclass(frozen=True)
class Bounds:
    """Geographic extent of a north-up image."""

    top: float      # latitude of the top edge
    left: float     # longitude of the left edge
    bottom: float   # latitude of the bottom edge
    right: float    # longitude of the right edge

    @property
    def centre_lat(self) -> float:
        return (self.top + self.bottom) / 2.0

    def width_m(self) -> float:
        return (self.right - self.left) * m_per_deg_lon(self.centre_lat)

    def height_m(self) -> float:
        return (self.top - self.bottom) * M_PER_DEG_LAT

    def mpp(self, shape: tuple[int, int]) -> tuple[float, float]:
        """Metres per pixel as ``(x, y)`` for an image of ``(height, width)``."""
        h, w = shape[:2]
        return self.width_m() / w, self.height_m() / h

    def to_pixel(self, lat: float, lon: float, shape: tuple[int, int]) -> tuple[float, float]:
        """Geographic position -> ``(x, y)`` in pixels. Not clamped to the image."""
        h, w = shape[:2]
        x = (lon - self.left) / (self.right - self.left) * w
        y = (self.top - lat) / (self.top - self.bottom) * h
        return x, y

    def to_latlon(self, x: float, y: float, shape: tuple[int, int]) -> tuple[float, float]:
        """``(x, y)`` in pixels -> ``(lat, lon)``."""
        h, w = shape[:2]
        lon = self.left + (x / w) * (self.right - self.left)
        lat = self.top - (y / h) * (self.top - self.bottom)
        return lat, lon

    def as_overpass(self) -> str:
        """``south,west,north,east`` -- the order Overpass expects."""
        return f"{self.bottom},{self.left},{self.top},{self.right}"


# The 2025 Gonchary orthophoto, corners as recorded in the archive's
# gps_to_satellite_point.ipynb.
GONCHARY = Bounds(top=49.762965, left=24.166472, bottom=49.759291, right=24.174046)


class Affine2D:
    """Geographic -> pixel transform fitted from control points.

    ``Bounds`` assumes the image is a perfect north-up box. A real Google Earth
    export is close to that but not exactly: the Davydiv pair came out with
    +0.486 deg of rotation, which over a 909 m frame displaces the edges by
    about 4 m -- small, but the same order as the effect we are trying to
    measure, so it should not be thrown away.

    Duck-types ``Bounds``: it offers ``to_pixel`` and ``mpp``, so the rasteriser
    takes either.
    """

    def __init__(self, matrix: np.ndarray, centre_lat: float):
        self.matrix = matrix          # 3x2, maps [lon, lat, 1] -> [x, y]
        self.centre_lat = centre_lat

    @classmethod
    def fit(cls, control: list[tuple[float, float, float, float]]) -> "Affine2D":
        """Least-squares fit from ``(lat, lon, x, y)`` control points."""
        source = np.array([[lon, lat, 1.0] for lat, lon, _, _ in control])
        target = np.array([[x, y] for _, _, x, y in control])
        matrix, *_ = np.linalg.lstsq(source, target, rcond=None)
        return cls(matrix, float(np.mean([lat for lat, _, _, _ in control])))

    def to_pixel(self, lat: float, lon: float, shape=None) -> tuple[float, float]:
        x, y = np.array([lon, lat, 1.0]) @ self.matrix
        return float(x), float(y)

    def to_latlon(self, x: float, y: float, shape=None) -> tuple[float, float]:
        a = self.matrix[:2, :]
        offset = self.matrix[2, :]
        lon, lat = np.linalg.solve(a.T, np.array([x, y]) - offset)
        return float(lat), float(lon)

    def mpp(self, shape=None) -> tuple[float, float]:
        """Metres per pixel along each image axis."""
        jacobian = np.array([
            self.matrix[0, :] / m_per_deg_lon(self.centre_lat),   # d(pixel)/d(east metre)
            self.matrix[1, :] / M_PER_DEG_LAT,                    # d(pixel)/d(north metre)
        ])
        scales = np.linalg.svd(jacobian, compute_uv=False)
        return float(1 / scales[0]), float(1 / scales[1])

    def residuals(self, control) -> np.ndarray:
        return np.array([
            np.hypot(*(np.array(self.to_pixel(lat, lon)) - np.array([x, y])))
            for lat, lon, x, y in control
        ])

    def image_bounds(self, shape: tuple[int, int], margin_deg: float = 0.001) -> Bounds:
        """Geographic box covering the whole image, padded -- for the OSM query."""
        height, width = shape[:2]
        corners = [self.to_latlon(x, y)
                   for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        return Bounds(top=max(lats) + margin_deg, left=min(lons) - margin_deg,
                      bottom=min(lats) - margin_deg, right=max(lons) + margin_deg)
