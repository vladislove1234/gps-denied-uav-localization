"""Geometry conversions between the two forms a layer can take.

* raster -> polygons, for vector classes fed by a model (building, field)
* polygons / polylines -> raster, for raster classes fed by a map (OSM wood)
  and for export

Coordinates are continuous image coordinates: pixel ``(i, j)`` covers
``[i, i+1] x [j, j+1]`` -- the convention of the browser canvas and of Web
Mercator global pixels. cv2 puts integer coordinates on pixel *centres*, so
contours get ``+0.5`` on the way out and rasterization ``-0.5`` on the way in.

Polygons are single outer rings: no holes. OSM ways and MS footprints are
already single rings, and a model blob with a hole is filled -- a courtyard is
rarely worth the editing complexity. Rings are GeoJSON order, ``[lon, lat]``.
"""

from __future__ import annotations

import cv2
import numpy as np

SHIFT = 4                       # cv2 sub-pixel bits: coordinates * 16
SCALE = 1 << SHIFT


def mask_to_rings(mask: np.ndarray, simplify_px: float, min_area_px: float) -> list[np.ndarray]:
    """Outer contours of a binary mask as ``(n, 2)`` float pixel arrays."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        s = cv2.approxPolyDP(c, simplify_px, True) if simplify_px > 0 else c
        if len(s) < 3:
            continue
        out.append(s[:, 0, :].astype(np.float64) + 0.5)
    return out


def mask_to_features(mask, grid, cls: str, mpp: float, simplify_m: float = 0.5,
                     min_area_m2: float = 8.0, props: dict | None = None) -> list[dict]:
    rings = mask_to_rings(mask, simplify_m / mpp, min_area_m2 / (mpp * mpp))
    feats = []
    for r in rings:
        ring = grid.ring_to_lonlat(r)
        ring.append(ring[0])
        feats.append(polygon_feature(ring, cls, props))
    return feats


def polygon_feature(ring_lonlat: list, cls: str, props: dict | None = None) -> dict:
    return {"type": "Feature",
            "properties": {"class": cls, "kind": "polygon", **(props or {})},
            "geometry": {"type": "Polygon", "coordinates": [ring_lonlat]}}


def polyline_feature(line_lonlat: list, cls: str, width_m: float, props: dict | None = None) -> dict:
    return {"type": "Feature",
            "properties": {"class": cls, "kind": "polyline", "width_m": float(width_m), **(props or {})},
            "geometry": {"type": "LineString", "coordinates": line_lonlat}}


def _px(grid, coords) -> np.ndarray:
    return np.round((np.array(grid.ring_to_px(coords), np.float64) - 0.5) * SCALE).astype(np.int32)


def burn(canvas: np.ndarray, features: list[dict], grid, mpp: float, value: int = 1) -> np.ndarray:
    """Rasterize features into ``canvas`` in place (and return it).

    Polygons are filled; polylines are stroked at ``width_m`` with round caps --
    the same shapes the browser draws, so what you see is what exports.
    """
    for f in features:
        g = f["geometry"]
        if g["type"] == "Polygon":
            pts = _px(grid, g["coordinates"][0])
            if len(pts) >= 3:
                cv2.fillPoly(canvas, [pts], value, lineType=cv2.LINE_8, shift=SHIFT)
        elif g["type"] == "LineString":
            pts = _px(grid, g["coordinates"])
            width = max(1, int(round(f["properties"].get("width_m", 3.0) / mpp)))
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, value, thickness=width,
                              lineType=cv2.LINE_8, shift=SHIFT)
    return canvas


def rasterize(features: list[dict], grid, shape: tuple[int, int], mpp: float) -> np.ndarray:
    return burn(np.zeros(shape, np.uint8), features, grid, mpp).astype(bool)
