"""Georeferenced rasters named by a small spec, so a notebook switches imagery with one parameter.

    {"kind": "multidate", "place": "bushcha-50p2422-25p8305", "date": "current"}   # or "oldest" / "newest" / a date
    {"kind": "file", "path": "data/derived/<...>.jpg", "top_left": [lat, lon], "bottom_right": [lat, lon]}

`path` is relative to the repo root (`src/`).
"""

from __future__ import annotations

from typing import Any, Mapping

from uavloc.data.multidate import load_date
from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon
from uavloc.track import REPO_ROOT

RASTER_KINDS = ("multidate", "file")


def load_raster(spec: Mapping[str, Any]) -> GeoImage:
    kind = spec["kind"]
    if kind == "multidate":
        return load_date(spec["place"], spec["date"])
    if kind == "file":
        return GeoImage.load(REPO_ROOT / spec["path"], LatLon(*spec["top_left"]), LatLon(*spec["bottom_right"]))
    raise ValueError(f"raster kind must be one of {RASTER_KINDS}, got {kind!r}")
