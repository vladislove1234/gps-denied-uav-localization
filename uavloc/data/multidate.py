"""Dates of one `ua-village-multidate` location as georeferenced images."""

from __future__ import annotations

import json

from uavloc.data.drone_frames import DERIVED
from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon

MULTIDATE = "ua-village-multidate"
NON_WAYBACK_DATES = ("current", "google")   # `google` is co-registered to metres, not centimetres


def esri_dates(place: str) -> list[str]:
    """Esri Wayback dates on disk, oldest first, excluding `current` and `google`."""
    return sorted(p.stem for p in (DERIVED / MULTIDATE / place).glob("*.jpg") if p.stem not in NON_WAYBACK_DATES)


def load_date(place: str, date: str) -> GeoImage:
    """One date of `place`; `oldest` / `newest` pick from the Wayback dates."""
    if date in ("oldest", "newest"):
        dates = esri_dates(place)
        date = dates[0] if date == "oldest" else dates[-1]
    bounds = json.loads((DERIVED / MULTIDATE / place / "meta.json").read_text())["bounds"]
    return GeoImage.load(DERIVED / MULTIDATE / place / f"{date}.jpg", LatLon(bounds["top"], bounds["left"]),
                         LatLon(bounds["bottom"], bounds["right"]))
