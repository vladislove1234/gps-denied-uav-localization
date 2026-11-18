"""OpenStreetMap features for a bounding box, rasterised onto a georeferenced image.

Why OSM rather than a pretrained segmenter: labels from a pretrained RGB model
inherit the domain gap we are trying to escape -- the same field can be labelled
differently in a drone frame and a stale satellite tile, because the model reads
pixels. OSM reads geometry, so the same ground gets the same label in both
domains by construction.

Two practical notes learned the hard way:

* The main Overpass instance 504s constantly. ``fetch`` rotates endpoints and
  caches to disk, because re-running the driver should not re-hit the API.
* OSM roads are *centrelines*, not polygons. They are buffered by an assumed
  width from ``ROAD_WIDTH_M``; those widths are guesses and are the largest
  source of label error in anything linear.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geo import Bounds, m_per_deg_lon, M_PER_DEG_LAT

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

CACHE = Path("data/derived/osm-cache")

# Class ids in the rasterised mask. 0 is background/unmapped -- which over rural
# Ukraine is most of the image, and that is a finding, not a bug.
CLASSES = {"background": 0, "building": 1, "road": 2, "water": 3, "wood": 4}

# Assumed full width in metres for buffering centrelines. Guesses.
ROAD_WIDTH_M = {
    "primary": 8.0, "secondary": 7.0, "tertiary": 6.0, "residential": 6.0,
    "service": 4.0, "unclassified": 5.0, "track": 3.0, "path": 1.5,
    "footway": 1.5, "bridleway": 1.5,
}
WATERWAY_WIDTH_M = {"stream": 2.5, "ditch": 2.0, "drain": 2.0, "river": 12.0}

QUERY = """[out:json][timeout:60];
(
  way["building"]({b});
  way["highway"]({b});
  way["waterway"]({b});
  way["natural"]({b});
  way["landuse"]({b});
);
out tags geom;
"""


def fetch(bounds: Bounds, cache_dir: Path = CACHE, refresh: bool = False) -> list[dict]:
    """All tagged ways intersecting ``bounds``, cached on disk by bbox."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(bounds.as_overpass().encode()).hexdigest()[:12]
    path = cache_dir / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    query = QUERY.format(b=bounds.as_overpass())
    last = None
    for attempt in range(6):
        endpoint = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            request = urllib.request.Request(
                endpoint,
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": "uavloc-research"},
            )
            with urllib.request.urlopen(request, timeout=75) as response:
                elements = json.load(response)["elements"]
            path.write_text(json.dumps(elements))
            return elements
        except Exception as exc:          # noqa: BLE001 -- any failure means try the next mirror
            last = exc
            time.sleep(3)
    raise RuntimeError(f"all Overpass endpoints failed; last error: {last}")


def classify(tags: dict) -> str | None:
    """Map OSM tags onto one of ``CLASSES``. ``None`` means "not a class we use"."""
    if "building" in tags:
        return "building"
    if "highway" in tags:
        return "road"
    if tags.get("waterway") in WATERWAY_WIDTH_M or tags.get("natural") == "water":
        return "water"
    if tags.get("natural") in ("wood", "scrub", "tree_row") or tags.get("landuse") == "forest":
        return "wood"
    return None


def _is_closed(geometry: list[dict]) -> bool:
    first, last = geometry[0], geometry[-1]
    return len(geometry) > 3 and first["lat"] == last["lat"] and first["lon"] == last["lon"]


def rasterise(
    elements: list[dict], bounds, shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, int]]:
    """Burn ``elements`` into a class-id mask of ``shape`` = ``(height, width)``.

    ``bounds`` is anything offering ``to_pixel`` and ``mpp`` -- a ``Bounds`` for a
    north-up box, or an ``Affine2D`` fitted from control points when the export
    carries a little rotation.

    Returns the mask and a per-class count of the features actually drawn.
    """
    height, width = shape[:2]
    canvas = Image.new("L", (width, height), CLASSES["background"])
    draw = ImageDraw.Draw(canvas)
    mpp_x, _ = bounds.mpp(shape)
    drawn: dict[str, int] = {}

    for element in elements:
        geometry = element.get("geometry")
        tags = element.get("tags", {})
        if not geometry:
            continue
        name = classify(tags)
        if name is None:
            continue
        value = CLASSES[name]
        points = [bounds.to_pixel(p["lat"], p["lon"], shape) for p in geometry]
        if len(points) < 2:
            continue

        if _is_closed(geometry):
            draw.polygon(points, fill=value)
        else:
            metres = (
                ROAD_WIDTH_M.get(tags.get("highway"), 4.0)
                if name == "road"
                else WATERWAY_WIDTH_M.get(tags.get("waterway"), 2.0)
            )
            draw.line(points, fill=value, width=max(1, round(metres / mpp_x)), joint="curve")
        drawn[name] = drawn.get(name, 0) + 1

    return np.array(canvas), drawn
