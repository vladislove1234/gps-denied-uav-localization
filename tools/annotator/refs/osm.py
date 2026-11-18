"""OpenStreetMap via Overpass. Ported from ``pocs/map-derived-labels/osm.py``.

Geometry comes out as vectors wherever it can stay vectors:

* building, field (``landuse=farmland``) -> polygons
* track -> centreline polylines with an assumed ``width_m`` (editable)
* woody, water -> polygons / buffered lines, burnt to masks by ``normalise``

Label offset. The dataset build measured, per location, the shift that best
lays OSM building outlines on image edges (``meta.label_offset``). When that
measurement is trustworthy (``peak_z >= 4`` and not at the search limit) it is
applied here, moving every OSM vertex by ``(dx_px, dy_px)`` -- the same
direction ``pocs/ms-building-footprints`` rolls ``label.png`` by. Otherwise the
geometry is left as mapped, and the layer-translate tool is the fix.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request

from .. import config
from .base import Adapter, Location, RefResult, register
from .vectorize import polygon_feature, polyline_feature

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Assumed full widths (m) for centrelines. Guesses, same as the PoC.
ROAD_WIDTH_M = {
    "primary": 8.0, "secondary": 7.0, "tertiary": 6.0, "residential": 6.0,
    "service": 4.0, "unclassified": 5.0, "track": 3.0, "path": 1.5,
    "footway": 1.5, "bridleway": 1.5,
}
WATERWAY_WIDTH_M = {"stream": 2.5, "ditch": 2.0, "drain": 2.0, "river": 12.0}
TREE_ROW_WIDTH_M = 6.0

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


def fetch(bbox: str, refresh: bool = False) -> list[dict]:
    """All tagged ways in ``bbox`` (south,west,north,east); cached like the PoC."""
    cache = config.OSM_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{hashlib.sha1(bbox.encode()).hexdigest()[:12]}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    query = QUERY.format(b=bbox)
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                ENDPOINTS[attempt % len(ENDPOINTS)],
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": "uavloc-research"})
            with urllib.request.urlopen(req, timeout=75) as r:
                elements = json.load(r)["elements"]
            path.write_text(json.dumps(elements))
            return elements
        except Exception as exc:          # noqa: BLE001 -- try the next mirror
            last = exc
            time.sleep(3)
    raise RuntimeError(f"all Overpass endpoints failed; last error: {last}")


def classify(tags: dict) -> str | None:
    if "building" in tags:
        return "building"
    if "highway" in tags:
        return "track"
    if tags.get("waterway") in WATERWAY_WIDTH_M or tags.get("natural") == "water":
        return "water"
    if (tags.get("natural") in ("wood", "scrub", "tree_row") or tags.get("landuse") == "forest"
            or tags.get("barrier") == "hedge"):
        return "woody"
    if tags.get("landuse") == "farmland":
        return "field"
    return None


def _closed(geom: list[dict]) -> bool:
    return len(geom) > 3 and geom[0]["lat"] == geom[-1]["lat"] and geom[0]["lon"] == geom[-1]["lon"]


def line_width(name: str, tags: dict) -> float:
    if name == "track":
        return ROAD_WIDTH_M.get(tags.get("highway"), 4.0)
    if name == "water":
        return WATERWAY_WIDTH_M.get(tags.get("waterway"), 2.0)
    return TREE_ROW_WIDTH_M


@register
class OSM(Adapter):
    name = "osm"
    version = "1"
    provides = ("building", "track", "woody", "water", "field")
    params = {"road_width_m": ROAD_WIDTH_M, "waterway_width_m": WATERWAY_WIDTH_M,
              "tree_row_width_m": TREE_ROW_WIDTH_M, "offset_min_peak_z": 4.0}

    def run(self, loc: Location) -> RefResult:
        b = loc.grid.bounds()
        m = 0.001        # same margin as the dataset build, so the cache hits
        bbox = f"{b['bottom'] - m},{b['left'] - m},{b['top'] + m},{b['right'] + m}"
        elements = fetch(bbox)

        off = loc.meta.get("label_offset") or {}
        trusted = (off.get("peak_z", 0) >= self.params["offset_min_peak_z"]
                   and not off.get("at_search_limit", True))
        dx, dy = (float(off.get("dx_px", 0)), float(off.get("dy_px", 0))) if trusted else (0.0, 0.0)

        def ll(geom):
            out = []
            for p in geom:
                x, y = loc.grid.to_px(p["lat"], p["lon"])
                lat, lon = loc.grid.to_latlon(x + dx, y + dy)
                out.append([lon, lat])
            return out

        feats, counts = [], {}
        for e in elements:
            geom, tags = e.get("geometry"), e.get("tags", {})
            if not geom or len(geom) < 2:
                continue
            name = classify(tags)
            if name is None:
                continue
            props = {"osm_id": e.get("id")}
            if _closed(geom) and not (name == "track" and tags.get("area") != "yes"):
                feats.append(polygon_feature(ll(geom), name, props))
            elif name in ("building", "field"):
                continue            # an unclosed building/farmland way is a mapping error
            else:
                feats.append(polyline_feature(ll(geom), name, line_width(name, tags), props))
            counts[name] = counts.get(name, 0) + 1

        return RefResult(vectors=feats, info={
            "ways": len(elements), "drawn": counts,
            "offset_applied_px": [dx, dy], "offset_trusted": trusted,
            "offset_measured": off})
