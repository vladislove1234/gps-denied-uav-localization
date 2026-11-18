"""Where to build the dataset: finding well-mapped rural ground in Ukraine.

The 2025 flight area is useless for map-derived labels -- 0 of 4,181 positions
have any mapped feature in frame, nearest building 129 m away. The fix is to
stop tying the training set to the flight area. The encoder has to learn what
makes structure match across domains; that is not a property of a particular
village, so the training ground can be anywhere well mapped.

Villages are the unit because they are where OSM density lives: a Ukrainian
village carries buildings, lanes, treelines and a strip-parcel field edge within
the same few hundred metres, which is exactly the mix a drone frame sees.

Deliberately avoids the east and south -- not a data-quality judgement, a
practical one about where imagery and mapping stay current.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import osm

# Rural west and centre: Lviv, Ternopil, Ivano-Frankivsk, Khmelnytskyi, Vinnytsia,
# Zhytomyr, Rivne, Volyn. Deliberately excludes the east and the far south.
REGION = (48.30, 23.20, 50.80, 29.00)      # south, west, north, east

# The country filter is not optional: a lon 23.2 west edge reaches well into
# Poland, and the first version happily returned Jozefowka and Zdanowek.
#
# Doing that filter *inside* Overpass (`area["ISO3166-1"="UA"]`) 504s on every
# endpoint -- resolving a country-sized area and intersecting it with a large
# bbox is too much for one query. Fetching the boundary once and testing points
# locally takes 1 s and never times out.
VILLAGE_QUERY = """[out:json][timeout:180];
node["place"~"^(village|hamlet)$"]({b});
out body;
"""

BOUNDARY_QUERY = """[out:json][timeout:280];
relation["ISO3166-1"="UA"][admin_level=2];
out geom;
"""


def _overpass(query: str, timeout: int = 300):
    last = None
    for endpoint in osm.ENDPOINTS:
        try:
            request = urllib.request.Request(
                endpoint, data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": "uav-localization-research/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"all Overpass endpoints failed: {last}")


def boundary_segments(cache_dir: Path = Path("data/derived/osm-cache")):
    """Ukraine's national outline as a flat array of edges, ``(N, 4)`` lon/lat.

    The relation comes back as 1,986 unordered outer ways. They are *not*
    stitched into rings here and do not need to be: ray casting only needs the
    set of edges, and the edges collectively close.
    """
    import numpy as np
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "ua-boundary-segments.npy"
    if path.exists():
        return np.load(path)
    payload = _overpass(BOUNDARY_QUERY)
    edges = []
    for element in payload["elements"]:
        for member in element.get("members", []):
            if member.get("role") != "outer" or "geometry" not in member:
                continue
            g = member["geometry"]
            for a, b in zip(g[:-1], g[1:]):
                edges.append((a["lon"], a["lat"], b["lon"], b["lat"]))
    out = np.asarray(edges, dtype=np.float64)
    np.save(path, out)
    return out


def inside_ukraine(lats, lons, cache_dir: Path = Path("data/derived/osm-cache")):
    """Vectorised ray-cast point-in-polygon against the national outline."""
    import numpy as np
    seg = boundary_segments(cache_dir)
    x1, y1, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    inside = np.zeros(lats.shape, dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        straddles = (y1 > lat) != (y2 > lat)
        if not straddles.any():
            continue
        a, b, c, d = x1[straddles], y1[straddles], x2[straddles], y2[straddles]
        crossing_x = a + (lat - b) / (d - b) * (c - a)
        inside[i] = bool((crossing_x > lon).sum() % 2)
    return inside


def villages(region=REGION, cache_dir: Path = Path("data/derived/osm-cache"),
             refresh: bool = False) -> list[dict]:
    """Every mapped village and hamlet node in ``region``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    box = "{},{},{},{}".format(*region)
    path = cache_dir / f"villages-ua-{box.replace(',', '_')}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    payload = _overpass(VILLAGE_QUERY.replace("{b}", box))
    found = [{"name": e.get("tags", {}).get("name", ""),
              "lat": e["lat"], "lon": e["lon"]}
             for e in payload.get("elements", [])]
    keep = inside_ukraine([p["lat"] for p in found], [p["lon"] for p in found], cache_dir)
    out = [p for p, ok in zip(found, keep) if ok]
    path.write_text(json.dumps(out))
    return out


def spread(candidates: list[dict], count: int, min_separation_km: float = 12.0) -> list[dict]:
    """Greedily pick ``count`` places no closer than ``min_separation_km``.

    Geographic spread matters more than count. Twenty villages in one raion see
    one landscape, one soil colour and one field pattern -- the "one site, one
    season" failure of 2025 rebuilt at a larger scale.
    """
    import math
    import random
    chosen: list[dict] = []
    # Overpass returns nodes in roughly spatial order, so walking the list in
    # order packs the picks into one corner. Shuffle with a fixed seed: spread
    # by construction, and still reproducible.
    order = list(candidates)
    random.Random(20260913).shuffle(order)
    for place in order:
        if len(chosen) >= count:
            break
        ok = True
        for other in chosen:
            dlat = (place["lat"] - other["lat"]) * 110.574
            dlon = (place["lon"] - other["lon"]) * 111.320 * math.cos(math.radians(place["lat"]))
            if math.hypot(dlat, dlon) < min_separation_km:
                ok = False
                break
        if ok:
            chosen.append(place)
    return chosen
