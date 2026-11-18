"""Microsoft Global ML Building Footprints. Ported from
``pocs/ms-building-footprints/probe.ipynb``.

Footprints come per z9 Bing quadkey as gzipped GeoJSON-lines (the ``.csv.gz``
extension lies). A location can straddle two quadkeys, so all four corners are
looked up. Absence of footprints means *unknown*, never *no building*: the PoC
found a location (``ploske``) whose quadkey is empty over mapped houses.

Licence: CDLA Permissive 2.0.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import urllib.request

from .. import config
from .base import Adapter, Location, RefResult, register
from .vectorize import polygon_feature

LINKS_URL = "https://bfppub.blob.core.windows.net/$web/2026-08-13/dataset-links.csv"
COUNTRY = "Ukraine"


def get(url: str, dest=None) -> bytes:
    if dest is not None and dest.exists():
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "uav-localization-research/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    if dest is not None:
        dest.write_bytes(blob)
    return blob


def quadkey(lat: float, lon: float, z: int = 9) -> str:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 1 << z
    x = int((lon + 180.0) / 360.0 * n)
    s = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    x, y = min(x, n - 1), min(y, n - 1)
    key = ""
    for i in range(z, 0, -1):
        d, mask = 0, 1 << (i - 1)
        if x & mask:
            d += 1
        if y & mask:
            d += 2
        key += str(d)
    return key


@register
class MSFootprints(Adapter):
    name = "ms-footprints"
    version = "1"
    provides = ("building",)
    params = {"links_url": LINKS_URL, "country": COUNTRY}

    def __init__(self):
        self._rows = None
        self._tiles: dict[str, list[dict]] = {}

    def rows(self) -> dict[str, dict]:
        if self._rows is None:
            config.MS_CACHE.mkdir(parents=True, exist_ok=True)
            raw = get(LINKS_URL, config.MS_CACHE / "dataset-links.csv").decode("utf-8", "replace")
            header, *lines = raw.strip().split("\n")
            cols = [c.strip() for c in header.split(",")]
            recs = [dict(zip(cols, r.split(","))) for r in lines]
            self._rows = {r["QuadKey"]: r for r in recs if r.get("Location", "").strip() == COUNTRY}
        return self._rows

    def tile(self, key: str) -> list[dict]:
        if key not in self._tiles:
            row = self.rows().get(key)
            feats = []
            if row is not None:
                blob = get(row["Url"].strip(), config.MS_CACHE / f"{key}.csv.gz")
                with gzip.open(io.BytesIO(blob), "rt", encoding="utf-8") as fh:
                    feats = [json.loads(line) for line in fh if line.strip()]
            self._tiles[key] = feats
        return self._tiles[key]

    def run(self, loc: Location) -> RefResult:
        b = loc.grid.bounds()
        keys = sorted({quadkey(la, lo) for la in (b["top"], b["bottom"])
                       for lo in (b["left"], b["right"])})
        h, w = loc.shape
        feats = []
        for k in keys:
            for f in self.tile(k):
                g = f.get("geometry") or {}
                if g.get("type") != "Polygon":
                    continue
                ring = g["coordinates"][0]
                lons = [p[0] for p in ring]
                lats = [p[1] for p in ring]
                if (max(lons) < b["left"] or min(lons) > b["right"]
                        or max(lats) < b["bottom"] or min(lats) > b["top"]):
                    continue
                feats.append(polygon_feature([[float(x), float(y)] for x, y in ring], "building",
                                             {"confidence": (f.get("properties") or {}).get("confidence")}))
        return RefResult(vectors=feats, info={"quadkeys": keys, "missing_quadkeys":
                                              [k for k in keys if k not in self.rows()]})

    def close(self):
        self._tiles.clear()
