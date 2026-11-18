"""Adapter contract and the on-disk form of a reference.

A reference is what one source says about one location, on the latest date
(``current``). It is written once by ``python -m tools.annotator.refs run`` and
never edited: the annotator copies it into editable sublayers.

    refs/<source>/<slug>/
        meta.json          adapter, version, params, runtime, per-class stats
        vectors.geojson    features in lat/lon: polygons, polylines (+ width_m)
        <class>.png        0/255 mask on the location grid, raster classes only

Adding a source = one module in this package that subclasses ``Adapter`` and
calls ``register``. Nothing else changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from ..geo import Grid
from . import vectorize

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Location:
    slug: str

    def __post_init__(self):
        self.dir = config.loc_dir(self.slug)
        self.meta = config.meta(self.slug)
        self.grid = Grid.from_meta(self.meta)
        self.shape = self.grid.shape
        self.mpp = float(self.meta["mpp"])
        self.date = self.meta.get("reference_date", "current")

    def image(self, date: str | None = None) -> Image.Image:
        return Image.open(self.dir / f"{date or self.date}.jpg").convert("RGB")


@dataclass
class RefResult:
    """What an adapter returns, before geometry conversion."""
    vectors: list[dict] = field(default_factory=list)       # GeoJSON features, lat/lon
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)                # anything worth recording


class Adapter:
    name: str = ""
    version: str = "1"
    provides: tuple[str, ...] = ()        # our class names
    params: dict = {}

    def run(self, loc: Location) -> RefResult:
        raise NotImplementedError

    def close(self) -> None:
        """Free models between batches (optional)."""


REGISTRY: dict[str, type[Adapter]] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    REGISTRY[cls.name] = cls
    return cls


# -- geometry normalisation: every source ends in the class's own form ---------

def normalise(result: RefResult, loc: Location, simplify_m: float = 0.5) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Convert a raw result to the geometry each class is edited in.

    * vector class, given as a mask   -> polygons (kept as polygons, not boxes)
    * raster class, given as polygons -> burnt into its mask
    * raster class with a polyline tool (track) keeps polylines as vectors,
      so an OSM road seeds as an editable centreline with a width
    """
    vectors: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    min_area = {"building": 8.0, "field": 400.0}

    for name, m in result.masks.items():
        c = config.cls(name)
        if c["geometry"] == "vector":
            vectors += vectorize.mask_to_features(m, loc.grid, name, loc.mpp, simplify_m,
                                                  min_area.get(name, 8.0))
        else:
            masks[name] = masks.get(name, np.zeros(loc.shape, bool)) | m.astype(bool)

    h, w = loc.shape
    for f in result.vectors:
        name = f["properties"]["class"]
        # OSM is fetched with a ~100 m margin: drop shapes that miss the image
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
        xy = loc.grid.ring_to_px(ring)
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        if max(xs) < 0 or min(xs) > w or max(ys) < 0 or min(ys) > h:
            continue
        c = config.cls(name)
        keep = c["geometry"] == "vector" or (
            f["geometry"]["type"] == "LineString" and "polyline" in c["tools"])
        if keep:
            vectors.append(f)
        else:
            m = masks.get(name, np.zeros(loc.shape, np.uint8))
            masks[name] = vectorize.burn(m.astype(np.uint8), [f], loc.grid, loc.mpp).astype(bool)

    for i, f in enumerate(vectors):
        f["id"] = f"{i}"
    return vectors, masks


# -- storage --------------------------------------------------------------------

def ref_dir(source: str, slug: str) -> Path:
    return config.REFS / source / slug


def exists(source: str, slug: str) -> bool:
    return (ref_dir(source, slug) / "meta.json").exists()


def write(adapter: Adapter, loc: Location, result: RefResult, seconds: float) -> dict:
    vectors, masks = normalise(result, loc)
    out = ref_dir(adapter.name, loc.slug)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    (out / "vectors.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": vectors}, separators=(",", ":")))
    stats: dict[str, dict] = {}
    for f in vectors:
        s = stats.setdefault(f["properties"]["class"], {"features": 0, "px": 0})
        s["features"] += 1
    for name, m in masks.items():
        if not m.any():
            continue
        Image.fromarray((m * 255).astype(np.uint8), "L").save(out / f"{name}.png", optimize=True)
        stats.setdefault(name, {"features": 0, "px": 0})["px"] = int(m.sum())

    meta = {
        "source": adapter.name, "version": adapter.version, "params": adapter.params,
        "location": loc.slug, "date": loc.date,
        "provides": list(adapter.provides),
        # which forms each class arrives in: the app loads exactly these files
        "classes": {n: {"vector": s["features"] > 0, "raster": s["px"] > 0}
                    for n, s in sorted(stats.items())},
        "stats": stats, "info": result.info,
        "seconds": round(seconds, 2), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def read_meta(source: str, slug: str) -> dict | None:
    p = ref_dir(source, slug) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def read_vectors(source: str, slug: str) -> list[dict]:
    p = ref_dir(source, slug) / "vectors.geojson"
    return json.loads(p.read_text())["features"] if p.exists() else []


def read_mask(source: str, slug: str, name: str) -> np.ndarray | None:
    p = ref_dir(source, slug) / f"{name}.png"
    return np.array(Image.open(p)) > 0 if p.exists() else None


def available(slug: str) -> dict[str, dict]:
    """Every source that has a reference for ``slug``: {source: meta}."""
    out = {}
    if config.REFS.exists():
        for src in sorted(p.name for p in config.REFS.iterdir() if p.is_dir()):
            m = read_meta(src, slug)
            if m:
                out[src] = m
    return out
