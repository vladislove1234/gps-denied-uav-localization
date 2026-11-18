"""Paths and the class registry. Every other module reads them from here."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent                      # the repo root, src/

# Input dataset (read-only) and the annotation workspace (outside it).
DATASET = Path(os.environ.get("ANNOT_DATASET", SRC / "data/derived/ua-village-multidate"))
ANNOT = Path(os.environ.get("ANNOT_ROOT", SRC / "data/derived/annotation"))
REFS = ANNOT / "refs"
WORK = ANNOT / "work"

# Export target: bytes in data/derived, recipe/card/manifest in datasets/.
EXPORT_NAME = "ua-village-annotated"
EXPORT_DATA = Path(os.environ.get("ANNOT_EXPORT", SRC / "data/derived" / EXPORT_NAME))
# Only the real export updates datasets/<name>/manifest.json; a scratch export
# (checks/e2e.py sets ANNOT_EXPORT) must never touch it.
EXPORT_IS_REAL = "ANNOT_EXPORT" not in os.environ
EXPORT_RECIPE = SRC / "datasets" / EXPORT_NAME

# Collections: each input dataset is annotated into its own export. The
# satellite names above stay the defaults so nothing that exists changes.
# Drone frames are pixel-only (no georeference, no references, no seeding).
DRONE_DATASET = Path(os.environ.get("ANNOT_DRONE_DATASET", SRC / "data/derived/drone-2025-easy"))
DRONE_EXPORT_NAME = "ua-drone-annotated"
DRONE_EXPORT = Path(os.environ["ANNOT_EXPORT"] + "-drone") if "ANNOT_EXPORT" in os.environ \
    else SRC / "data/derived" / DRONE_EXPORT_NAME
ORTHO_DATASET = Path(os.environ.get("ANNOT_ORTHO_DATASET", SRC / "data/derived/gonchary-2025-ortho"))
ORTHO_EXPORT_NAME = "ua-ortho-annotated"
ORTHO_EXPORT = Path(os.environ["ANNOT_EXPORT"] + "-ortho") if "ANNOT_EXPORT" in os.environ \
    else SRC / "data/derived" / ORTHO_EXPORT_NAME
COLLECTIONS = {
    "sat": {"dataset": DATASET, "export": EXPORT_DATA, "export_name": EXPORT_NAME, "label": "satellite",
            "description": "Hand-corrected 6-class segmentation labels over ua-village-multidate: "
                           "reference sources seeded per class, fixed in tools/annotator."},
    "drone": {"dataset": DRONE_DATASET, "export": DRONE_EXPORT, "export_name": DRONE_EXPORT_NAME, "label": "drone",
              "description": "Hand-drawn 6-class segmentation labels over drone frames, in frame pixels "
                             "(no georeference), drawn in tools/annotator with no reference seeding."},
    "ortho": {"dataset": ORTHO_DATASET, "export": ORTHO_EXPORT, "export_name": ORTHO_EXPORT_NAME, "label": "ortho",
              "description": "Hand-drawn 6-class segmentation labels over test-area satellite images that have "
                             "no exact georeference (Google Earth frames), in image pixels, no reference seeding."},
}
# collections without a Web Mercator grid: drawn by hand, stored in pixels
PIXEL_ONLY = ("drone", "ortho")

# Caches shared with the PoCs, so nothing is downloaded twice.
OSM_CACHE = SRC / "data/derived/osm-cache"
MS_CACHE = SRC / "data/derived/ms-buildings"
LANDCOVERAI_WEIGHTS = Path(os.environ.get(
    "LANDCOVERAI_WEIGHTS", SRC / "pocs/landcoverai-finetune/model.pt"))


@lru_cache(maxsize=1)
def registry() -> dict:
    return json.loads((HERE / "classes.json").read_text())


def classes() -> list[dict]:
    return registry()["classes"]


def classes_version() -> int:
    return int(registry()["version"])


def cls(name: str) -> dict:
    for c in classes():
        if c["name"] == name:
            return c
    raise KeyError(name)


def editable() -> list[dict]:
    """Classes that have layers (everything but background)."""
    return [c for c in classes() if c["geometry"] != "none"]


def _slugs(coll: str) -> list[str]:
    root = COLLECTIONS[coll]["dataset"]
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())


def locations(collection: str | None = None) -> list[str]:
    """Location slugs, sorted: one collection, or all of them (satellite first)."""
    colls = [collection] if collection else list(COLLECTIONS)
    return [s for c in colls for s in _slugs(c)]


def collection_of(slug: str) -> str:
    for c, v in COLLECTIONS.items():
        if (v["dataset"] / slug / "meta.json").exists():
            return c
    raise FileNotFoundError(slug)


def loc_dir(slug: str) -> Path:
    return COLLECTIONS[collection_of(slug)]["dataset"] / slug


def meta(slug: str) -> dict:
    return json.loads((loc_dir(slug) / "meta.json").read_text())


def dates(slug: str) -> list[str]:
    """Image dates of a location, oldest first, ``current`` last. A drone
    frame has one: ``frame``."""
    names = [p.stem for p in loc_dir(slug).glob("*.jpg")]
    named = sorted(n for n in names if n != "current")
    return named + (["current"] if "current" in names else [])
