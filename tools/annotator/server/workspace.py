"""The annotation workspace of one location: sublayers, overrides, state.

    work/<slug>/
        state.json
        base/vectors.geojson                     features: class, sublayer, kind, width_m?
        base/masks/<class>/<sublayer>.png        0/255, raster part of a sublayer
        overrides/<date>/add.geojson             features present on this date only
        overrides/<date>/add/<class>/<sub>.png   pixels present on this date only
        overrides/<date>/remove.json             {"feature_ids": [...]} hidden on this date
        overrides/<date>/remove/<class>/<sub>.png  base pixels erased on this date

Label for a date = base - remove(date) + add(date). Base edits therefore reach
every date except where an override removed them.

Masks travel to and from the browser as gzip(packbits(mask)) -- exact, and a
full 3328x2560 layer is ~1 MB before compression.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from ..refs import base as refs

NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
STATUSES = ("todo", "in_progress", "done", "rejected")


def _safe(*parts: str) -> None:
    for p in parts:
        if not NAME.match(p) or p in (".", ".."):
            raise ValueError(f"bad path component {p!r}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Workspace:
    def __init__(self, slug: str):
        _safe(slug)
        self.collection = config.collection_of(slug)      # FileNotFoundError if unknown
        self.slug = slug
        self.dir = config.WORK / slug
        self.meta = config.meta(slug)
        self.shape = tuple(self.meta["shape"])       # (h, w)

    # -- existence / creation --------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def exists(self) -> bool:
        return self.state_path.exists()

    def open(self) -> dict:
        """State of the workspace, creating and seeding it on first open."""
        if not self.exists():
            self.create()
        state = json.loads(self.state_path.read_text())
        if self._migrate(state):
            self.write_state(state)
        return state

    def create(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {
            "location": self.slug, "classes_version": config.classes_version(),
            "created": now(), "updated": now(), "rejected": False, "note": "",
            "sublayers": {}, "seed_missing": {},
            "dates": {d: {"status": "todo", "excluded": False, "reason": "", "done_at": None}
                      for d in config.dates(self.slug)},
        }
        features: list[dict] = []
        for c in config.editable():
            subs = state["sublayers"].setdefault(c["name"], [])
            # pixel-only frames (drone) have no references: manual layers only
            for src in (c["seed"] if self.collection == "sat" else []):
                meta = refs.read_meta(src, self.slug)
                if meta is None:
                    state["seed_missing"].setdefault(c["name"], []).append(src)
                    continue
                subs.append({"name": src, "source": src, "source_version": meta["version"],
                             "seeded": now()})
                features += self.copy_ref_vectors(src, c["name"], src)
                m = refs.read_mask(src, self.slug, c["name"])
                if m is not None and m.any():
                    self.write_mask(("base", c["name"], src), m)
            subs.append({"name": "manual", "source": None})
        self.write_vectors(("base",), features)
        self.write_state(state)
        return state

    def copy_ref_vectors(self, source: str, cls: str, sublayer: str) -> list[dict]:
        out = []
        for f in refs.read_vectors(source, self.slug):
            if f["properties"]["class"] != cls:
                continue
            p = dict(f["properties"], sublayer=sublayer, ref=f"{source}:{f.get('id')}")
            out.append({"type": "Feature", "id": new_id(), "properties": p,
                        "geometry": f["geometry"]})
        return out

    def _migrate(self, state: dict) -> bool:
        """Bring an older workspace up to the current registry (new classes,
        new dates) without touching what was drawn."""
        changed = False
        for c in config.editable():
            if c["name"] not in state["sublayers"]:
                state["sublayers"][c["name"]] = [{"name": "manual", "source": None}]
                changed = True
        for d in config.dates(self.slug):
            if d not in state["dates"]:
                state["dates"][d] = {"status": "todo", "excluded": False, "reason": "", "done_at": None}
                changed = True
        return changed

    # -- state -------------------------------------------------------------------

    def write_state(self, state: dict) -> None:
        for d, s in state["dates"].items():
            if s["status"] not in STATUSES:
                raise ValueError(f"bad status {s['status']!r} for {d}")
        state["updated"] = now()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_path)

    def status(self) -> str:
        """Location status, derived from its dates."""
        if not self.exists():
            return "todo"
        st = json.loads(self.state_path.read_text())
        if st.get("rejected"):
            return "rejected"
        live = [s for s in st["dates"].values() if not s["excluded"]]
        if live and all(s["status"] == "done" for s in live):
            return "done"
        if any(s["status"] != "todo" for s in st["dates"].values()) or st.get("edited"):
            return "in_progress"
        return "todo"

    def touch(self) -> None:
        """Record that something was drawn (moves the location out of `todo`)."""
        st = json.loads(self.state_path.read_text())
        if not st.get("edited"):
            st["edited"] = True
            self.write_state(st)

    # -- vectors -----------------------------------------------------------------
    # scope = ("base",) | ("ovr", date)

    def vectors_path(self, scope: tuple) -> Path:
        if scope[0] == "base":
            return self.dir / "base" / "vectors.geojson"
        _safe(scope[1])
        return self.dir / "overrides" / scope[1] / "add.geojson"

    def read_vectors(self, scope: tuple) -> list[dict]:
        p = self.vectors_path(scope)
        return json.loads(p.read_text())["features"] if p.exists() else []

    def write_vectors(self, scope: tuple, features: list[dict]) -> None:
        p = self.vectors_path(scope)
        p.parent.mkdir(parents=True, exist_ok=True)
        for f in features:
            f.setdefault("id", new_id())
            f.setdefault("type", "Feature")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                  separators=(",", ":")))
        tmp.replace(p)

    def read_removed(self, date: str) -> list[str]:
        _safe(date)
        p = self.dir / "overrides" / date / "remove.json"
        return json.loads(p.read_text())["feature_ids"] if p.exists() else []

    def write_removed(self, date: str, ids: list[str]) -> None:
        _safe(date)
        p = self.dir / "overrides" / date / "remove.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"feature_ids": sorted(set(ids))}))

    # -- masks -------------------------------------------------------------------
    # key = ("base", cls, sub) | ("add", date, cls, sub) | ("remove", date, cls, sub)

    def mask_path(self, key: tuple) -> Path:
        _safe(*key[1:])
        if key[0] == "base":
            _, cls, sub = key
            return self.dir / "base" / "masks" / cls / f"{sub}.png"
        if key[0] in ("add", "remove"):
            kind, date, cls, sub = key
            return self.dir / "overrides" / date / kind / cls / f"{sub}.png"
        raise ValueError(f"bad mask key {key}")

    def read_mask(self, key: tuple) -> np.ndarray | None:
        p = self.mask_path(key)
        return np.array(Image.open(p)) > 0 if p.exists() else None

    def write_mask(self, key: tuple, mask: np.ndarray | None) -> None:
        p = self.mask_path(key)
        if mask is None or not mask.any():
            p.unlink(missing_ok=True)
            return
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} != {self.shape}")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.png")
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(tmp, optimize=False, compress_level=6)
        tmp.replace(p)

    def mask_keys(self) -> list[tuple]:
        """Every mask file that exists, as keys."""
        out = []
        base = self.dir / "base" / "masks"
        if base.exists():
            for p in sorted(base.glob("*/*.png")):
                out.append(("base", p.parent.name, p.stem))
        ov = self.dir / "overrides"
        if ov.exists():
            for p in sorted(ov.glob("*/*/*/*.png")):
                out.append((p.parent.parent.name, p.parent.parent.parent.name, p.parent.name, p.stem))
        return out

    def delete_date_override(self, date: str) -> None:
        import shutil
        _safe(date)
        shutil.rmtree(self.dir / "overrides" / date, ignore_errors=True)

    def image_offset(self, date: str) -> tuple[int, int]:
        """Per-date image alignment (dx, dy) in px: that date's image is drawn
        shifted by it under the shared layers. (0, 0) when never set."""
        st = json.loads(self.state_path.read_text()) if self.exists() else {}
        off = st.get("dates", {}).get(date, {}).get("offset_px") or [0, 0]
        return int(off[0]), int(off[1])

    def has_override(self, date: str) -> bool:
        _safe(date)
        d = self.dir / "overrides" / date
        if not d.exists():
            return False
        return bool(self.read_vectors(("ovr", date)) or self.read_removed(date)
                    or any(d.glob("*/*/*.png")))

    # -- wire format ---------------------------------------------------------------

    def pack(self, mask: np.ndarray) -> bytes:
        return gzip.compress(np.packbits(mask.astype(bool), axis=None).tobytes(), compresslevel=5)

    def unpack(self, blob: bytes) -> np.ndarray:
        h, w = self.shape
        bits = np.frombuffer(gzip.decompress(blob), np.uint8)
        return np.unpackbits(bits, count=h * w).reshape(h, w).astype(bool)


def list_locations() -> list[dict]:
    out = []
    for slug in config.locations():
        ws = Workspace(slug)
        m = ws.meta
        out.append({"slug": slug, "collection": ws.collection, "name": m.get("name", slug), "n_dates": len(config.dates(slug)),
                    "status": ws.status(), "lat": m.get("lat"), "lon": m.get("lon")})
    return out
