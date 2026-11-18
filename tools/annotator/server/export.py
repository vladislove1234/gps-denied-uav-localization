"""Export `done` annotations to the ``ua-village-annotated`` dataset.

Per (location, date):
    class mask   = union over its sublayers of
                   (base - remove(date)) + add(date), vectors rasterized
    label        = class ids 0..5, overlaps resolved by registry priority
    file         = label.png if no date of the location has an override,
                   else label-<date>.png

Every exported sample gets a record in ``samples.jsonl``, keyed by
(location, date). The records make the dataset extendable without a rebuild:

    --new    done samples with no record yet
    --stale  done samples whose annotation_hash or classes_version changed
    --all    every done sample

Images are not copied: records point into ``ua-village-multidate``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from ..geo import grid_for
from ..refs import vectorize
from .workspace import Workspace


# -- composition -------------------------------------------------------------------

def date_layers(ws: Workspace, date: str) -> dict[str, dict[str, np.ndarray]]:
    """{class: {sublayer: bool mask}} for one date, overrides applied."""
    grid = grid_for(ws.meta)
    mpp = float(ws.meta["mpp"])
    removed = set(ws.read_removed(date))
    feats = [f for f in ws.read_vectors(("base",)) if f["id"] not in removed]
    feats += ws.read_vectors(("ovr", date))

    out: dict[str, dict[str, np.ndarray]] = {}
    state = ws.open()
    for c in config.editable():
        name = c["name"]
        subs = {s["name"] for s in state["sublayers"].get(name, [])}
        subs |= {f["properties"]["sublayer"] for f in feats if f["properties"]["class"] == name}
        for sub in sorted(subs):
            m = ws.read_mask(("base", name, sub))
            m = np.zeros(ws.shape, bool) if m is None else m
            r = ws.read_mask(("remove", date, name, sub))
            if r is not None:
                m &= ~r
            a = ws.read_mask(("add", date, name, sub))
            if a is not None:
                m |= a
            fs = [f for f in feats if f["properties"]["class"] == name
                  and f["properties"]["sublayer"] == sub]
            if fs:
                m |= vectorize.rasterize(fs, grid, ws.shape, mpp)
            if m.any():
                out.setdefault(name, {})[sub] = m
    return out


def compose(layers: dict[str, dict[str, np.ndarray]], shape) -> tuple[np.ndarray, dict]:
    """Class-id label with priority applied, and per-sublayer provenance."""
    label = np.zeros(shape, np.uint8)
    by_priority = sorted(config.editable(), key=lambda c: -c["priority"])   # lowest first
    for c in by_priority:
        subs = layers.get(c["name"])
        if subs:
            union = np.logical_or.reduce(list(subs.values()))
            label[union] = c["id"]
    prov: dict[str, dict] = {}
    for c in config.editable():
        final = label == c["id"]
        n = int(final.sum())
        if not n:
            continue
        prov[c["name"]] = {"px": n, "frac": n / label.size,
                           "sublayers": {s: round(float((m & final).sum()) / n, 4)
                                         for s, m in layers.get(c["name"], {}).items()}}
    return label, prov


def to_image_frame(label: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Layers live in the reference frame. A date aligned by (dx, dy) draws its
    image shifted by (dx, dy), so layer point (x, y) is that image's pixel
    (x - dx, y - dy): out[v, u] = label[v + dy, u + dx]. The strip the shift
    uncovers is background (0) -- recorded as ``image_offset_px`` so a loader
    can crop it."""
    dx, dy = offset
    if not dx and not dy:
        return label
    h, w = label.shape
    out = np.zeros_like(label)
    ud = slice(max(0, -dx), min(w, w - dx)); us = slice(max(0, dx), min(w, w + dx))
    vd = slice(max(0, -dy), min(h, h - dy)); vs = slice(max(0, dy), min(h, h + dy))
    out[vd, ud] = label[vs, us]
    return out


def annotation_hash(ws: Workspace, date: str) -> str:
    """Hash of everything that decides this date's label -- content, not files."""
    h = hashlib.sha256()
    off = ws.image_offset(date)
    if any(off):                     # unset offsets leave every existing hash unchanged
        h.update(f"offset:{off}".encode())
    removed = sorted(ws.read_removed(date))
    for scope in (("base",), ("ovr", date)):
        feats = sorted(ws.read_vectors(scope), key=lambda f: f["id"])
        if scope[0] == "base":
            feats = [f for f in feats if f["id"] not in set(removed)]
        h.update(json.dumps([[f["id"], f["properties"].get("class"), f["properties"].get("sublayer"),
                              f["properties"].get("width_m"), f["geometry"]] for f in feats],
                            sort_keys=True).encode())
    h.update(json.dumps(removed).encode())
    for key in ws.mask_keys():
        if key[0] == "base" or key[1] == date:
            m = ws.read_mask(key)
            h.update(repr(key).encode())
            h.update(np.packbits(m).tobytes())
    return h.hexdigest()


# -- records -------------------------------------------------------------------------

# Every function below works on one collection (satellite by default); each
# collection exports into its own dataset with its own samples.jsonl.

def records_path(coll: str = "sat") -> Path:
    return config.COLLECTIONS[coll]["export"] / "samples.jsonl"


def read_records(coll: str = "sat") -> dict[tuple[str, str], dict]:
    p = records_path(coll)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["location"], r["date"])] = r
    return out


def write_records(recs: dict[tuple[str, str], dict], coll: str = "sat") -> None:
    p = records_path(coll)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                           for _, r in sorted(recs.items())))
    tmp.replace(p)


def done_samples(coll: str = "sat") -> list[tuple[str, str]]:
    out = []
    for slug in config.locations(coll):
        ws = Workspace(slug)
        if not ws.exists():
            continue
        st = ws.open()
        if st.get("rejected"):
            continue
        for d, s in st["dates"].items():
            if s["status"] == "done" and not s["excluded"]:
                out.append((slug, d))
    return out


def plan(mode: str, coll: str = "sat") -> dict:
    """Which samples an export in ``mode`` would write, and why the others not."""
    recs = read_records(coll)
    done = done_samples(coll)
    todo, fresh, stale = [], [], []
    for slug, d in done:
        r = recs.get((slug, d))
        if r is None:
            fresh.append((slug, d))
            continue
        h = annotation_hash(Workspace(slug), d)
        if h != r["annotation_hash"] or r["classes_version"] != config.classes_version():
            stale.append((slug, d))
    todo = {"new": fresh, "stale": stale, "all": done}[mode]
    done_set = set(done)
    orphaned = [k for k in recs if k not in done_set]
    return {"export": todo, "new": fresh, "stale": stale, "orphaned": orphaned}


def needs_review(coll: str | None = None) -> list[dict]:
    """Exported samples reviewed under an older class registry (all
    collections unless one is named)."""
    v = config.classes_version()
    return [{"location": r["location"], "date": r["date"], "classes_version": r["classes_version"]}
            for c in ([coll] if coll else config.COLLECTIONS)
            for r in read_records(c).values() if r["classes_version"] != v]


# -- export -----------------------------------------------------------------------------

def export(mode: str = "new", log=print) -> dict:
    """Export every collection; the summary sums them and keeps each apart."""
    t0 = time.time()
    per = {c: export_collection(mode, c, log) for c in config.COLLECTIONS}
    tot = lambda k: sum(v[k] for v in per.values())
    return {"mode": mode, "written": tot("written"), "locations": tot("locations"),
            "new": tot("new"), "stale": tot("stale"),
            "orphaned": [o for v in per.values() for o in v["orphaned"]],
            "needs_review": len(needs_review()), "seconds": round(time.time() - t0, 1),
            "collections": per}


def export_collection(mode: str, coll: str, log=print) -> dict:
    p = plan(mode, coll)
    recs = read_records(coll)
    out_root = config.COLLECTIONS[coll]["export"]
    dataset = config.COLLECTIONS[coll]["dataset"]
    written = []

    by_loc: dict[str, list[str]] = {}
    for slug, d in p["export"]:
        by_loc.setdefault(slug, []).append(d)

    done_by_loc: dict[str, list[str]] = {}
    for slug, d in done_samples(coll):
        done_by_loc.setdefault(slug, []).append(d)

    for slug, dates in sorted(by_loc.items()):
        ws = Workspace(slug)
        st = ws.open()
        live = [d for d, s in st["dates"].items() if not s["excluded"]]
        per_date = any(ws.has_override(d) or any(ws.image_offset(d)) for d in live)
        loc_dir = out_root / slug
        loc_dir.mkdir(parents=True, exist_ok=True)
        # Layout switch (shared <-> per-date) invalidates the files every other
        # record of this location points at: rewrite all its done dates, and
        # drop the old-layout files.
        if any(r["per_date_labels"] != per_date for k, r in recs.items() if k[0] == slug):
            dates = sorted(set(dates) | set(done_by_loc.get(slug, [])))
            for old in loc_dir.glob("label*.png"):
                old.unlink()
            for k in [k for k in recs if k[0] == slug and k[1] not in dates]:
                del recs[k]
        shared = None
        for d in sorted(dates):
            if per_date or shared is None:
                label, prov = compose(date_layers(ws, d), ws.shape)
                label = to_image_frame(label, ws.image_offset(d))
                fname = f"label-{d}.png" if per_date else "label.png"
                Image.fromarray(label, "L").save(loc_dir / fname, optimize=True)
                shared = (fname, prov)
            fname, prov = shared
            recs[(slug, d)] = {
                "location": slug, "date": d,
                "label_file": f"{slug}/{fname}",
                "collection": coll,
                "image_file": f"{dataset.name}/{slug}/{d}.jpg",
                "done_at": st["dates"][d].get("done_at"),
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "annotation_hash": annotation_hash(ws, d),
                "classes_version": config.classes_version(),
                "classes": {c["name"]: c["id"] for c in config.classes()},
                "refs": {c: [{"sublayer": s["name"], "source": s.get("source"),
                              "source_version": s.get("source_version")}
                             for s in subs if s.get("source")]
                         for c, subs in st["sublayers"].items()},
                "provenance": prov,
                "mpp": ws.meta["mpp"], "shape": ws.meta["shape"],
                **({"bounds": ws.meta["bounds"], "grid": ws.meta["grid"]} if "grid" in ws.meta else
                   {"crs": "pixel", "position": {k: ws.meta.get(k) for k in
                                                 ("timestamp_ms", "lat", "lon", "alt_m", "yaw_deg", "flight")
                                                 if k in ws.meta},
                    **{k: ws.meta[k] for k in ("source_file", "source_shape", "scale_to_source", "bounds_approx")
                       if k in ws.meta}}),
                "excluded_dates": {k: v.get("reason", "") for k, v in st["dates"].items()
                                   if v["excluded"]},
                "per_date_labels": per_date,
                "image_offset_px": list(ws.image_offset(d)),
            }
            written.append((slug, d))
        log(f"  {slug:32s} {len(dates)} date(s) -> {'per-date' if per_date else 'label.png'}")

    if written or recs:
        write_records(recs, coll)
    summary = {"written": len(written), "locations": len(by_loc),
               "new": len(p["new"]), "stale": len(p["stale"]),
               "orphaned": [list(k) for k in p["orphaned"]]}
    if written and config.EXPORT_IS_REAL:
        write_dataset_manifest(recs, coll)
    return summary


def write_dataset_manifest(recs: dict, coll: str = "sat") -> None:
    import sys
    sys.path.insert(0, str(config.SRC))
    from uavloc.datasets import write_manifest

    locs = sorted({k[0] for k in recs})
    class_px: dict[str, int] = {}
    for r in recs.values():
        for c, p in r["provenance"].items():
            class_px[c] = class_px.get(c, 0) + p["px"]
    total = sum(class_px.values()) or 1
    c = config.COLLECTIONS[coll]
    desc = c["description"]
    source = f"datasets/{c['dataset'].name} + tools/annotator" + (
        " (refs: osm, ms-footprints, landcoverai, flair, worldcover)" if coll == "sat" else "")
    write_manifest(
        c["export_name"], version="1",
        description=desc,
        source=source,
        config={"classes": {c["name"]: c["id"] for c in config.classes()},
                "priority": [c["name"] for c in sorted(config.editable(), key=lambda c: c["priority"])],
                "classes_version": config.classes_version(), "images_from": c["dataset"].name},
        samples=len(recs),
        checks={"locations": len(locs), "samples": len(recs),
                "class_share_of_labelled_px": {c: round(v / total, 4) for c, v in class_px.items()}},
        root=c["export"])
