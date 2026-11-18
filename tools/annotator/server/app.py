"""HTTP API of the annotator. The browser holds the editing logic; this serves
images and references, and stores what the browser sends back.

Mask bodies are ``gzip(packbits(mask))``, row-major, the location's full shape.
"""

from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..refs import base as refs
from . import export as exporter
from .workspace import STATUSES, Workspace, _safe, list_locations, now

app = FastAPI(title="annotator")
_lock = threading.Lock()            # one writer at a time; single user, but autosave overlaps


def ws_or_404(slug: str) -> Workspace:
    try:
        return Workspace(slug)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"no location {slug!r}")


def mask_key(kind: str, cls: str, sub: str, date: str | None) -> tuple:
    if kind == "base":
        return ("base", cls, sub)
    if kind in ("add", "remove") and date:
        return (kind, date, cls, sub)
    raise HTTPException(400, "kind must be base, or add/remove with a date")


# -- config & locations -----------------------------------------------------------------

@app.get("/api/config")
def get_config():
    from ..refs import REGISTRY
    return {"registry": config.registry(), "statuses": STATUSES,
            "sources": {n: {"provides": list(a.provides), "version": a.version}
                        for n, a in REGISTRY.items()}}


@app.get("/api/locations")
def get_locations():
    return list_locations()


@app.get("/api/loc/{slug}")
def get_location(slug: str):
    ws = ws_or_404(slug)
    with _lock:
        state = ws.open()
    return {"slug": slug, "collection": config.collection_of(slug), "meta": ws.meta, "dates": config.dates(slug), "state": state,
            "refs": refs.available(slug)}


@app.get("/img/{slug}/{date}.jpg")
def get_image(slug: str, date: str):
    _safe(slug, date)
    try:
        p = config.loc_dir(slug) / f"{date}.jpg"
    except FileNotFoundError:
        raise HTTPException(404)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


# -- references (read-only) --------------------------------------------------------------

@app.get("/api/refs/{source}/{slug}/vectors")
def get_ref_vectors(source: str, slug: str):
    _safe(source, slug)
    p = refs.ref_dir(source, slug) / "vectors.geojson"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json")


@app.get("/api/refs/{source}/{slug}/mask/{cls}")
def get_ref_mask(source: str, slug: str, cls: str):
    _safe(source, slug, cls)
    m = refs.read_mask(source, slug, cls)
    if m is None:
        return Response(status_code=204)
    return Response(ws_or_404(slug).pack(m), media_type="application/octet-stream")


# -- workspace ------------------------------------------------------------------------------

@app.get("/api/loc/{slug}/work")
def get_work(slug: str):
    """Everything vector-shaped plus the list of masks that exist."""
    ws = ws_or_404(slug)
    with _lock:
        state = ws.open()
        dates = config.dates(slug)
        return {
            "state": state,
            "base": ws.read_vectors(("base",)),
            "overrides": {d: {"add": ws.read_vectors(("ovr", d)), "remove": ws.read_removed(d)}
                          for d in dates},
            "masks": [list(k) for k in ws.mask_keys()],
        }


@app.get("/api/loc/{slug}/mask")
def get_mask(slug: str, kind: str, cls: str, sub: str, date: str | None = None):
    ws = ws_or_404(slug)
    m = ws.read_mask(mask_key(kind, cls, sub, date))
    if m is None:
        return Response(status_code=204)
    return Response(ws.pack(m), media_type="application/octet-stream")


@app.put("/api/loc/{slug}/mask")
async def put_mask(slug: str, request: Request, kind: str, cls: str, sub: str,
                   date: str | None = None, empty: int = 0):
    ws = ws_or_404(slug)
    key = mask_key(kind, cls, sub, date)
    blob = await request.body()
    try:
        m = None if empty else ws.unpack(blob)
    except Exception as exc:          # noqa: BLE001
        raise HTTPException(400, f"bad mask body: {exc}")
    with _lock:
        ws.write_mask(key, m)
        ws.touch()
    return {"ok": True}


@app.put("/api/loc/{slug}/vectors")
async def put_vectors(slug: str, request: Request, date: str | None = None):
    ws = ws_or_404(slug)
    body = await request.json()
    feats = body["features"] if isinstance(body, dict) else body
    with _lock:
        ws.write_vectors(("ovr", date) if date else ("base",), feats)
        ws.touch()
    return {"ok": True, "n": len(feats)}


@app.put("/api/loc/{slug}/removed/{date}")
async def put_removed(slug: str, date: str, request: Request):
    ws = ws_or_404(slug)
    body = await request.json()
    with _lock:
        ws.write_removed(date, body["feature_ids"])
        ws.touch()
    return {"ok": True}


@app.delete("/api/loc/{slug}/override/{date}")
def delete_override(slug: str, date: str):
    ws = ws_or_404(slug)
    with _lock:
        ws.delete_date_override(date)
    return {"ok": True}


@app.put("/api/loc/{slug}/state")
async def put_state(slug: str, request: Request):
    """Replace the editable parts of state: sublayers, dates, rejected, note."""
    ws = ws_or_404(slug)
    body = await request.json()
    with _lock:
        st = ws.open()
        for k in ("sublayers", "rejected", "note"):
            if k in body:
                st[k] = body[k]
        if "dates" in body:
            for d, s in body["dates"].items():
                if d not in st["dates"]:
                    continue
                prev = st["dates"][d]["status"]
                st["dates"][d].update({k: s[k] for k in ("status", "excluded", "reason") if k in s})
                if "offset_px" in s:
                    off = [int(round(float(v))) for v in s["offset_px"]]
                    if len(off) != 2 or max(abs(v) for v in off) > 200:
                        raise HTTPException(400, f"bad offset_px {s['offset_px']!r} for {d}")
                    if any(off):
                        st["dates"][d]["offset_px"] = off
                    else:
                        st["dates"][d].pop("offset_px", None)
                if st["dates"][d]["status"] == "done" and prev != "done":
                    st["dates"][d]["done_at"] = now()
        try:
            ws.write_state(st)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return st


@app.post("/api/loc/{slug}/reset")
def reset_location(slug: str):
    """Throw away the workspace and re-seed from references. Irreversible."""
    import shutil
    ws = ws_or_404(slug)
    with _lock:
        shutil.rmtree(ws.dir, ignore_errors=True)
        return ws.open()


# -- export ------------------------------------------------------------------------------------

@app.get("/api/export/plan")
def export_plan():
    per = {}
    for c in config.COLLECTIONS:
        p = exporter.plan("all", c)
        per[c] = {"new": len(p["new"]), "stale": len(p["stale"]), "done": len(p["export"]),
                  "orphaned": p["orphaned"]}
    tot = lambda k: sum(v[k] for v in per.values())
    return {"new": tot("new"), "stale": tot("stale"), "done": tot("done"),
            "orphaned": [o for v in per.values() for o in v["orphaned"]],
            "needs_review": exporter.needs_review(), "collections": per}


@app.post("/api/export")
def run_export(mode: str = "new"):
    if mode not in ("new", "stale", "all"):
        raise HTTPException(400, "mode must be new, stale or all")
    lines: list[str] = []
    with _lock:
        summary = exporter.export(mode, log=lines.append)
    return {**summary, "log": lines}


# -- static frontend (mounted last so /api wins) ----------------------------------------------

app.mount("/", StaticFiles(directory=config.HERE / "web", html=True), name="web")


@app.exception_handler(ValueError)
def value_error(_, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)
