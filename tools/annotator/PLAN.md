# Annotator — implementation plan

2026-09-25. Implements [SPEC.md](SPEC.md). **Status: M0–M5 built and checked** (see bottom). Six milestones (M0–M5). Each one ends
with something you can open and check before the next one starts.

## Ground rules

- Python runs in the `uavloc` env. New packages: `fastapi`, `uvicorn`. Nothing
  else is needed: `shapely`, `rasterio`, `cv2`, `torch`, `smp` are already
  there.
- **PoC code is copied into `tools/annotator/`, not imported from `pocs/`.**
  PoCs stay frozen as the record, and the annotator must not break if a PoC
  folder changes.
  Sources: `map-derived-labels/{tiles,geo,osm}.py`, `class-label-sources/worldcover.py`,
  and the `ms-building-footprints` / `landcoverai-finetune` /
  `flair-landcover-transfer` probe notebooks.
- Weights are not moved. The LandCover.ai adapter points at
  `pocs/landcoverai-finetune/model.pt` via config. FLAIR comes from the HF cache.
- Frontend: vanilla JS, no build step. Konva and OpenCV.js are downloaded once
  by `python -m tools.annotator vendor` into `web/vendor/` (gitignored), so the
  app works offline.
- Every milestone has a **check notebook** in `tools/annotator/checks/`. It holds
  numbers and figures you can read, per the equipment rule "quality evidence
  even without a bar".

## Target layout

```
src/tools/annotator/
  SPEC.md  PLAN.md  classes.json
  __main__.py              serve | vendor | export
  geo.py                   Web Mercator grid, latlon <-> px (from tiles.py/geo.py)
  refs/
    base.py                Adapter, Ref, Location, read/write refs
    vectorize.py           mask -> polygons, polygons/polylines -> mask
    osm.py  ms_footprints.py  landcoverai.py  flair.py  worldcover.py
    __main__.py            CLI: run --source --locations [--force]
  server/
    app.py                 FastAPI routes
    workspace.py           state, sublayers, overrides I/O, seeding
    export.py              compose, rasterize, priority, samples.jsonl
  web/
    index.html  css/  js/  vendor/
  checks/
    m1_refs.ipynb  m3_vectors.ipynb  m5_export.ipynb
```

---

## M0 — Scaffold (small)

1. Folder skeleton, `classes.json` (6 classes: id, name, color, geometry, tools,
   priority, seed sublayers), `classes_version: 1`.
2. `geo.py`: `Grid.from_meta(meta.json)`, `latlon_to_px`, `px_to_latlon`. Copied
   math from `tiles.py`.
3. Install `fastapi uvicorn` into `uavloc`.

**Done when:** latlon → px → latlon round-trip error < 1e-6 px on all 59
`meta.json`.

## M1 — Reference adapters (largest piece of Python)

1. `refs/base.py`
   - `Location` (slug, meta, grid, image path for `current`).
   - `Ref` = GeoJSON FeatureCollection in lat/lon, or a uint8 mask on the grid.
   - Writes to `data/derived/annotation/refs/<source>/<loc>/` with a
     `meta.json`: adapter, version, params, runtime, and per-class counts / area.
2. `refs/vectorize.py`
   - `mask_to_polygons`: `cv2.findContours` with holes, Douglas–Peucker 0.5 m,
     minimum area per class.
   - `rasterize`: polygons + polylines-with-width → mask.
3. Adapters, one file each:

   | adapter | from | output |
   |---|---|---|
   | `osm` | `osm.fetch` + `classify` (Overpass, cached) | building/water/wood/farmland **polygons**; roads → **polylines with `width_m`** from the `road_width_m` table, so they seed as editable polylines in `track` |
   | `ms_footprints` | quadkey download (cached) + `features_for` | building polygons |
   | `landcoverai` | `predict_ua` @ 0.25 m, MPS | building (→ polygons), road → track, woodland → woody, water |
   | `flair` | HF `IGNF/FLAIR-INC_rgb_15cl_resnet34-fpn` @ native 0.386 m | building (→ polygons), woody (conif + decid + brushwood), water, field (agri classes, overlay only) |
   | `worldcover` | `worldcover.sample` + `class_masks` | woody mask (10 m, nearest upsample) |

   - OSM alignment: `meta.json` carries a measured `label_offset`. The adapter
     applies it to OSM geometry. `ms-footprints` found a 2.84 m offset
     independently, so check the sign on 3 locations before the full run.
4. CLI: `python -m tools.annotator.refs run --source all --locations all`.
   - Resumable (skips existing unless `--force`).
   - One log line per location.
   - Failures are logged and skipped, never fatal to the batch.
5. Full run over 59 locations.
   - Estimate: OSM/MS/WorldCover are network-bound, minutes.
   - LandCover.ai + FLAIR run on MPS, roughly 1–2 min per location each.

**Check (`m1_refs.ipynb`):**
- `osm` rasterized vs the existing `label.png`: per-class IoU ≥ 0.95 (proves
  georeferencing and the offset handling).
- MS / FLAIR / LandCover.ai building IoU vs OSM on the 9 PoC locations
  reproduces the PoC `results.json` within ±0.02. That shows the port did not
  change behaviour.
- Overlay figure of all sources on 3 locations. **You review it before M2.**

## M2 — Server + viewer (read-only)

1. `server/app.py`
   - `GET /api/locations`: slug, name, n_dates, status.
   - `GET /api/loc/{slug}`: meta, dates, available refs.
   - Static routes: images, ref files.
2. `web/`: layout from spec §5.2 / §6
   - location list (left), canvas (centre), layer panel (right).
   - Pan/zoom to 1:1+ on 3328×2560.
   - Dates: visibility, opacity, **swipe**, **blink** hotkey.
   - Ref overlays: every (source × class), toggle + opacity. Vectors via Konva;
     masks as tinted canvases.
3. Performance design, fixed now so M3/M4 don't redo it:
   - Raster layers are `Uint8Array` masks, painted into one canvas per class
     with dirty-rect redraw.
   - Konva only for vectors; non-active layers get `listening(false)` and are
     cached.

**Done when:** you open any location, compare dates by swipe/blink, and toggle
every source. Pan and zoom stay smooth with all OSM buildings shown.

## M3 — Seeding + vector editing

1. `workspace.py`
   - Creates `work/<loc>/` on first open.
   - **Seeds per `classes.json`**, one sublayer per source:
     - building ← osm, ms-footprints
     - track ← landcoverai (+ osm polylines)
     - woody ← osm, landcoverai, worldcover, flair
     - water ← osm
     - field ← empty `manual`
   - Every class also gets an empty `manual` sublayer.
   - Read/write vectors, sublayer PNGs, `state.json`.
2. API: `GET/PUT` vectors and sublayer masks, `POST` seed / re-seed / add
   source / delete / merge sublayer, `PUT` status.
3. Layer panel: class → sublayer tree with visibility, opacity, lock, delete,
   merge-into, add-source-sublayer. Active-sublayer highlight.
4. Tools: select (click/box, multi), move, Konva Transformer (resize/rotate),
   rotated rectangle, polygon, vertex drag/add/delete, delete. **Adopt**: click
   an overlay feature → copy into the chosen sublayer.
5. Undo/redo command stack. Autosave 2 s after the last edit, plus on location
   switch.
6. Status per location/date. Editing a `done` item sets it back to
   `in progress`.

**Check (`m3_vectors.ipynb`):**
- Save → reload is lossless (geometry equal to 1e-9 deg).
- Seeded counts equal ref counts.

**You annotate 1 location end-to-end on vectors** and report friction.

## M4 — Raster tools

1. Brush / eraser on the active raster sublayer: size in px or m, round stamp,
   interpolated strokes. Undo stores dirty-rect patches.
2. Track polyline-with-width tool: draw, edit vertices, set `width_m`,
   rasterized live preview.
3. Layer translate (class or sublayer): drag or numeric dx/dy in px/m. Moves
   vectors and masks together.
4. **Magic wand** (OpenCV.js in a Web Worker):
   - hover → ROI 512×512 around the cursor, Canny edges as a barrier, flood fill
     with color tolerance → preview outline, throttled to about 30 Hz
   - click → commit to the active sublayer; a vector class gets a polygon via
     contours + DP
   - sliders: tolerance, edge sensitivity, ROI size
5. Conflicts overlay: pixels claimed by ≥ 2 classes.

**Done when:** you clean one `woody` sublayer set and fill a road by wand on a
real location. Undo and reload behave.

## M5 — Multi-date overrides + export

1. Editing target switch: **BASE** / **date override**, with a coloured banner.
   In override mode:
   - drawing → `add`
   - erasing base pixels → `remove` mask
   - deleting a base feature → `remove.json` id
   - Render = base − remove + add.
2. Exclude date (with reason) and per-date status.
3. `export.py` + `python -m tools.annotator export --new|--stale|--all`:
   - per (location, date): compose base − remove + add, rasterize vectors
     (polygons, polylines at `width_m / mpp`)
   - union the sublayers of each class, apply priority, write class ids 0–5
   - `label.png` when there are no overrides, else `label-<date>.png`
   - `samples.jsonl` records per SPEC §8.1: `annotation_hash` = sha256 over the
     canonicalised base + override files; `classes_version`; refs versions;
     provenance px-share per sublayer
   - `--stale` and "needs review for class X" come from comparing records to the
     workspace
4. `datasets/ua-village-annotated/`: `build.py`, `card.md`, `manifest.json` per
   `datasets/_template`. Bytes in `data/derived/ua-village-annotated/`.

**Check (`m5_export.ipynb`):**
- Synthetic overlaps resolve by priority.
- An override date differs from base only inside its delta.
- Editing a done sample → `--stale` picks it up.
- Bumping `classes_version` → all samples flagged.
- Exported label overlaid on imagery for 3 locations.

## Later

- SAM point-prompt tool (server, MPS), same interface as the wand.
- More adapters: a new file in `refs/` + a registry entry.

## Risks

| risk | mitigation |
|---|---|
| Konva slow with ~1100 buildings × 2 sublayers | cache non-active layers; if still slow, render inactive vectors to a bitmap |
| 12+ full-size masks in the browser (~8.5 MB each as Uint8) | ~100 MB total, fine on a Mac; RGBA canvases only per class, not per sublayer |
| OSM offset sign / MS 2.84 m offset | M1 check on 3 locations before the full run |
| OpenCV.js hover latency | ROI-limited, worker thread, throttled |
| LandCover.ai `model.pt` exists only in the PoC folder | note in adapter config; fail with a clear message if missing |

## Status (2026-09-25)

| milestone | state | evidence |
|---|---|---|
| M0 | done | lat/lon ↔ px round-trip 9e-9 px on all 59 `meta.json`; `geo.js` matches `geo.py` (`web_selftest.mjs`) |
| M1 | done | 5 adapters × 59 locations, 0 failures; `checks/m1_refs.ipynb` |
| M2–M4 | done | `checks/e2e.py`: 36/36 in headless Chromium on real imagery; screenshots in `checks/figures/` |
| M5 | done | `checks/export_check.py`: 17/17 |

Not built: the SAM tool (later, as planned). Konva and OpenCV.js were replaced
by a Canvas2D renderer and a pure-JS wand; see SPEC §12.
