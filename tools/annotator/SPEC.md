# Annotator — spec

Draft 2026-09-25. Status: **built 2026-09-25** (M0–M5). Deviations from this text are listed in §12.

## 1. Purpose

A local app to turn machine-made reference labels into checked segmentation
labels for `ua-village-multidate`. Automatic sources (OSM, MS footprints,
LandCover.ai, FLAIR, WorldCover) each get some classes right and some wrong. The
annotator puts their output in front of a person, who adopts the good parts,
fixes the rest by hand, and exports a training-ready mask.

This is **correction, not labelling from scratch.** Every location starts
from reference output.

Non-goals: multi-user, drone frames, training, changing the imagery.

## 2. Input

`data/derived/ua-village-multidate/<location>/`, read-only:

| file | content |
|---|---|
| `<date>.jpg`, `current.jpg` | 3328 × 2560 px, z18, 0.385 m/px, all dates co-registered to `current` |
| `label.png` | OSM class mask (old 5-class set, used only as an OSM ref) |
| `meta.json` | bounds, tile grid, `mpp`, `reference_date` |

All 59 locations are in scope.

## 3. Classes

Class registry `classes.json`. It is the one place a class is defined, so
**adding a class = adding an entry** and nothing else in the code changes.

| export id | name | geometry | default tools | export priority |
|---|---|---|---|---|
| 0 | background | — | — | lowest |
| 1 | building | vector | rectangle, polygon | 1 (wins) |
| 2 | track | raster + polyline | brush, polyline-with-width | 2 |
| 3 | woody | raster | brush, wand | 4 |
| 4 | field | vector | polygon, wand | 5 |
| 5 | water | raster | brush, wand | 3 |

Export priority order: `building > track > water > woody > field > background`.

Each entry has: `id, name, color, geometry, tools, priority`.

**A class layer is a group of editable sublayers**, one per source, plus
`manual` for hand-drawn work. Example: `woody/osm`, `woody/landcoverai`,
`woody/worldcover`, `woody/flair`, `woody/manual`.

- Each sublayer is edited on its own: erase the bad parts of one source, fix
  what is left. Each has its own visibility, opacity and lock.
- Sublayer colors are shades of the class color, so it stays clear which class
  a pixel belongs to and which source it came from.
- **Delete sublayer**: drop a source that is useless for this location.
- **Merge sublayers**: fold one into another, e.g. into `manual` once cleaned.
- At export the class = **union of its remaining sublayers**. This happens only
  at export; while editing, sublayers stay separate.
- Provenance falls out of the structure: every pixel or feature in the export
  is attributed to the sublayer it came from.

- **Every class layer can hold both vector features and a raster bitmap.**
  `geometry` only picks the default tools and the form a reference is imported
  in. Example: `track` is brushed, but a polyline with `width_m` is also stored
  as a vector and rasterized at export.
- **Overlaps are kept while editing.** Class layers are independent, so a pixel
  can be `woody` and `field` at once, and you can see and fix the conflict.
  Priority is applied **only at export**.
- An optional "show conflicts" overlay highlights pixels claimed by 2+ classes.

## 4. Reference sources (adapters)

### 4.1 Contract

Adapters run **offline in Python** (`uavloc` env), before annotation. The app
only reads what they produced.

```python
class Adapter:
    name: str                   # "ms-footprints"
    version: str                # bump when output changes
    class_map: dict[str, str]   # source class -> our class
    def run(self, loc: Location, date: str) -> dict[str, Ref]: ...
    # Ref = GeoJSON FeatureCollection (lat/lon) | uint8 mask on the location grid
```

- Registry: one module per adapter, registered by name. **A new source = one new
  file** and nothing else changes.
- CLI: `python -m tools.annotator.refs run --source <name|all> --locations <all|slug...>`.
  It skips runs that are already done unless `--force` is given.
- **Runs on the latest date only** (`reference_date`, i.e. `current`). One
  reference set per location.
- Conversion to our geometry happens here, not in the app:
  - A raster source for a vector class (building, field) → contours → polygons,
    Douglas–Peucker simplify (~0.5 m), with a minimum area. **Kept as polygons**,
    not squared to rectangles.
  - A vector source for a raster class (OSM wood) → rasterized onto the grid.
- Every output carries provenance: adapter name, version, params, run time.

### 4.2 Initial adapters

| adapter | classes it provides | note |
|---|---|---|
| `osm` | building, track, woody, water, field (`landuse=farmland`) | Overpass. Reuse the `map-derived-labels` code. |
| `ms-footprints` | building | best building source (IoU 0.204, 5/9) |
| `landcoverai` | building, track, woody, water | **all classes imported.** Only connected track network. Known to over-claim water on ploughed fields: expect to fix. |
| `flair` | building, woody, water, field (Agricultural + Plowed + Herbaceous?) | native 0.386 m/px, no upsampling. No track class. |
| `worldcover` | woody | 10 m, coarse |

Known weaknesses come from PoC notes. The app does not hide them. The person
decides.

### 4.3 From reference to edit layer

Reference layers are **read-only overlays**, one per (source × class), and each
can be toggled on or off and has its own opacity.

To use a reference:
- **Seed:** each seeded source is copied into **its own sublayer** of the class.
  Sources are never merged at seed time. Done automatically on first open,
  using these defaults:

  | class | default seed sublayers |
  |---|---|
  | building | osm, ms-footprints (2 sublayers); flair, landcoverai as overlays only |
  | track | landcoverai, osm (centrelines with width, editable as polylines) |
  | woody | osm, landcoverai, worldcover, flair (4 sublayers) |
  | water | osm |
  | field | none: only `manual`, drawn by hand (refs still viewable as overlays) |
- **Add source sublayer:** seed another source into a class later (e.g.
  `building/osm`).
- **Adopt selected:** click reference features/blobs → copy just those into a
  chosen sublayer.
- **Re-seed:** reset a sublayer from its source (asks for confirmation; it can
  be undone).

Once seeded, the edit layer does not change when a reference is re-run.

## 5. Multi-date model

### 5.1 Base + per-date override

- **Base**: one annotation per location, shared by all dates.
- **Override** (optional, per date): only where that date differs (a new
  building, a cut forest). Stored as a **delta**:
  - `add`: vector features and raster pixels present on this date only
  - `remove`: base feature ids hidden on this date + a raster erase mask
- Label for date D = `base − remove(D) + add(D)`. Edits to the base therefore
  flow through to every date except where an override removed them.
- The editing target is always visible: **"Editing: BASE"** vs
  **"Editing: 2019-05-15 override"**, with a distinct color.
- **Exclude date:** a flag for unusable dates (clouds, bad co-registration).
  An excluded date is not exported.

### 5.2 Compare views

- Date list: visibility + opacity per date. The top date's opacity is a slider.
- **Swipe:** a vertical divider, date A on the left and B on the right.
- **Blink:** a hotkey flips between two chosen dates.
- The class layers stay on top in all modes.

## 6. Editing tools

| tool | applies to | behaviour |
|---|---|---|
| Select | vectors | click / box select, multi-select, move, delete |
| Transform | vectors | resize, rotate (rectangles), vertex drag/add/delete (polygons) |
| Rectangle | building | rotated rectangle, stored as a 4-point polygon |
| Polygon | building, field | click vertices, close; fallback for non-rectangular houses |
| Polyline + width | track | centerline + `width_m`, editable; rasterized at export |
| Brush / eraser | raster classes | size in px and m, round |
| Layer translate | class or sublayer | drag or numeric dx/dy (px/m); moves vectors **and** raster together, used for global offset fixes |
| Magic wand | any class | see below |

**Magic wand (v1, in browser).** On hover, show a preview of a flood fill from
the cursor pixel: color distance ≤ tolerance, stopped by Canny edges. Click
commits it to the active class. For a vector class, the fill becomes a
simplified polygon. Tolerance and edge sensitivity are sliders, and the preview
updates live.

**v2 (later, pluggable):** SAM point-prompt on the server. The tool interface is
the same as the wand, so it is a new tool rather than a rewrite.

General:
- Undo/redo (per location session), autosave.
- Per-layer visibility, opacity, lock.
- Hotkeys for tools, classes, dates, blink.
- Zoom/pan up to 1:1 and beyond at full resolution.

## 7. Workflow and state

- Location list with a status per location **and** per date:
  `todo / in progress / done / rejected`.
- Single user, local.

Workspace (outside the immutable dataset):

```
data/derived/annotation/
  refs/<source>/<location>/{vectors.geojson, <class>.png, meta.json}
  work/<location>/
    state.json                  status, excluded dates, seed choices, per-date status
    base/vectors.geojson        features: id, class, sublayer, geometry (lat/lon), width_m?
    base/<class>/<sublayer>.png 1-bit raster per sublayer
    overrides/<date>/{add.geojson, add/<class>/<sublayer>.png,
                      remove.json, remove/<class>/<sublayer>.png}
```

Vectors are stored in lat/lon. The app converts them to pixels using the Web
Mercator grid in `meta.json`.

## 8. Export

A new dataset `ua-village-annotated` (recipe + card + manifest, per
`datasets/_template`). `ua-village-multidate` is left untouched.

- Per location: `label.png` if no date has an override, otherwise
  `label-<date>.png` per non-excluded date.
- Raster: rasterize vectors (building polygons, field polygons, track polylines
  at `width_m`), union with the bitmaps, resolve overlaps by priority, write
  class ids 0–5.
- Only `done` locations/dates are exported. The card records the per-class
  source provenance share (how much came from each source, how much was
  hand-drawn).
- Images: referenced from `ua-village-multidate` by the manifest, not copied.

### 8.1 Sample metadata (incremental export)

Every exported sample gets a record in `samples.jsonl` in the dataset root. The
records let the dataset be **extended without rebuilding it**: when new
locations are marked `done`, export adds only those. When something changes,
the records show which samples are stale.

One record per (location, date):

| field | purpose |
|---|---|
| `location`, `date`, `label_file`, `image_file` | what the sample is |
| `done_at`, `exported_at` | when it was approved and when it was written |
| `annotation_hash` | hash of base + that date's override. It changes whenever the annotation is edited |
| `classes_version` + class ids | registry version at export. Adding a class makes old samples detectably "not reviewed for class X" |
| `refs` | adapter name + version + params used for each seed |
| `provenance` | px share per class: from each source / hand-drawn |
| `mpp`, `bounds`, `shape` | georeferencing, copied from `meta.json` |
| `excluded_dates` | dates skipped, and why |

Export modes:
- `export --new`: `done` samples that have no record yet.
- `export --stale`: `done` samples whose `annotation_hash` or `classes_version`
  differs from their record.
- `export --all`: full rebuild.

A sample edited after `done` returns to `in progress` until approved again, so
nothing half-fixed gets exported.

Adding a class means bumping `classes_version`. Existing `done` samples are
then marked **needs review for the new class**. They stay in the dataset, and
the list of them is the to-do.

## 9. Architecture

- **Local server + single-page frontend.** Python FastAPI. It serves images,
  refs and workspace files, saves edits, and runs export. Later it can host SAM
  (v2 tool) on MPS.
  Start: `python -m tools.annotator` → open `localhost`.
- **Frontend:** vanilla JS, no build step, no external libraries. One Canvas2D
  renderer for images, raster sublayers and vectors; the magic wand is pure JS
  in a Web Worker (see §12).
- **Everything lives in the annotator**, reference-source code included:

```
src/tools/annotator/
  SPEC.md
  classes.json
  server/          FastAPI app, workspace I/O, export
  refs/            adapter contract + one module per source
                   CLI: python -m tools.annotator.refs run --source ... --locations ...
  web/             index.html, js/, css/
```

## 10. Open

Nothing. Settled 2026-09-25: "latest date" = `current`; sources stay on
separate sublayers, not merged (§3).

## 11. Milestones

1. Adapter contract + `osm`, `ms-footprints`, `landcoverai`, `flair`,
   `worldcover` → refs for all 59 locations.
2. Viewer: location list, dates, compare modes, ref overlays.
3. Vector editing (building, field) + seeding + save.
4. Raster editing (brush, polyline track, wand) + layer translate.
5. Overrides + exclude date + incremental export with `samples.jsonl`.

## 12. Deviations made while building (2026-09-25)

| spec said | built | why |
|---|---|---|
| Konva for vectors | own Canvas2D renderer | ~2k polygons × 2 sublayers per location would be ~4k Konva nodes; one canvas with viewport culling redraws them per frame without that overhead |
| OpenCV.js for the wand | pure-JS wand (blur → Sobel edge walls → colour flood → edge-band growth) in a Web Worker | no ~9 MB download, works offline, testable in node (`checks/web_selftest.mjs`) |
| track seeded from landcoverai | landcoverai **and** osm | OSM roads arrive as centreline + width polylines, the one form of track that is editable as vectors |
| — | reference vectors outside the image are dropped | OSM is fetched with a ~100 m margin; Birky lost 276 of 1119 buildings that were entirely off-image |
| — | coordinates: pixel *i* covers `[i, i+1]` | canvas/Web Mercator convention; cv2 contours get +0.5, cv2 rasterization −0.5, so what is drawn is what exports |
| — | per-date image alignment (`offset_px`, integer px) | a date whose image sits a few px off: moving the shared layers would break every other date. Display + wand sampling shift the image; export shifts that date's label by (−dx, −dy); the uncovered strip is background and `image_offset_px` is recorded in `samples.jsonl` |
