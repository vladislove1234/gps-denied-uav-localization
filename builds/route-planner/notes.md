# route-planner

**Status:** ready for your review · **Unblocks:** route planner steps 2–5 (`docs/route-planner.md`)

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

Step 1 of the route planner ([docs/route-planner.md](../../docs/route-planner.md)): given a start, an
end and a mission altitude, it proposes routes over terrain where the onboard matcher is likely to
get a fix. This is the planning core the Qt app will call; the ArduPilot, onboard and app steps follow.

## What I needed from you

Nothing. Esri tiles are fetched on demand; the satellite model is the existing 007 LR-ASPP export.

## What I did

| Module | What |
|---|---|
| `uavloc/geo.py` | `GeoBounds` (lat/lon box, `around(points, margin_m)`). |
| `uavloc/data/tiles.py` | Promoted from `pocs/map-derived-labels/tiles.py` (second use). Tile sources (`ESRI_WORLD_IMAGERY`), cached fetch into the existing `data/derived/tile-cache/esri-current/`, mosaics, and `mercator_to_geoimage`, which resamples rows to be linear in latitude so the result is an exact `GeoImage`. |
| `uavloc/data/satellite.py` | `GeoImage.window`: a cheap axis-aligned view, so patches are warped from a small window, not the whole many-channel corridor. |
| `uavloc/features/tiling.py` | `tiled_predict` takes any numpy batch function; `tiled_logits` is now a torch wrapper around it (same numbers). `TILE_PX`/`TILE_STEP_PX` moved here. |
| `uavloc/features/classes.py` | `CLASSES` without the torch/smp import; `segmentation.py` re-exports it. |
| `uavloc/features/onnx_segmenter.py` | `OnnxSegmenter`: onnxruntime class probabilities, tiled at the model's fixed 512 px input. |
| `uavloc/matching/correlator.py` | `patch_size_for_footprint_m`, `half_range_for_footprint_m`, `PositionCorrelator.surface_from_map`; the old methods delegate to them, no behaviour change. |
| `uavloc/planning/config.py` | `MissionInputs`, `PlannerConfig`, `PLANNING_MATCHER` (007.1 settings, T 0.5). |
| `uavloc/planning/classmap.py` | Corridor class map: 4×4-tile blocks segmented with 1 tile of context, cached as uint8 `.npz` per (source, model hash, block) under `data/derived/planner-cache/class-maps/`. |
| `uavloc/planning/localizability.py` | `SelfSimilarity` (map patch vs its surroundings, batched FFT weighted NCC, posterior long-axis std + coverage → blind) and the `ClassMix` fallback, behind `LocalizabilityModel`. |
| `uavloc/planning/drift.py` | INS drift sigma under `InsConfig`, and the blind-stretch limit. |
| `uavloc/planning/grid.py` | `CostGrid` and 8-connected A*. |
| `uavloc/planning/routes.py` | Straightening, blind stretches, flight time, near-duplicate removal, `plan_routes`. |
| `uavloc/planning/elevation.py` | Open-Meteo DEM profile and the clearance / footprint-change check. |
| `uavloc/planning/planner.py` | `plan()`, with the localizability map cached per area, altitude and settings. |
| `uavloc/plotting/planning.py` | Rasters in lon/lat with routes and blind stretches. |
| `tests/planning/test_planning.py` | 13 tests: tile maths, Mercator row resampling, A* (blind wall, blocked cells), straightening, stretches, drift limit, duplicates, elevation warnings. |

## Does it work

Run: `python -m papermill builds/route-planner/verify.ipynb builds/route-planner/runs/verify.ipynb --cwd . -k python3`
from `src/` in the `uavloc` env. Numbers are from that run's outputs (2026-11-02).

Case: a 3.08 km start–end across Birky, 40 m, 5 m/s, 20 min battery.

- `figures/routes-birky.png`: Esri, class map, blind map, two routes. The shortest route (w = 0, 3.17 km)
  is 30% blind, with a longest blind stretch of 139 m across the open field to the southwest. The weighted
  route (w = 2, 3.36 km, 6% longer) follows forest edges and village streets: 3% blind, longest stretch
  45 m. Weights 5 and 20 gave the same route as w = 2 and were dropped as duplicates.
- `figures/localizability-scores.png`: s1 sits mostly at 2–7 m, with a spike at ~8.2 m. That spike is
  the std of a uniform posterior over the 1.7× window, i.e. a flat surface. About a third of the
  points have coverage < 0.01 (fields).
- `figures/elevation-birky.png`: ground rises ~17–19 m along both routes. Height above ground drops to
  ~28 m, a ~30% footprint change, so both routes get a warning.
- **Time:** class map ~3 min the first time (77 blocks), then cached. Localizability 115 s for the
  8.2 km² area at 25 m spacing, after switching to batched FFT NCC; it was 14 min with the conv NCC
  (bug-143). The FFT and conv heatmaps agree to 3e-4 in z.
- Unit tests: 13 passed.

**What is still wrong or unverified:**

- **The blind limit is 64 m.** It comes from `InsConfig` defaults (0.2 m/s velocity bias, 2σ)
  against the 007.1 window's short half-range (7.8 m at 40 m). At this limit almost any route that
  crosses a field is flagged. The SITL coasting measurement (step 2/3) and a real INS model are what
  should set it.
- **Localizability depends on the temperature.** In the Birky village core, T = 0.25 / 0.5 / 1.0 gives
  22% / 34% / 80% blind. T = 0.5 sits inside 007's tuned range (0.25–2), but nothing here calibrates
  it against real fix success.
- **Self-similarity ranks terrain; it doesn't predict fix rates.** It has no stale map, no season
  change and no drone-side error. It is also scored north-up.
- Both routes exceed the 16 min budget (20 min minus a 20% reserve): 6+ km out and back at 5 m/s
  takes ~22 min. That is the check doing its job, not a bug.
- The w = 0 route has 4 legs, not 1: a straight shortcut would lengthen its longest blind stretch,
  and the straightening rule refuses that.

## Your call

1. The blind-stretch limit: keep the drift-based default (64 m) until SITL gives the coasting time, or
   set a working value by hand for now?
2. Temperature / s_max for the planner: keep T = 0.5, s_max = 5 m, or calibrate against a bench first?
3. The app will need torch CPU after all: localizability runs the correlator, which is torch.
   Segmentation still runs on onnxruntime, as agreed.
