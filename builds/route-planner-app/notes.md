# route-planner-app

**Status:** ready for your review · **Unblocks:** the PyInstaller `.app` (build step 5 in `docs/route-planner.md`)

## What this is

The PySide6 desktop app of `docs/route-planner.md` step 4: a Leaflet map in a `QWebEngineView` and a native side
panel. It only wires `uavloc.planning`, `uavloc.ardupilot` and `uavloc.onboard` together; everything non-trivial
is in Qt-free modules so it is unit-tested. Launch from `src/`: `python -m tools.route_planner [--settings PATH]`.

## What I needed from you

Nothing. The Birky tiles, class blocks and both `.onnx` files were already cached.

## What I did

Code is in `tools/route_planner/` (not `uavloc/`: it is the app, not shared equipment).

| File | Job |
|---|---|
| `config.py` | `AppConfig`: all paths, from `REPO_ROOT`, overridable by `settings.json` in the app-data dir (`QStandardPaths`) or `--settings` |
| `tile_sources.py` | registry of imagery layers (Esri only) with the Leaflet options |
| `planning_job.py` | `PlannerEngine.plan`: plan, elevation check per route, render both overlays |
| `overlays.py` | class map and blind map as PNG data URLs with bounds |
| `view_model.py` | table rows, map polylines, blind sub-polylines, legend text |
| `export.py` | `export_route` (three ArduPilot files + package zip), `upload_route`, `write_vehicle_params` |
| `map_bridge.py`, `workers.py`, `panel.py`, `main_window.py`, `__main__.py` | Qt wiring only |
| `web/` | `index.html`, `app.js`, vendored Leaflet 1.9.4 (no CDN at run time) |

Decisions made while you slept:

- Clicking sets Start then End; further clicks do nothing. Moving a marker or resetting clears the planned routes
  (they no longer match the points).
- Each export goes into a new folder `route-w<weight>-<UTC stamp>/` under the chosen folder, so exporting two routes
  never overwrites. The package id is `<stamp>-w<weight>`.
- Elevation is advisory: if Open-Meteo fails, planning still succeeds and the row says "elevation check unavailable".
- "Upload mission" uploads only the mission, reads it back and compares (`mission_differences`); "Write parameters"
  is a separate button. Waiting for the vehicle's home takes up to 40 s (SITL sets its home about 20 s after the
  first connection).
- Layers default to satellite on, class map and blind map off.
- Elevation `check_routes` and area-too-large errors reach the user as a message box and a log line.

## Does it work

Birky case (start 49.905, 23.885; end 49.915, 23.925), real cached data, real Open-Meteo, real SITL.

- `tests/route_planner/`: 24 tests (config, overlays, rows/paths, planning stages, failed elevation, area too large,
  full export loaded back with `load_package`, upload report). `python -m pytest tests -q`: 107 passed.
- `figures/09-map-routes.png`: the map page with Esri tiles, two routes, red blind stretches, S/E markers, legend,
  attribution. The route w=2 is selected. `figures/10-map-classes.png`, `figures/10-map-blind.png`: the overlays line
  up with the imagery; the red blind stretches sit on red patches of the heatmap, the 3 % blind route runs through
  the green ones.
- `figures/01..08-*.png`: the side panel after each step of `smoke.py --no-map` (points, plan, layers, selection,
  export, upload): two routes in the table, the elevation warning for route w=2, the exported paths, and
  "49 mission items uploaded / vehicle home is 0.0 m from the planned start / read back identical" from a live SITL.
- Marker drag, map click and route click reach Python (checked in headless Chrome against a stand-in
  channel: `mapClicked`, `markerMoved`, `routeSelected` arrive with the right arguments).

**Not verified:** the app inside a real `QWebEngineView`. QtWebEngine's renderer cannot start in the agent sandbox
(`mach_msg ... msg too large`), so the map page and the Python side were each verified on their own: `app.js` in
headless Chrome with a stand-in `QWebChannel` replaying the real plan (`harness/`), and the Python side with
`smoke.py --no-map`. What that leaves open: that `qrc:///qtwebchannel/qwebchannel.js` loads from the `file://` page,
and that Qt delivers the JSON commands and the ~1 MB overlay data URLs. `smoke.py` without `--no-map` drives the whole
window and takes the screenshots; run it once on your Mac.

Also unverified: a real flight controller over serial; Windows.

Found in another module, not mine to edit: `uavloc/onboard/package_writer.py` uses `hashlib` in `make_mission_id`
without importing it, so `write_package` without an explicit `mission_id` raises `NameError`. The app passes its own id.

## Your call

1. Run `PYTHONPATH=. python builds/route-planner-app/smoke.py` once with a window to see the real map.
2. Both routes on this case are flagged "blind limit" (limit 40 m; longest stretches 139 m and 45 m) and
   "flight time" (22 and 23 min against 16 min: 20 min minus the 20 % reserve). Those are the planner's numbers, not
   the app's, but the table shows them first.
3. Packaging needs: `web/` (including `vendor/`), `uavloc/ardupilot/lua/vision-gps-switch.lua`, the two `.onnx` files
   (or paths in `settings.json`), QtWebEngine's `QtWebEngineProcess` helper and resources, and the transitive
   `segmentation-models-pytorch` stack listed in `requirements-planner.txt`.

## Fix 2026-11-02: blank map in the installed app

In the real QtWebEngine (installed `.app`), none of the page's local `file://` assets loaded (Leaflet, CSS,
`app.js`, and the `qrc:` qwebchannel.js), so the map was empty with no console error. The headless-Chrome
check above could not catch it. Now `qwebchannel.js` is injected as a `QWebEngineScript` and the page is
served by `setHtml` with every local asset inlined (`tools/route_planner/page.py`). The app writes
`~/Library/Application Support/uav-localization/route-planner/route-planner.log`. After the fix the log shows
the QWebChannel connected and 25/25 Esri tiles loaded in the installed app.
