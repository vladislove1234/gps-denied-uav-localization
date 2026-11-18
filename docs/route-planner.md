# Route planner

The design for the route planner, as agreed in the design interview on 2026-11-02.
A laptop app plans a route over terrain where our matcher can hold a fix, sends
the route to ArduPilot as a mission, and exports a package for the Raspberry Pi
that runs localization onboard. The onboard localizer is not finished, so the
contracts it will plug into are defined now and exercised with a stub.

Related: [ardupilot-integration.md](ardupilot-integration.md) covers the Pi ↔ flight
controller link, `GPS_INPUT` and the Lua switch. This document reuses those choices.

## Scope of v1

v1 is done when the whole chain runs in ArduPilot SITL on the Mac:

1. Plan a route in the app.
2. Upload it to the simulated drone, directly or as a `.waypoints` file.
3. Export the Pi mission package.
4. Run the onboard runner with the **stub localizer**. The stub sends the simulator's
   true position plus noise as `GPS_INPUT`, and sends nothing over the stretches
   the localizability map marks as blind.
5. The simulated drone flies the mission out and back on those fixes.

That run also measures how long ArduPilot can coast on its IMU without fixes,
which feeds the blind-stretch limit.

Not in v1: waypoint editing, via-points, no-fly polygons, offline field mode,
a Windows installer, terrain following, a "Send to Pi" button, and the vision
(`VISION_POSITION_ESTIMATE`) output.

## Decisions

| # | Decision |
|---|---|
| Map | Esri World Imagery or Google satellite (Map Tiles API), picked under *Layers → Imagery*. The chosen source also feeds the class map and the export; switching clears the current plan. Google needs `GOOGLE_MAPS_API_KEY` in the environment or in `src/.env` (the packaged app: `~/Library/Application Support/uav-localization/route-planner/.env`); its tiles cache under `tile-cache/google-satellite/`. `tile_source_key` in `settings.json` sets the startup source. OSM or own GeoTIFFs can be added to `TILE_SOURCES` the same way. |
| Class map | On demand: Esri z18 tiles for the corridor are downloaded, segmented on the laptop with onnxruntime using the **same `.onnx` the Pi runs**, and cached per block of tiles. The class map the planner routes on is the reference map shipped to the Pi. |
| Cost | Distance plus a localizability penalty. Hard-blocked classes are optional and off by default. |
| Localizability | Self-similarity with the real `PositionCorrelator`: at grid points, a footprint-sized patch of the class map is matched against its own surroundings. The posterior's long-axis std and the weighted-class coverage decide "fix likely" vs "blind", with the onboard gate's thresholds. A class-mix heuristic is the fallback, behind the same interface. |
| Altitude | Set per mission before routing (default 40 m), relative to takeoff. The localizability map is cached per altitude, because the footprint depends on it. |
| Routes | A* on a ~10 m grid at several distance/localizability weights; near-duplicates are dropped, leaving 2–4 routes. |
| Straightening | The cell path is turned into a few long legs. A shortcut is accepted only if it doesn't make the cost or the longest blind stretch noticeably worse. |
| Blind-stretch limit | A maximum distance without a likely fix. Routes over it are flagged red, never hidden. Default = the smaller of (a) the distance after which expected INS drift outgrows the matcher's search window and (b) the IMU coasting time measured in SITL × speed. |
| Per-mission inputs | Altitude (40 m), speed (5 m/s), max flight time, hover time at the end point (0 s). |
| Mission | ArduCopter only. Out and back: takeoff at the start, fly to the end point, optional hover, return along the same route reversed, land at the start. |
| Failsafes | Low battery and lost radio → SmartRTL (flies back along the path actually flown), falling back to LAND. Lost position → LAND. |
| Into ArduPilot | Both a `.waypoints` (QGC WPL 110) export and a direct "Upload to drone" over MAVLink. The app warns if the vehicle's home is more than 20 m from the planned start. |
| Parameters | One `.param` file plus the Lua script `vision-gps-switch.lua`. A transmitter switch chooses between shadow mode (fly on real GPS, ours only logged) and active mode (fly on ours). |
| Position output | `GPS_INPUT` as GPS2, horizontal only (altitude ignored). Height comes from ArduPilot (GPS + baro) behind a replaceable height-source interface, ready for a rangefinder later. |
| App | PySide6 desktop app, Leaflet map in a `QWebEngineView`, Python ↔ JS over `QWebChannel`. Targets macOS and Windows; v1 ships an unsigned PyInstaller `.app` only. |
| Map layers | Satellite, class map, localizability heatmap, routes with blind stretches highlighted. |
| Onboard | Raspberry Pi 5, 64-bit Raspberry Pi OS, UART to TELEM2 at 921600 baud (`/dev/ttyAMA0`), install script + systemd service. In SITL the same runner connects over UDP. |
| Onboard parts | Package loader, camera source, flight state source, localizer, position sink, logger. Each is a `Protocol` with frozen-dataclass messages, run in one loop. v1 fakes the camera and the localizer. |
| Pi software vs data | The onboard software is installed once. Each mission is a versioned, data-only package that you copy over by hand. |
| Logging onboard | Every fix or "no fix" with its reason, flight state as JSON lines, every 5th frame as JPEG, a copy of the manifest. |

## Code layout

| Path | What |
|---|---|
| `uavloc/data/tiles.py` | Web Mercator tile maths, tile sources (`ESRI_WORLD_IMAGERY`), cached fetch, mosaic → `GeoImage`. Promoted from `pocs/map-derived-labels/tiles.py`. |
| `uavloc/features/onnx_segmenter.py` | `OnnxSegmenter`: onnxruntime class probabilities, tiled like `class_probabilities`. |
| `uavloc/planning/` | Planning area, corridor class map, localizability, cost grid, A*, route alternatives, straightening, route checks, elevation. |
| `uavloc/ardupilot/` | `.waypoints`, `.param`, Lua script, mission upload, `GPS_INPUT` bridge. |
| `uavloc/onboard/` | Contracts, package loader and writer, stub localizer, runner. |
| `tools/route_planner/` | The Qt app. It only calls the modules above. |
| `builds/route-planner/` | Build record, verification notebook, figures. |

## Planning pipeline

1. **Area.** The start–end bounding box plus a margin, capped in area.
2. **Imagery.** Esri z18 tiles for the area (cache: `data/derived/tile-cache/esri-current/`).
3. **Class map.** Blocks of 4 × 4 tiles are segmented with one tile of context on each
   side and cached per (model, block). The stitched map is resampled to rows linear in
   latitude, so it is an exact `GeoImage` (Web Mercator rows are not linear in latitude:
   over 10 km that is metres of error).
4. **Localizability.** At grid points every `sample_spacing_m`, the footprint at the
   mission altitude (north-up) is cut from the class map and correlated against the
   search patch around it with the onboard matcher's settings. Per point:
   the posterior long-axis std `s1_m` and the weighted coverage. "Blind" means
   `s1_m > s_max_m` or `coverage < c_min`, the onboard gate's rule.
5. **Cost grid.** Blind share resampled to `cell_m` cells. Step cost = step length ×
   (1 + weight × blind share). Optional hard-blocked classes make cells impassable.
6. **A\*.** 8-connected, Euclidean heuristic (admissible: no step costs less than its
   length). Run once per weight in `route_weights`.
7. **Straighten** each path, then **drop near-duplicates**.
8. **Checks per route.** One-way length, out-and-back flight time against the battery,
   blind share, blind stretches and the longest one against the limit, and an
   elevation profile (Open-Meteo DEM) against the takeoff ground level.

### Known approximations

- Self-similarity compares the map with itself: no season change, no stale map, no
  drone-side segmentation error. It ranks terrain; it does not predict real fix rates.
- Localizability is scored north-up. The drone's heading along a leg changes which
  axis of a road ridge is constrained, but not whether there is structure.
- The INS term of the blind limit comes from the simulator's `InsConfig`, not from a
  measured flight controller. The SITL coasting time replaces or tightens it.

## Mission package (to the Pi)

A versioned folder, zipped for transfer:

```
mission-<id>/
  manifest.json              format version, mission id, created, inputs, waypoints,
                             blind stretches, route stats, sha256 per file
  reference/classes.tif      class probabilities, uint8, corridor around the route
  reference/satellite.jpg    RGB crop of the same corridor (debugging, plots)
  models/drone.onnx          drone-side segmenter
  config/camera.json         FOV, resolution
  config/localizer.json      matcher settings
  ardupilot/mission.waypoints
  ardupilot/uavloc.param
  ardupilot/vision-gps-switch.lua
```

The corridor half-width is set from the blind limit: the worst expected drift plus
half the footprint plus the search margin.

Licensing flag: Esri's terms restrict storing imagery offline, and the package does.
For thesis research that is the user's call; a source licensed for offline use can
replace Esri behind the tile-source interface.

## Build order

1. `uavloc/planning` with unit tests and a real-data verification (`builds/route-planner/`).
2. `uavloc/ardupilot`, tested against SITL (installed natively on macOS).
3. `uavloc/onboard`: contracts, package, stub, runner. The v1 milestone is the full chain in SITL.
4. The Qt app in `tools/route_planner/`.
5. The PyInstaller `.app`.

## Status (2026-11-02)

All five steps are built. Evidence and open items per step:

| Step | Record | Verified | Not verified |
|---|---|---|---|
| 1 Planning | `builds/route-planner/notes.md` | Birky 3.1 km case: 2 routes, the weighted one 3% blind | Localizability calibration (temperature, s_max) against real fixes |
| 2 ArduPilot | `builds/ardupilot-mission/notes.md` | SITL (ArduPilot 4.8-dev, `~/Documents/Projects/ardupilot`): params and mission round trip, AUTO flight out and back, GPS2 via GPS_INPUT, Lua switch. IMU coasting is 7.5–8.1 s before LAND, so the default `ekf_coast_s` is 8 s | Mission Planner/QGC loading the files, a real flight controller |
| 3 Onboard | `builds/onboard-runner/notes.md`, `builds/sitl-v1/notes.md` | **v1 milestone:** the runner process (`python -m uavloc.onboard --stub`) flew the route out and back in SITL with GPS1 disabled. With planned blind drops, the EKF failsafe landed the drone at the 44.8 m stretch, over the 40 m limit: the planner's flag confirmed | Pi install script, real latency, wind, real-time speed |
| 4 App | `builds/route-planner-app/notes.md` | Qt-free logic tested (24 tests); the map page renders in headless Chrome; mission upload to SITL | The app in a real QtWebEngine window: the agent sandbox can't start it, so run `builds/route-planner-app/smoke.py` once |
| 5 .app | `builds/route-planner-app/packaging.md` | `tools/route_planner/packaging/build_macos.sh` → `dist/Route Planner.app` (1.14 GiB, unsigned, ad-hoc signed); frozen self-test and frozen Birky plan pass | GUI rendering in the bundle, Gatekeeper on another Mac |

Findings that change the design numbers:
- The blind-stretch limit is set by the EKF, not the matcher: 8 s of coasting is 40 m at 5 m/s.
- The GPS2 rate at arming in AUTO must be about 5 Hz of simulated time; after arming, 2 Hz is enough.
- Routes that start inside a blind stretch can't arm in AUTO. The SITL sequence arms in ALT_HOLD first.
- GPS_INPUT must carry the AMSL altitude: arming compares both GPSes in 3D.
