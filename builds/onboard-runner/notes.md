# onboard-runner

**Status:** ready for your review · **Unblocks:** the SITL end-to-end flight (`docs/route-planner.md`, step 3 milestone)

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

The Pi side of the route planner ([docs/route-planner.md](../../docs/route-planner.md)): the mission package that
carries a route's map to the drone, and the loop that runs on the Pi. The loop reads ArduPilot's state, asks a
localizer for a fix, carries the fix forward to send time, and sends it as `GPS_INPUT` (GPS 2). v1 plugs in a stub
localizer (simulator truth plus noise, no fix over blind stretches), because the real one is not finished. The
contracts it will plug into are fixed now.

## What I needed from you

Nothing.

## What I did

| Module | What |
|---|---|
| `uavloc/onboard/contracts.py` | `Frame`, `FlightState`, `Fix`, `ReferenceMap`; Protocols `CameraSource`, `FlightStateSource`, `HeightSource`, `Localizer`, `PositionSink`, `MotionSource`, `TruthSource`, `FlightLogger`. |
| `uavloc/onboard/package.py` | The reading side: `MissionPackage`, `load_package`, `newest_package`, sha256 and format-version checks. |
| `uavloc/onboard/package_writer.py` | The laptop side: `PackageRequest`, `write_package`. Split from `package.py` so the Pi never imports the planner (and with it torch). |
| `uavloc/onboard/corridor.py`, `raster.py` | Corridor width and mask, crop to bounds, GeoTIFF read/write. |
| `uavloc/onboard/mavlink_sources.py` | `MavlinkRouter` (one reader per connection), `MavlinkFlightState` (2 s history, interpolation, `velocity_displacement_m`), `ArdupilotHeight`, `SimTruth`, `connect`. |
| `uavloc/onboard/stub.py`, `sinks.py`, `logger.py`, `cameras.py` | `StubLocalizer`, `GpsInputSink`, `JsonlFlightLogger`, `NullCamera`. |
| `uavloc/onboard/runner.py`, `cli.py`, `__main__.py` | `OnboardRunner` and `python -m uavloc.onboard`. |
| `uavloc/onboard/replay.py` | Flies a package's route in simulated time through the real runner (used here to see where fixes drop). |
| `tools/onboard/` | `install-pi.sh` and `uavloc-onboard.service`. |
| `tests/onboard/` | 49 tests (48 functions) with fakes, no SITL: package round trip, tamper detection, stub, sink, sources, runner, logger, CLI against a fake flight controller. |

Decisions made without asking (you were asleep):

- **Corridor half-width** = `drift_sigmas` x INS drift after the blind limit + half the footprint diagonal + the
  matcher's larger search half-range. Pixels outside the strip are blanked (classes 0, blind 1, satellite black), so
  the bounding box of a diagonal route compresses to the strip. Outside the strip the stub, like the real localizer,
  has no map.
- **`is_blind`** interpolates the 25 m blind map bilinearly and cuts at 0.5, the way the planner's cost grid does.
  Nearest-cell lookup dropped fixes at single blind cells that the route's stretches ignore.
- **No torch on the Pi.** The onboard path imports numpy, cv2, PIL, rasterio and pymavlink only. `uavloc.ardupilot.link`
  imports the mission planner (torch), so `mavlink_sources.connect` repeats its three lines.
- `Fix.time_s` is the capture time on the flight-controller clock. The runner swaps `HeightSource` height into
  `FlightState.altitude_m` before calling the localizer. A localizer that raises costs one step a fix
  (`reason="error: ..."`), not the flight.
- The stub's `noise_m` is the std per axis (radial mean about 1.25x), and `latency_s` makes the fix describe an
  earlier moment so the sink's carry-forward has something to correct.

## Does it work

Run: `python -m papermill builds/onboard-runner/verify.ipynb builds/onboard-runner/runs/verify.ipynb --cwd . -k python3`
from `src/` in the `uavloc` env. Numbers are from that run's outputs (2026-11-02).

Case: the Birky route from the route-planner build, lowest-blind route (w = 2: 3.36 km one way, 3.1% blind,
longest blind stretch 45 m, blind limit 64 m). ArduPilot files written by the real `uavloc.ardupilot` module.

- Package: **15.0 MB zip** (16.4 MB folder). `models/drone.onnx` 12.9 MB, `reference/classes.tif` 2.45 MB,
  `reference/satellite.jpg` 1.09 MB, `reference/blind.tif` 2 KB.
- Corridor: **half-width 48.1 m** = drift 7.8 + footprint half-diagonal 26.5 + search half-range 13.8. The bounding
  box of the diagonal route is 2980 x 1274 m (7748 x 3313 px) against 11264 x 7168 px for the planning area.
- Replay of the out-and-back route (6.73 km, 5 m/s, a step every 0.5 s, stub noise 6 m per axis, 2693 steps):
  **3.0% of fixes dropped** (81 steps), all `blind`; longest dropped run 40 m (planner: 45 m); planner one-way
  blind share 3.1%. Mean error of the sent fixes 7.55 m (6 m per axis gives 7.5 m radial).
- `figures/package-corridor.png`: class map of the planning area with the package extent, then the package's
  classes and blind rasters with the route. The strip follows the route and is blank elsewhere.
- `figures/package-zoom.png`: satellite, classes and blind map in a 300 m window around the longest blind stretch.
  The stretch crosses a road next to a building block, and the map in the package matches the imagery.
- `figures/replay-dropped-fixes.png`: dropped fixes against the planner's blind stretches, out and back. They line up.
- Unit tests: `python -m pytest tests -q` passes.

**What is still wrong or unverified:**

- **Nothing here has run against ArduPilot.** The MAVLink side is tested with a fake connection. Not yet seen
  working in SITL: `SET_MESSAGE_INTERVAL` for `SIMSTATE` (the stub's truth), `GPS_INPUT` accepted as GPS 2, and the
  latency carry-forward on a real vehicle. That is the integration run's job.
- **`install-pi.sh` and the service were never run.** There is no Pi here and no shellcheck; the script passes
  `bash -n`. The UART lines (`dtparam=uart0`, `raspi-config nonint do_serial_*`, `cmdline.txt` edit) follow the
  doc and standard Raspberry Pi OS Bookworm practice but are untested on a Pi 5.
- **No real localizer and no real camera.** The service starts only once `UAVLOC_ARGS` in `/etc/default/uavloc-onboard`
  names one (`--stub` for bench tests); without it the CLI exits 4 and systemd does not retry.
- **GPS_INPUT altitude is 0.** The doc wants the AMSL altitude from `GLOBAL_POSITION_INT`; our sink does not pass it
  (position is horizontal only in v1, and `FlightState` has no AMSL field).
- **Fix uncertainty is not grown by the carry-forward.** Velocity error over 0.5 s is small next to the 3 m floor.
- The package includes Esri imagery; see the licensing flag in `docs/route-planner.md`.

## Your call

1. Whether the corridor should be wider than the formula for routes that exceed the blind limit (a longer blind stretch
   means more drift than the formula covers). Today a route over the limit still gets the limit's width.
2. Whether the Pi should keep the `--stub` fallback in the shipped service defaults, or only for bench use as now.
