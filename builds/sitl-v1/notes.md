# sitl-v1

**Status:** ready for your review · **Unblocks:** the v1 milestone of `docs/route-planner.md` (the Qt app's "Upload to drone" and the PyInstaller `.app` now have a flown reference chain)

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

The v1 milestone of the route planner: the whole chain in ArduPilot SITL on this Mac. Plan a route, write the
`.waypoints` / `.param` / Lua files and the Pi mission package, load the package, upload the mission to the simulated
drone, start the onboard runner as its own process exactly as on the Pi (`python -m uavloc.onboard --package ... --stub`),
and let the drone fly the mission out and back on the runner's fixes. The stub sends the simulator's truth plus noise
and nothing over the stretches the package marks as blind.

## What I needed from you

Nothing.

## What I did

| Where | What |
|---|---|
| `uavloc/ardupilot/chain.py` | `RunnerConfig`, `ChainScenario`, `running_runner` (the runner as a subprocess, stopped with SIGTERM), `fly_chain`, `run_chain`, `probe_arming`. |
| `uavloc/ardupilot/chain_analysis.py` | Reads a flight back: the runner's `fixes.jsonl`, `blind_runs` (EKF error and failsafe per stretch without fixes), `summarise_flight`. |
| `uavloc/ardupilot/sitl.py` | `telem2_url()`, `SITL_TELEM2_PORT`. |
| `tests/integration/` | A smoke test of the whole chain (skipped when SITL is not built, waits up to 120 s for port 5760) and unit tests for what needs no simulator. |
| `verify.ipynb` | The chain for three flights and the arming probes. Run: `python -m papermill builds/sitl-v1/verify.ipynb builds/sitl-v1/runs/verify.ipynb --cwd . -k python3` (9 min). Numbers below are from `results.json`. |

**Fixes in the finished builders' code** (public signatures kept; every new parameter has a default):

- `uavloc/onboard/sinks.py`, `contracts.py`, `mavlink_sources.py`: `GpsInputSink(altitude=)` sends the AMSL altitude. `MavlinkFlightState.altitude_amsl_m()` reads it from `GLOBAL_POSITION_INT.alt`. A fix is dropped (counted as stale) until the first altitude has arrived, never sent with 0.
- `uavloc/onboard/cli.py`: `--speedup N` runs the period and all clocks in simulated seconds (without it, a runner at speedup 8 sends 1/8 of the fixes per simulated second and mis-extrapolates the flight-controller clock); `--no-blind` makes the stub answer over blind stretches too; exit code 5 for a bad `--speedup`.
- `uavloc/onboard/stub.py`: `StubLocalizer(ignore_blind=)` behind `--no-blind`.

**Two MAVLink clients.** SITL serves one TCP client per serial port. The test harness is on `tcp:127.0.0.1:5760` (SERIAL0),
the runner on `tcp:127.0.0.1:5763` (SERIAL2, which `vehicle_params()` sets to MAVLink2 and which stands in for TELEM2). No
router and no extra arducopter flags were needed; `5762` (SERIAL1) is open too. The runner asks for its own message
rates with `SET_MESSAGE_INTERVAL` per port.

**Flight sequence** (`fly_chain`): wait for the EKF's absolute position, arm in ALT_HOLD (Copter skips the GPS arming checks
there), switch to AUTO (the mission's takeoff item takes off, `AUTO_OPTIONS=3`), flying on the simulated real GPS 1
with ours only logged (shadow). Once five consecutive `GPS2_RAW` fixes are within 15 m of the truth, flip RC9 (the Lua
switch, `GPS_PRIMARY` 1 and EK3 source set 2), fly 10 s with both, then set `SIM_GPS1_ENABLE=0`. From there the vehicle flies on our
fixes only. The flight ends at touchdown (height under 0.5 m) because a vehicle that lost position can stay armed.

Three flights, each a fresh SITL at speedup 8, runner period 0.5 s of simulated time, 6 m noise per axis:

- **a-planned-blind**: the real planned route, the stub drops fixes over blind stretches, EKF failsafe on (`FS_EKF_ACTION` 1, Land).
- **b-no-blind**: the same route, `--no-blind`, failsafe on.
- **c-planned-blind-failsafe-off**: as (a) but `FS_EKF_ACTION` 0, to watch the EKF through every stretch.

## Does it work

Route: the lowest-blind route of the Birky pair (weight 2): 3364 m one way, 3.1 % blind, 24 waypoints, 4 blind stretches
(15.5, 30.7, 38.5, 44.8 m). The planner's blind limit is now 40.0 m (`ekf_coast_s` 8 s at 5 m/s), so the route is flagged as exceeding it by the 44.8 m
stretch. Package 14.8 MB. The vehicle's home is 0.007 m from the planned start. The mission downloaded from the vehicle
differs from the uploaded one in no item in all three flights.

| Flight | Touched down | From start | Closest to end point | EKF vs truth after GPS 1 off (AUTO): max / mean / p95 | Fixes sent / dropped | Runner exit |
|---|---|---|---|---|---|---|
| a-planned-blind | after 447.8 s, in LAND | 1584.3 m | 1512.4 m | 8.6 / 3.08 / 5.7 m | 804 / 96 | 0 |
| b-no-blind | after 1510.4 s, mission complete | 8.9 m | 5.5 m | 10.6 / 3.07 / 5.9 m | 3022 / 0 | 0 |
| c-planned-blind-failsafe-off | after 1516.6 s, mission complete | 11.2 m | 3.5 m | 20.7 / 3.14 / 6.0 m | 2857 / 176 | 0 |

**v1 is met by flight (b)**: the whole chain runs, the drone takes off on the simulated GPS, switches to our fixes, the
simulated GPS 1 is switched off, and it flies out to 5.5 m of the end point and back to touch down 8.9 m from the start
(the stub's fix noise is 6 m per axis; the EKF error stays under 10.6 m). Flight (c) completes it too with failsafe off.
**Flight (a) does not complete the mission, and that is the finding:** the route's longest stretch is over the blind limit.

**Flight (a), stretch by stretch** (`figures/error-vs-time.png`, red bands; gap = last fix sent to next fix sent):

| Stretch (time of last fix) | Gap | Flown | Fixes dropped | EKF error peak | EKF error at next fix | Position lost after | Failsafe after |
|---|---|---|---|---|---|---|---|
| 107.0 s (the 30.7 m stretch) | 6.1 s | 27.0 m | 11 | 5.4 m | 2.3 m | not lost | none |
| 371.4 s (the 38.5 m stretch) | 6.0 s | 33.4 m | 11 | 6.7 m | 6.7 m | not lost | none |
| 443.8 s (the 44.8 m stretch) | 8.7 s | 39.0 m | 16 | 9.9 m | 7.1 m | 7.4 s | 8.2 s: **EKF failsafe, LAND** |

The vehicle landed in place, 1584 m from the start. The first two stretches were flown through with the EKF error under 7 m.
The third, over the limit, lost the EKF's absolute position 7.4 s after the last fix, and the failsafe fired 8.2 s after it;
this matches the 7.5-8.1 s measured in `builds/ardupilot-mission/results.json`.

**Flight (c), failsafe off.** The same third stretch (last fix 443.5 s): gap 9.0 s, 41.4 m flown, 17 fixes dropped, error peak
20.7 m, position lost 7.6 s after the last fix, "EKF Failsafe" reported at 451.5 s and cleared at 453.4 s with no mode change, and 3.3 m error at the
next fix (GPS 2 re-detected, the EKF reset to it). Other stretches in the log: 33.7 m / 6.5 s (peak 10.6 m), 32.9 m / 7.5 s (peak 10.3 m),
30.5 m / 6.5 s (5.5 m), 27.4 m / 6.5 s (6.3 m), and several runs shorter than 7 m where one or two fixes drop at the edge of a stretch.
Over the last blind stretch, the landing at the start, the EKF error reached 15.3 m (position lost 7.6 s after the last fix,
"EKF Failsafe" at 1563.0 s) before touchdown. Full table: `results.json`, `flights.c-planned-blind-failsafe-off.blind_runs`.

**GPS 2 rate needed.** Arming in AUTO checks that GPS 2 is healthy (average interval under 215 ms) and has a 3D fix
(`arming` in `results.json`, runner period in simulated seconds):

| Runner period | Fixes on the ground | Armed in AUTO | Vehicle said |
|---|---|---|---|
| 0.5 s (2 Hz) | yes | no | `Arm: GPS 2: not healthy` |
| 0.25 s (4 Hz) | yes | yes | |
| 0.2 s (5 Hz) | yes | yes | |
| 0.1 s (10 Hz) | yes | yes | |
| 0.1 s (10 Hz) | no | no | `Arm: GPS 2: Bad fix` |

So arming in AUTO needs about 5 Hz, and 4 Hz sits on the limit (one earlier, unrecorded attempt was refused at 0.25 s; the recorded one armed).
Two further facts make the doc's arm-in-ALT_HOLD design necessary, not just convenient: the planned route's first
15.5 m are a blind stretch, so the stub sends no fix at takeoff, and the runner must not resend. After arming, **2 Hz is enough**: flight (b) flew 1510 s at 2.00 Hz
(0.5 s period) and flight (a) and (c) at 1.80-1.88 Hz average (fixes dropped over stretches). The runner never resends a fix; no higher rate was needed in flight.
AUTO is refused (`requires position`) until the EKF has an absolute position, so the sequence waits for it first.

- `figures/track-over-route.png`: truth and EKF lie on the planned out-and-back path in all three flights. Red marks are the planned stretches; red dots, the fixes the runner dropped. (a) ends at the third stretch.
- `figures/error-vs-time.png`: EKF error vs truth, stretches without fixes shaded, switch and GPS 1 off marked, LAND marked in (a).

**Not verified, or wrong:**

- The stub sends the truth with zero latency (`--latency-s 0`); the 0.4-0.55 s localizer latency and the carry-forward on a real vehicle are not tested.
- SITL speedup is 8, not real time; the runner runs on simulated-time clocks (`--speedup`). Wall-clock jitter of a real Pi at 1x is not tested.
- One seed, one route, one run per flight. The stretch outcomes differ between runs (an earlier unrecorded run of (c) landed several times farther from the start after the same last-stretch error growth).
- No wind, no vibration, simulated IMU noise: the 7.4-8.2 s coast is a lower bound for a real airframe.
- The 5 Hz arming limit is from SITL timing: the runner's period in wall time at 8x has its own jitter.
- `SIM_GPS1_ENABLE=0` stands for GPS 1 failing; a physically absent GPS 1 (`GPS1_TYPE` 0) was not flown.
- `install-pi.sh` and the systemd service were still not run. Mission Planner / QGC loading the `.waypoints` file was not tried.

## Your call

1. The planned route is over the 40 m blind limit and the drone lands at it, exactly as the planner flags. Options: a route
   with no stretch above about 38 m (the longest stretch that flew through, 38.5 m, peaked at 6.7 m error), a velocity source
   for source set 2, or accepting LAND.
2. The route starts inside a blind stretch (the first 15.5 m) and ends in one, so GPS 2 has no fix at takeoff and during the
   last descent. The app could refuse starts whose first cell is blind, or the package could say the takeoff must be in ALT_HOLD.
3. Whether the runner should leave the 5 Hz arming rule to the pilot (arm in AltHold, as flown here) or run at 5 Hz until armed.
