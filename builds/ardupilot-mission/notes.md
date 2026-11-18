# ardupilot-mission

**Status:** ready for your review · **Unblocks:** route planner steps 3-4 (`docs/route-planner.md`): onboard runner with the stub localizer, the Qt app's "Upload to drone"

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

Step 2 of the route planner: turn a planned route into an ArduCopter out-and-back mission, one parameter
set and the Lua switch, and talk to a vehicle over MAVLink. It is verified against a native ArduPilot SITL
copter on this Mac, which also gives the IMU-only coasting time the blind-stretch limit needs.

## What I needed from you

Nothing. ArduPilot SITL is built outside the repo at `~/Documents/Projects/ardupilot` (master, 4.8-dev,
commit `1d44127`, shallow clone).

## What I did

| Module (`uavloc/ardupilot/`) | What |
|---|---|
| `mission.py` | `MissionPlan`, `MissionItem`, `mission_items` (home, DO_CHANGE_SPEED, TAKEOFF, route out, optional LOITER_TIME, route back reversed, LAND at the start), QGC WPL 110 `write_waypoints` / `read_waypoints`, `mission_differences` (what the vehicle changed). |
| `params.py` | `vehicle_params()`: GPS2 as MAVLink GPS, EK3 source sets 1 and 2, SERIAL2, RC9 switch, Lua on, failsafes. Every name and enum was read from the ArduPilot source. `.param` read/write. |
| `lua/vision-gps-switch.lua`, `scripts.py` | The switch script from `docs/ardupilot-integration.md` (it also re-finds the RC channel if the option appears after boot). |
| `link.py` | pymavlink: `connect`, mission upload/download, `vehicle_home`, `set_params` with read-back, `HOME_WARNING_M`, `home_offset_m`. |
| `gps_input.py` | `send_gps_input`: GPS2, 3D fix, 10 satellites, ignore flags 188 (no velocity). |
| `sitl.py` | `SitlConfig`, `start_sitl` (starts `arducopter`, kills it on exit, refuses if port 5760 is taken), `default_ardupilot_dir` (`ARDUPILOT_DIR` or `~/Documents/Projects/ardupilot`). |
| `vehicle.py`, `flight.py`, `scenarios.py` | Verification helpers: modes, message rates, RC override, a `FlightSession` that records the track and feeds GPS_INPUT from simulator truth, and the three scenarios (GPS_INPUT check, full mission, coasting). |

SITL build (arm64, Apple clang): `PATH=$HOME/miniconda3/envs/uavloc/bin:/usr/bin:/bin CC=/usr/bin/clang CXX=/usr/bin/clang++ python ./waf configure --board sitl && python ./waf copter`, 2.5 min. The first attempt failed
because the Intel Homebrew clang (`/usr/local/opt/llvm/bin`) is first on PATH (`no such sysroot MacOSX26.sdk`,
`'string.h' not found`). Needed in the env: `pymavlink`, `empy==3.3.4`, `pexpect`, `future`.

Launch by hand: `build/sitl/bin/arducopter --model + --speedup 8 --defaults Tools/autotest/default_params/copter.parm,<extra.parm> -I0 --home 49.905,23.885,300,90 --wipe`, MAVLink on `tcp:127.0.0.1:5760`.

## Does it work

Run: `python -m papermill builds/ardupilot-mission/verify.ipynb builds/ardupilot-mission/runs/verify.ipynb --cwd . -k python3`
(4 min). Numbers are in `results.json` (written by the notebook).

- Planned route: the lowest-blind-share route of the Birky pair, weight 2, 3364 m one way, 3.1 % blind, 24
  waypoints, 49 mission items out and back.
- Parameter set: all 23 parameters set and read back from the vehicle.
- Home: the vehicle's home is 0.007 m from the planned start (warning limit 20 m).
- Mission upload and download: 49 items, no differences (home item and the frame of DO commands are rewritten by
  ArduPilot by design).
- GPS_INPUT: a fix offset 30 m north is seen in `GPS2_RAW` with 0.0006 m error, 8 fixes in 4 s.
- Flight (AUTO, takeoff on the real SITL GPS, GPS2 fed from truth in shadow mode): 1418 simulated s, 40.0 m
  height, closest to the end point 1.4 m, landed 0.03 m from the start.
- `figures/track-over-route.png`: truth and EKF track lie on the planned out-and-back path.
- `figures/height-and-ekf-error.png`: height and EKF error over the flight.
- Lua switch: RC9 override high gives `Vision GPS: ON` from the script (GPS_PRIMARY=1 and source set 2).

**IMU-only coasting** (`figures/imu-coasting.png`), flying AUTO at 5 m/s at 40 m, real GPS switched off in the
simulator (`SIM_GPS1_ENABLE=0`), our fixes stopped at t = 0 (two runs of the notebook differed by 0.1 s):

| Variant | Absolute position lost | EKF failsafe (Land) |
|---|---|---|
| fly on our GPS2, source set 2 (no velocity) | 7.5 s | 8.0 s |
| real GPS lost, source set 1 | 7.6 s | 8.1 s |
| our GPS2, failsafe off (watch drift) | 7.6 s | never; EKF stops aiding at about 15 s |

With the failsafe off, EKF error vs truth is 2.4 m at 8 s, about 8 m at 15 s, then grows by roughly 10 m/s
(40 m at 20 s, 440 m at 60 s). So `ekf_coast_s` for the planner is **about 8 s, which is 40 m at 5 m/s**; the
EKF is trustworthy to about 8 m for 15 s, and not at all after.

## Not verified

- The EKF's 8 s is a SITL number with simulated IMU noise, no wind and no vibration. A real airframe will be
  worse, not better. Measure on the bench or in the first flight before trusting a larger blind limit.
- The "fly on our fixes with the real GPS still on" switch-over (0b in the test ladder) was exercised only in
  the coasting runs, not mid-mission with the jump at the switch measured.
- SITL runs at speedup 8 with fixes from truth, so the 0.4-0.55 s localiser latency and the forward propagation
  (section 3 of `docs/ardupilot-integration.md`) are not tested.
- Mission Planner / QGC loading the `.waypoints` file was not tried (the file follows the QGC WPL 110 layout and
  round-trips through our reader).
- Hover (`LOITER_TIME`) is covered by a unit test, not flown.
- Packaging: `lua/vision-gps-switch.lua` is found through the source tree; `pyproject.toml` has no
  `package-data` entry, so a wheel install would miss it. `environment.yml` does not list `pymavlink` yet.

## Your call

The coasting result is much shorter than a "long blind stretch" assumption would need. At 8 s the planner's blind
limit becomes `ekf_coast_s * speed` = 40 m at 5 m/s, tighter than the matcher's search window. Options to widen
it: a velocity source for source set 2 (optical flow, or our own velocity from consecutive fixes sent in GPS_INPUT
with the velocity flags cleared), tuning the EKF3 noise parameters, or accepting LAND over long blind stretches.
