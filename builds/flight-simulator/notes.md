# flight-simulator

**Status:** ready for your review · **Unblocks:** `008-particle-filter-sequence`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

A whole flight instead of single draws. It flies a route over one raster (what the
drone sees), drifts on a simulated INS, and corrects against another raster (the
onboard map) every few seconds. It logs truth, the INS-only track and the corrected
track, and can render a video: the drone's view, with the map and the segmentation
in the corners. The model, imagery, route, INS noise, correction interval, corrector
and every video element are notebook parameters. The same code will run 008's
sequence benchmark.

## What I needed from you

Nothing.

## What I did

Code, all in `uavloc/`:

| Module | What |
|---|---|
| `simulation/route.py` | `Pose`; waypoints as map fractions, lat/lon or random; constant-speed sampling; `check_inside` |
| `simulation/ins.py` | `InsConfig` + `InsSimulator`. Correlated drift: a velocity bias that random-walks, a constant heading error, and an odometry scale error. Reported yaw / altitude noise go to the corrector. Drift and report noise use separate RNG streams, so the INS track stays the same whatever the correction interval |
| `simulation/camera.py` | `RasterCamera`: heading-up nadir view at the true pose, with an optional `transform` hook (blur, colour shift, …) |
| `simulation/correctors.py` | `Corrector` protocol + `Fix`. `InsOnly`, and `CorrelatorCorrector`, which wraps `PositionCorrelator`: any features (VGG11, U-Net, LR-ASPP), window, prior and gate. A particle filter or Kalman corrector only has to implement `correct()` |
| `simulation/flight.py` | `simulate_flight` → `FlightResult` (per-pose track DataFrame + fixes); `flight_summary` (end-point errors first) |
| `plotting/flight.py` | `plot_tracks` (tracks over the map, end errors in the legend, fix markers by decision), `plot_errors` |
| `plotting/flight_video.py` | `FlightVideo` + `VideoConfig`. Main panel and corner panels are any of `drone`, `map`, `segmentation`, `heatmap`, or your own function via `extra_panels`. Map follows the drone or shows the whole route. Optional trail length, footprint, fixes, legend, HUD, segmentation overlay. Segmentation from the last fix or from every frame. `frame(step)` previews a frame; `save(path)` writes H.264 |
| `data/rasters.py` | `load_raster(spec)`: `{"kind": "multidate", ...}` or `{"kind": "file", path, corners}` |

Small edits: `geo.py` (`east_north_m_array`, `bearing_deg`, `rotate_clockwise`),
`GeoImage.crop_rotated(allow_outside=...)`, `PositionCorrelator.clear_cache()`.
None of them change existing behaviour.

The driver is `tools/flight_simulator/flight_simulator.ipynb` (Imports → Params → Data →
Model → Run → Plots → Save → Video; the Params cell is tagged `parameters`). Outputs go to
`tools/flight_simulator/outputs/<RUN_NAME>/`: `run.json` (summary + every setting),
`track.csv`, `tracks.png`, `errors.png`, `flight.mp4`. Outputs are gitignored. Executed
copies from papermill land in `tools/flight_simulator/runs/`.

## Does it work

Run on real imagery: bushcha, drone view = `current`, map = newest Wayback, LR-ASPP
satellite BN on both sides, weighted NCC, 1.02 km loop at 40 m, 10 m/s, a fix every 5 s,
seed 42.

- `figures/tracks-default.png`: the INS drifts to **30.8 m** at the end point. The corrected
  track ends at **2.6 m**. 15 of 20 fixes applied; the 5 rejects are flat heatmaps (field).
  The tracks land on the roads they are flown over.
- `figures/errors-default.png`: error over time. The fix at t = 75 s jumps from 0.4 to
  14.6 m (a confident wrong peak), and the next fix at t = 80 s recovers to 2.0 m.
- `figures/tracks-gated-divergence.png`: the same flight with the 007 decision stack (round
  R = 20 m, σ = 10 m, τ = 0.25, full gate). Early fixes are good: across-only works along
  the first road. Then two wrong accepts at 75 and 85 s (s1 ≈ 2.5 m) move the estimate
  28 m and then 43 m off. The next searches are centred on the wrong place, so it never
  recovers, and it ends at **45.5 m, worse than INS alone**. This is a sequence effect the
  single-frame benchmark cannot show. It is one flight, not a result.
- `figures/tracks-birky-random-80m.png`: a random 1.9 km route on birky at 80 m. It runs
  end to end, including one `out-of-raster` fix near the edge, which keeps the prior.
- `figures/video-frame-default.jpg`, `figures/frame-overlay.png`: video frames (drone view,
  segmentation, follow map with footprint and legend, heatmap showing the road ridge, HUD).
- Also run: `ins-only` (estimate = INS track exactly; 30.78 m both), VGG11 + cosine
  (runs; 1.1 s per fix on CPU), every-frame segmentation + route map + trail.
- Deterministic: re-running the default gives the same 2.593 m end error.

Timing (i7 CPU): whole-map segmentation ~8 s once; ~130 ms per fix (LR-ASPP) and
~1.1 s (VGG11). A 205-frame video takes ~5 s.

**Still wrong or unverified:**

- The drone view is a satellite crop (0.38 m/px, nadir, no perspective). It is blurry in
  the video because it is upsampled about 30×. Real drone frames are not supported yet.
  A recorded-flight camera (logged poses + frames) would plug into `DroneCamera`.
- Turns are instantaneous: yaw jumps at each waypoint.
- The seg models trained on dubrova and on birky below row 704, so routes there are not
  held out. The other 57 multidate villages (bushcha included) are unseen. The Params
  cell says so.
- The INS model is plausible, not calibrated against the 2025 flight logs.

## Your call

Open the notebook, change the Params, and run it. Things to look at: `tracks-default.png`
against `tracks-gated-divergence.png`, and a video (`outputs/bushcha-loop/flight.mp4`).
Is the INS drift magnitude realistic enough for your platform, or should the defaults
come from the 2025 `.bin` logs?
