# 008-particle-filter-sequence — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

On whole simulated flights over villages the segmentation models never saw, does a
particle filter with a per-pixel class likelihood end closer to the true position,
and diverge less often, than today's correlator? The comparison includes the correlator
with the 007 gate and with a Kalman filter fed its uncertainty.

## Hypothesis

It will. 005–007 measured three things that one frame at a time cannot fix. A single
46 × 26 m frame is not unique: it slides along roads, and the floor at 5 m sits around
4 m even with perfect masks. The window-normalised NCC score has no scale that
transfers between places (τ 0.25 vs 2.0 in 007). And confident wrong peaks (bushcha
trees) pass any per-frame gate.

In `builds/flight-simulator` one flight with the 007 decision stack shows what that
does over time. Two confident wrong fixes (s1 ≈ 2.5 m) move the estimate 28 m and then
43 m off, the next searches are centred on the wrong place, and the flight ends at
45.5 m, worse than INS alone (30.8 m). That is one flight, not evidence. It is the
failure the sequence setup exists to measure.

A particle filter changes the mechanism:

- **Road ambiguity:** particles stay spread along a road until a fork or a building
  pins them. The ambiguity is carried, not guessed.
- **Confident wrong peaks:** a wrong peak must also agree with the INS motion over
  several fixes before the particle cloud follows it.
- **Score scale:** the likelihood is a real probability read from the satellite
  class-probability map, so nothing is normalised per window and there is no τ.
- **Heading:** yaw is part of the state, so it is estimated instead of assumed exact.

## Method

Everything runs in the flight simulator (`builds/flight-simulator`). Drone view = the
`current` Esri date, map = the newest Wayback date. Model: MobileNetV3 LR-ASPP
satellite BN on both sides, as in 005–007. Map class probabilities are computed once
per village. A fix is attempted every 5 s.

### Arms

The same routes and the same INS realisation (seeded per route) in every arm.

| Run | Estimator | Uses |
|---|---|---|
| `008.1` | INS only | lower bound |
| `008.2` | correlator, replace | today's system: an applied fix overwrites the estimate |
| `008.3` | correlator + 007 gate, replace | τ = 0.25, `s_max` = 3 m, `c_min` = 0.01, full gate (the `007.8` values) |
| `008.4` | correlator + Kalman | state (east, north); predict with INS increments, `Q = q_pos² · dt · I`; measurement = posterior MAP, `R = Σ_post + (1 m)² I` at τ = 0.25; innovation gate χ²(2) > 9.21 → skip |
| `008.5` | **particle filter (x, y)** | yaw from the INS |
| `008.6` | **particle filter (x, y, yaw)** | yaw offset in the state |

### Particle filter (fixed now)

- **State:** east/north offset per particle, plus a yaw offset `δ` in `008.6`. There are
  **N = 2,000** particles. At GNSS loss they are Gaussian around the start with a 1 m std
  and δ = 0.
- **Predict (every pose):** the INS increment, rotated by the particle's `δ`, plus
  `N(0, q_pos² · dt)` per axis. In `008.6`, `δ` also random-walks by
  `N(0, q_yaw² · dt)`.
- **Update (every fix):** sample **K = 256** drone pixels from classes with weight > 0,
  stratified by class. If fewer than 1% of drone pixels are weighted, skip the update
  (INS only, as in 007). Then
  `log L_i = (1/κ) Σ_k w_{c_k} · log(ε + P_map(c_k at T_i(p_k)))`, where `T_i` places
  pixel `p_k` of the heading-up drone map on the map using particle `i`'s position and
  yaw. `κ` tempers the sum, because neighbouring pixels are not independent evidence.
  Class weights are the 005 ones, `[0, 1, 1, 0.5, 0, 0.5]`.
- **Resample:** systematic resampling when the effective sample size drops below N/2.
- **Output:** weighted mean position (the estimate) and weighted covariance.
- **No recovery heuristics** (no random re-seeding, no global search). If it diverges,
  that is logged as divergence.

### Tuning (fixed grid, tuning villages only)

| Knob | Grid | Arms |
|---|---|---|
| `ε` | 0.01, 0.05, 0.2 | 008.5, 008.6 |
| `κ` | 16, 64, 256 (effective pixels = K / κ … K) | 008.5, 008.6 |
| `q_pos` (m / √s) | 0.1, 0.3, 1.0 | 008.4, 008.5, 008.6 |
| `q_yaw` (deg / √s) | 0.1, 0.5 | 008.6 |

Each arm picks the combination with the lowest mean end-point error on the **tuning**
villages. The chosen values are logged in `runs.jsonl`. The test villages are scored
once, with those values.

### Flights

- **Villages:** every `ua-village-multidate` location except birky and dubrova, the
  seg models' training ground. Sorted by name and shuffled with seed 42: the first
  **10 are tuning** and the rest (**47**) are **test**. Routes on `bushcha` stay in, since
  it is held out.
- **Routes:** 2 per village, 4 random waypoints each, at least 80 m from both rasters'
  edges, seed = hash of (village, route index). 40 m altitude, 10 m/s, poses every
  0.5 s, a fix every 5 s. Route length follows the waypoints (logged per route).
- **INS** (seeded per route, the same in every arm): velocity bias std 0.3 m/s,
  random walk 0.02 m/s/√s, heading bias std 2°, speed scale std 1%, reported yaw
  noise 1°, reported altitude noise 1 m, no initial error.

## Success criterion

Unit of analysis: one **flight**. The CIs are 95% paired bootstraps over **villages**
(both routes of a village resampled together), 10,000 resamples, seed 42.

- **End-point error:** distance from the estimate to the truth at the last pose.
- **Diverged:** the end-point error is larger than INS-only's on the same flight.

**Metric:** `mean_end_error_m` over test flights (primary), `divergence_share`
**Baseline:** the best of `008.2`–`008.4` on `mean_end_error_m`, chosen after the
test runs (all three are run regardless)

**Bar (008.6 against that baseline), all of:**
1. `mean_end_error_m` ≤ **0.7 ×** the baseline's, with the CI of the difference clear of 0.
2. `divergence_share` no higher than the baseline's.
3. Median update time ≤ **2 ×** the correlator's `ms` per fix on the same CPU, drone
   inference included. The work has to stay Raspberry-Pi-shaped.

`008.5` against `008.6` isolates the value of yaw in the state. It is reported, but it
does not decide the bar.

Also reported for every arm: mean and p90 along-track error, max error, share of fixes
skipped or rejected, and a per-flight table.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/flight-simulator` (route, INS, camera, flight loop, plots, video) | built, **ready for your review** | its notes |
| Stateful estimators in the flight loop: `predict(move_m, dt_s)` every pose, `update(view, yaw, alt)` at fixes, estimate + covariance out. Today's `CorrelatorCorrector` wrapped as the replace policy | to build (`uavloc/simulation/`) | the wrapped correlator reproduces `simulate_flight`'s bushcha-loop track exactly (end error 2.593 m) |
| Kalman estimator (`008.4`) | to build | synthetic: with perfect measurements and R → 0 it snaps to them; with R → ∞ it equals INS |
| Particle filter + class likelihood (`008.5`, `008.6`) | to build | synthetic: (a) map = drone raster's own class map → converges within 1 m in ≤ 3 fixes; (b) a flat map → the cloud only follows the INS and spreads; (c) a single straight road → covariance elongated along the road (s1/s2 > 3); (d) N = 2,000 update timed on CPU |
| `sim-flights` route set: villages, splits, waypoints, seeds | to build (`datasets/sim-flights/`, manifest + card) | every route passes `check_inside` for both rasters; `preview.png` of 6 routes; tuning/test split written in the manifest |
| Multi-flight runner + village bootstrap | to build (glue here; to `uavloc/` if 009 reuses it) | the runner's INS-only arm matches the simulator's INS track per route |
| Map class-probability rasters per village (LR-ASPP, newest date) | to build (cache) | the same shapes and corners as the imagery |

Run order: build → acceptance checks → tune on tuning villages → run 008.1–008.6 on
test → report. Stop and report if an acceptance check fails.

**What only you can provide.** A decision after reviewing `builds/flight-simulator`:
are the INS defaults realistic for your platform, or should they come from the 2025
`.bin` logs first? The values above are plausible guesses, not calibrated.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | `ua-village-multidate` (drone view `current`, map newest Wayback), routes from `sim-flights` |
| Split | by village: 10 tuning / 47 test, seed 42; birky and dubrova excluded |
| Seed | 42 (splits, bootstrap); per-route seeds for routes and INS |
| Model | MobileNetV3-Large LR-ASPP, `model-lraspp-bn-satellite.pt`, both sides |
| Loss | none, no training |
| Hardware | CPU (i7) for runs and timing |

## Known confounds, accepted up front

- **Not drone imagery.** The camera is a satellite crop of a different date (0.38 m/px,
  nadir). The drone-to-satellite gap stays open, as in 005–007.
- **Uncalibrated INS.** The drift model is plausible, not fitted to the 2025 logs. A
  different drift changes how much there is to correct and how fast the filters
  must trust vision.
- **The PF gets tuned knobs; the correlator arms mostly reuse 007's.** The Kalman arm is
  tuned on the same grid for `q_pos`, so the strongest non-PF baseline is not starved.
- **Random routes** cross featureless fields as often as the villages allow. That is
  realistic, but results depend on how much of each route has structure; logged per
  flight as weighted-class coverage.
- **Newest map only.** A stale-map arm (oldest Wayback) is a natural follow-up, not
  part of this bar.

## What would falsify this

- **008.6 misses bar 1:** carrying the ambiguity over time does not beat committing to
  per-frame fixes on this imagery. Look at whether the likelihood or the motion model
  is at fault (the per-flight table), then decide between a better likelihood
  (structural terms, chamfer) and dropping the PF.
- **Bar 1 met, bar 2 missed:** the PF is better on average but fails harder. It needs
  divergence handling before it is usable.
- **Bars 1–2 met, bar 3 missed:** right idea, too slow. Reduce N or K and re-time,
  rather than abandon it.

## Related

- `builds/flight-simulator`: the equipment; the gated-flight divergence example.
- `007-search-prior-and-gate`: covariance at τ = 0.25 as confidence (`007.8`); τ does not transfer.
- `006-segmentation-ceiling`: masks are not the bottleneck (~1 m); the ~4 m floor at 5 m.
- `005-village-crossdate`: the birky-15 road slide; the newest map helps 2–2.6 m.
- `archive/uav_localization_2025/particle_filter/drone_localisator.py`: the 2025 particle filter (autoencoder cosine likelihood, no INS motion model), a starting point only.
- **2026-11-02 note:** `027-sequence-filters` tests the cheaper heatmap likelihood first, with a
  dense particle filter and a multi-hypothesis tracker side by side, on this proposal's flights
  and INS. 008's per-pixel likelihood waits on 027's result. Nothing above changed.
