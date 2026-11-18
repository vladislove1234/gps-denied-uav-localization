# 028-ins-error-state — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a particle filter that carries the INS error in its state (velocity bias, heading
bias, scale) end closer to the truth than 027's particle filter, which models the drift
as white noise on position, on the same simulated test flights?

## Hypothesis

Yes, by a lot on long flights. 027's tuning showed the mismatch:

- The best `q_pos` for 027's filter sat on the grid edge (2.0 m/√s, added by amendment 2),
  and its window held the truth on only 96.8 % of tuning fixes (bar 99 %).
- The simulator's INS drifts with a velocity bias held for the whole flight (0.3 m/s per
  axis), a constant heading bias (2°) and a scale error (1 %). Over a 2.6 km route that is
  ~100 m. White noise on position either under-covers that drift (the truth leaves the
  cloud) or over-spreads the cloud everywhere (the heatmap has to fight a huge cloud).
- On six long blind synthetic flights (build sanity check, 2026-11-02), the INS-error
  filter's 99 % ellipse held the truth on 100 % of poses; 027's filter at q = 2 on 15–48 %.

The mechanism: a particle moves the way its own INS-error guess says the INS went wrong,
so after a few fixes the survivors are the particles whose bias explains where the heatmap
kept finding the drone. The cloud then predicts the drift ahead instead of spreading to
cover it, and the window stays small and on the truth.

## Method

Everything is 027's setup (`sim-flights` routes, simulator, drone view `current`, map newest
Wayback, LR-ASPP satellite BN both sides, weighted NCC, a fix every 5 s, the belief-covering
window, blind-frame skip), except the motion model.

### Filter (`uavloc.filtering.ins_error.InsErrorParticleFilter`)

- **State per particle:** east/north position, velocity bias (east, north), heading bias,
  speed scale. N = 2,000.
- **Start:** position `N(0, σ₀²)`; velocity bias `N(0, 0.3²)` m/s per axis; heading bias
  `N(0, 2²)` deg; scale `1 + N(0, 0.01²)`. These are the IMU spec, read from the route set's
  INS config.
- **Predict (every pose):** true move = rotate(−heading bias) · (INS move − velocity bias · dt)
  / scale, plus `N(0, q_pos² dt)`. Velocity bias random-walks at the spec's 0.02 m/s/√s;
  heading bias jitters at 0.01 °/√s so resampled copies do not collapse.
- **Update, resample, output:** 027's particle filter, unchanged (`exp(z / T)`, systematic
  resampling at ESS < N/2, weighted mean and covariance).

### Runs

| Run | Scenario | Compared with |
|---|---|---|
| `028.1` | normal start (primary) | `027.4` (027's A, its flights CSV, paired by route) |
| `028.2` | cold start (40 m) | `027.9` |

Same routes, same INS realisation per route (route seed), same PF seed per route.

### Tuning (tuning villages, normal start, before any 028 test run)

| Knob | Grid |
|---|---|
| `T` | 0.25, 0.5, 1 |
| `q_pos` (m/√s) | 0.03, 0.1, 0.3 |

Lowest `mean_end_error_m` on the 20 tuning flights wins; chosen values logged in the config.

## Success criterion

Unit: one flight. CIs: 95 % paired bootstrap over villages (both routes together), 10,000
resamples, seed 42, as in 027.

**Metric:** `mean_end_error_m` over the 94 test flights, normal start
**Baseline:** `027.4` (027's particle filter, tuned T = 0.5, q = 2.0), its value in 027's `runs.jsonl`
**Bar, all of:**
1. `mean_end_error_m` ≤ **0.8 ×** `027.4`'s, with the CI of the paired difference clear of 0.
2. `truth_in_window_share` ≥ **0.99** (the consistency 027's filter missed).
3. `divergence_share` no higher than `027.4`'s.
4. Median per-fix time ≤ 1.0 s on the M1 CPU.

Secondary (reported, not barred): `028.2` vs `027.9` in the cold start; mean error over the
flight; truth-in-ellipse share; the estimated vs simulated INS bias per flight.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| 027's sequence equipment (`uavloc.simulation.{estimators,sequence}`, `uavloc.matching.window`, `sim-flights`) | built, checked in 027 | 027 checks |
| `uavloc.simulation.arms` (promoted from 027's `arms.py`, + `ins-error`) | built | 027's notebooks import it unchanged (re-export) |
| `uavloc.filtering.ins_error` | built | (a) blind flights: truth inside the 99 % ellipse on ≥ 95 % of poses over 6 tuning-village INS realisations; (b) self-match with exact reported pose: within 1 m by fix 3; (c) same timing bar as 027 |
| 027's `flights-027.4.csv`, `flights-027.9.csv` | written by 027's test runs | 94 flights each, same (village, route) keys |

> **2026-11-02 amendment 1 (after the acceptance checks, before any tuning or test run;
> the success criterion is unchanged; overnight autonomy granted by the user).** Check (a)
> **failed as registered**: over the 6 tuning-village INS realisations the truth sat inside
> the blind filter's 99 % ellipse on 88.3 % of poses on average (bar 95 %), because one flight
> (lisnyi #1) drew a 2.66σ north velocity bias and a 1.5σ scale error and spent 70 % of its
> poses outside. Six flights cannot separate calibration from one tail draw, so calibration
> was measured directly (`artifacts/blind_calibration.py`, 8 routes × 25 INS draws, no
> images): **1.03 %** of poses outside the 99 % ellipse (expected 1 %), 1.0 % at the last pose;
> 027's filter at q = 2.0: 18.9 % and 43.5 %. Checks (b) self-match 0.99 m by fix 3 and (c)
> timing 367 ms pass. Proceeding to tuning.

**What only you can provide.** Nothing.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why — do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | `sim-flights` (027's route set), `ua-village-multidate` imagery |
| Split | 10 tuning / 47 test villages, as 027 |
| Seed | 42 (bootstrap); route seeds for INS and particles |
| Model | MobileNetV3-Large LR-ASPP, `model-lraspp-bn-satellite.pt`, both sides |
| Loss | none |
| Hardware | Apple M1 Pro, CPU-only processes |

## Known confounds, accepted up front

- **The prior is the simulator's own error model.** The filter's priors equal the parameters
  the simulator draws from, and its error model has exactly the simulator's form. A real IMU
  has unmodelled terms (bias instability, misalignment, vibration), so this is a best case
  for the motion model. A follow-up should mis-specify the prior (e.g. 2× too narrow).
- **028.1 is compared against a run of another experiment.** Same code path for everything
  but the motion model, same seeds, so the pairing is exact; `code_changed` across experiments
  is expected and flagged.
- 027's other confounds (satellite-crop camera, uncalibrated INS) carry over.

## What would falsify this

- **Bar 1 missed:** the drift model is not what limits 027's filter on these flights; the
  heatmap (wrong peaks, blind stretches) is. Look at the flights where 027.4 ended worst.
- **Bar 2 missed with bar 1 met:** better on average, still overconfident: the heatmap
  update, not the motion model, makes the cloud collapse onto wrong peaks (T too sharp).
- **Cold start worse than 027.9:** the bias dimensions dilute 2,000 particles over a large
  initial area; more particles or a position-only start would be the next test.

## Related

- `027-sequence-filters`: the setup, the baseline arm (`027.4`), and the tuning evidence
  (q on the grid edge, window check 96.8 %).
- `builds/flight-simulator`: `InsSimulator`, whose error model this filter mirrors.
- `.wolf/cerebrum.md` 2026-10-22 literature note: INS bias in the state, as in marginalised
  terrain-relative navigation filters.
