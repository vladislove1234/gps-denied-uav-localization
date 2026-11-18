# 027-sequence-filters — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

On whole simulated flights over villages the segmentation models never saw, does a filter
that carries the correlator's whole heatmap from fix to fix end closer to the truth, and
diverge less often, than today's commit-to-the-argmax correction? And which belief carries
the ambiguity better: a dense particle cloud (A) or a few heatmap-peak hypotheses (B)?

## Hypothesis

Yes, and A and B win in different situations.

The correlator's failures that 005–007 and 023 measured happen when it has to commit to one cell:

- **Road slide.** The heatmap is a ridge along a road, so the argmax lands anywhere on it. The
  floor at 5 m sits around 4 m even with perfect masks (006).
- **Confident wrong peaks.** They pass any per-frame gate (007). In `builds/flight-simulator`
  two wrong accepts on the bushcha loop push the 007-gated flight to 45.5 m, against 30.8 m
  for INS alone.
- **No calibrated "keep the prior".** The 023 head ranks positions better on holdout (−0.83 m,
  CI clear of 0) but loses its bar because it moves on every draw.

A filter never commits per frame. A ridge narrows the belief across the road and leaves it
open along it until a junction or a building pins it. A wrong peak has to agree with the INS
motion over several fixes before the belief follows it. A flat heatmap changes almost nothing,
so no gate is needed.

- **A (dense particle filter)** should win from a normal start: the belief is small, one cloud
  covers it, and it can take any shape (a strip along a road).
- **B (multi-hypothesis tracker)** should win from a cold start: the belief is large and the
  heatmap has several separated peaks. A few weighted hypotheses represent that directly,
  while a cloud has to cover all the empty area between them.

Yaw comes from the INS in both. The large 2025 yaw offsets were a log-synchronisation
artefact, not heading drift (user, 2026-11-01), and 2° of heading error moves the edge of a
46 × 26 m frame by under 1 m.

## Method

Everything runs in the flight simulator. Drone view = the `current` Esri date, map = the
newest Wayback date. Model: MobileNetV3 LR-ASPP satellite BN on both sides. Matching:
weighted NCC with the 005 class weights `[0, 1, 1, 0.5, 0, 0.5]`, as in 007.1. A fix every 5 s.

### One heatmap per fix, shared by both filters

At each fix the correlator scores **one** search window that covers the filter's belief: the
bounding box of the 3σ region (A: the particle covariance; B: the union of the hypotheses'
3σ ellipses), never smaller than 007.1's window (rect, 1.7 × footprint) around the estimate,
and capped at 200 m a side (crop logged when it happens). Both filters read the same z-scored
heatmap, so their scores are comparable and A and B differ only in how they hold the belief.
Cells outside the window or NaN count as z = 0 (no evidence). A blind frame (weighted-class
coverage below 1 %, or a flat heatmap) skips the update.

### A: dense particle filter (x, y)

- **Belief:** N = 2,000 particles (east/north offsets), Gaussian around the start with the
  scenario's σ₀.
- **Predict (every pose):** the INS increment plus `N(0, q_pos² · dt)` per axis.
- **Update (every fix):** each particle's weight × `exp(z / T)`, z read at its heatmap cell.
- **Resample:** systematic, when the effective sample size drops below N / 2.
- **Output:** weighted mean and covariance.

### B: multi-hypothesis tracker (x, y)

Deterministic, as the user framed it on 2026-10-03: a handful of hypotheses, no random
particles, no resampling.

- **Belief:** up to K_max = 5 hypotheses, each a Gaussian (position, covariance) with a weight.
  It starts as one hypothesis at the start with σ₀.
- **Predict (every pose):** every hypothesis moves by the INS increment; covariance grows by
  `q_pos² · dt · I`.
- **Update (every fix):** for each hypothesis, multiply `exp(z / T)` by its Gaussian over the
  shared heatmap. Take the top 3 local maxima of that product, at least 10 m apart (NMS). Each
  becomes a child: position = the maximum, covariance = the product's covariance within 10 m
  of it plus (1 m)² (so a child on a road is elongated along it), weight = parent weight × the
  product's mass within 10 m. New hypotheses are born like this at every fix, not only the first.
- **Merge and prune:** children closer than 5 m merge (weights add). Keep the K_max heaviest;
  renormalise.
- **Output:** the heaviest hypothesis's position and covariance; its share of the total weight
  is logged as the dominance.

Neither filter has recovery heuristics (no reseeding, no global search). If it diverges, that
is logged as divergence.

### Arms

The same routes and the same INS realisation (seeded per route) in every arm, in both scenarios.

| Run (normal / cold) | Estimator | Uses |
|---|---|---|
| `027.1` / `027.6` | INS only | lower bound |
| `027.2` / `027.7` | correlator, replace | today's system (007.1: rect 1.7×, flat prior, no gate) |
| `027.3` / `027.8` | correlator + 007 gate, replace | τ = 0.25, `s_max` = 3 m, `c_min` = 0.01, full gate (the `007.8` values) |
| `027.4` / `027.9` | **A: dense particle filter** | tuned T, q_pos |
| `027.5` / `027.10` | **B: multi-hypothesis tracker** | tuned T, q_pos |

### Scenarios

| | Normal start (primary) | Cold start |
|---|---|---|
| Position error at GNSS loss | 0 m (`InsConfig.initial_error_m = 0`) | 40 m, random direction (`initial_error_m = 40`) |
| σ₀ the filters are told | 1 m | 25 m per axis |
| Stands for | GNSS just lost | a long blind stretch, or a bad initial fix |

The replace arms keep 007.1's window in both scenarios: that is today's system, and it cannot
reach 40 m. This favours the filters in the cold start, which is why the cold start does not
decide the bar.

### Tuning (fixed grid, tuning villages, normal start only)

| Knob | Grid | Arms |
|---|---|---|
| `T` | 0.25, 0.5, 1, 2 | A, B |
| `q_pos` (m / √s) | 0.1, 0.3, 1.0 | A, B |

Each filter takes the combination with the lowest `mean_end_error_m` on the tuning villages.
The chosen values are logged in `runs.jsonl` and reused unchanged for the cold start. The
filter that decides the bar (A or B) is the one with the lower tuning-village error, chosen
**before** the test runs. The test villages are scored once.

### Flights

The `sim-flights` route set specified in `008`, reused unchanged:

- **Villages:** every `ua-village-multidate` location except birky and dubrova (the seg models'
  training ground). Sorted by name, shuffled with seed 42: the first **10 are tuning**, the rest
  (**47**) are **test**.
- **Routes:** 2 per village, 4 random waypoints each, at least 80 m from both rasters' edges
  (the cold start's 40 m offset plus a footprint), seed = hash of (village, route index).
  40 m altitude, 10 m/s, poses every 0.5 s, a fix every 5 s.
- **INS** (seeded per route, the same in every arm): velocity bias std 0.3 m/s, random walk
  0.02 m/s/√s, heading bias std 2°, speed scale std 1 %, reported yaw noise 1°, reported
  altitude noise 1 m.

## Success criterion

Unit of analysis: one **flight**. CIs are 95 % paired bootstraps over **villages** (both routes
of a village resampled together), 10,000 resamples, seed 42.

- **End-point error:** distance from the estimate to the truth at the last pose.
- **Diverged:** the end-point error is larger than INS-only's on the same flight.

No flight-level run exists yet in any `runs.jsonl`, so the baseline is measured here. The only
earlier numbers are one bushcha flight in `builds/flight-simulator` (replace 2.593 m, 007-gated
45.5 m, INS 30.8 m): an example, not a baseline.

**Metric:** `mean_end_error_m` over the 94 test flights, normal start (primary); `divergence_share`
**Baseline:** the better of `027.2` / `027.3` on the test flights, chosen after the test runs (the
stronger baseline, so the choice can only make the bar harder)
**Bar (the chosen filter, `027.4` or `027.5`, against that baseline), all of:**
1. `mean_end_error_m` ≤ **0.7 ×** the baseline's, with the CI of the difference clear of 0.
2. `divergence_share` no higher than the baseline's.
3. Median per-fix time ≤ **1.0 s** on the M1 CPU, drone segmentation included (a 5 s fix
   interval with room for a Pi 5).

**Pre-registered secondary comparisons** (reported with CIs, do not decide the bar):
- A vs B, normal start: `mean_end_error_m` difference (expected: A ≤ B).
- A vs B, cold start: `mean_end_error_m` and `recovered_share` = share of flights ending
  within 10 m (expected: B better).
- Each filter vs INS only, cold start.

Also reported for every arm: mean error over the flight, p90 and max error, along-track error,
share of fixes skipped, per-fix time, a per-flight table. For B: number of live hypotheses and
dominance over time.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/flight-simulator` (route, INS, camera, flight loop, plots, video) | built, ready for your review | its notes |
| Stateful estimators in the flight loop: `predict(move_m, dt_s)` every pose, `update(view, yaw, alt)` at fixes, estimate + covariance out; today's `CorrelatorCorrector` wrapped as the replace policy | to build (`uavloc/simulation/`) | the wrapped correlator reproduces `simulate_flight`'s bushcha-loop end error (2.593 m) exactly |
| Belief-covering search window (the window fix): score a window given as a box in metres, cap and z = 0 fill | to build (`uavloc/matching/window.py`) | (i) with the 007.1 box it reproduces `PositionCorrelator.surface` heatmaps (same argmax, max z difference < 1e-3); (ii) **on the tuning villages, normal start, the chosen filter's window holds the truth on ≥ 99 % of fixes**; `truth_in_window` is logged per fix for every arm |
| A: dense particle filter | to build (`uavloc/filtering/particles.py`) | synthetic: (a) map = drone raster's own class map → within 1 m in ≤ 3 fixes; (b) flat heatmap → the cloud follows the INS and spreads; (c) single straight road → covariance elongated along it (s1 / s2 > 3); (d) N = 2,000 update timed |
| B: multi-hypothesis tracker | to build (`uavloc/filtering/hypotheses.py`) | synthetic: (a) as A; (b) as A; (c) two identical peaks 30 m apart → two hypotheses with weights within 10 % of each other, until a fix that tells them apart, after which the wrong one is pruned within 2 fixes; (d) a road ridge → the child covariance is elongated along it |
| `sim-flights` route set: villages, splits, waypoints, seeds | to build (`datasets/sim-flights/`, manifest + card) | every route passes `check_inside` for both rasters at 80 m margin; `preview.png` of 6 routes; tuning / test split in the manifest |
| Multi-flight runner + village bootstrap | to build (glue here; to `uavloc/` once 008 reuses it) | its INS-only arm matches the simulator's INS track per route |
| Map class-probability rasters per village (LR-ASPP, newest date) | to build (cache) | same shapes and corners as the imagery |

Run order: build → acceptance checks → tune on tuning villages → run 027.1–027.10 on test →
report. Stop and report if an acceptance check fails.

> **2026-11-02 amendment (at build, before any run; the success criterion is unchanged).**
> (1) The window fix is folded in here (user): 007 showed no fixed window works (rect 1.7
> reaches the truth on 56 % of 10 m draws and 10 % of 15 m; rect 3.0 and round 30 m lose
> elsewhere), so the window is sized from the belief, and check (ii) above verifies its reach.
> (2) The "widen ×2 when the peak sits on the window edge" rule discussed on 2026-11-02 is
> **logged only** (`peak_at_edge`), not acted on: nothing in either belief lives outside the
> window, so a wider window cannot change either filter's answer without a recovery rule,
> which this proposal excludes. (3) The filters score their window with FFT NCC
> (`learned.batched_ncc`, equal to direct conv2d to ~1e-7; 8 ms vs 4 s on a 520 px window).
> The replace arms keep `PositionCorrelator.correct` unchanged, so bar 3's timing compares
> different NCC implementations; that favours the filters and is reported. (4) Map
> class-probability rasters are computed in memory once per village per run instead of
> cached on disk (~100 MB each at float16; 57 villages).

> **2026-11-02 amendment 2 (after the acceptance checks, before any tuning or test run; the
> success criterion is unchanged; user chose to proceed).** Three checks failed at the default
> T = 1 (raw results kept in `artifacts/checks-as-registered.json`, diagnostics in
> `artifacts/diagnostics/`): self-match 2.54 / 2.36 m after fix 3 (bar 1 m), straight road
> s1/s2 2.15 / 1.89 (bar 3), twin weight 0.027 after the distinguishing fix (bar 0.01).
> Causes: (a) the self-match check flew the sim-flights INS, whose reported yaw (2° bias +
> 1° noise) and altitude noise floor matching at ~1–2.5 m on a perfect map; with exact
> reported pose both filters reach 0.16–0.48 m by fix 3, so the check is re-run with exact
> reported pose (`self_match_converges_exact_pose`). (b) exp(z / T) at T = 1 is soft: the road
> check passes at T = 0.25 (7.95 / 3.85) and the twin check at T = 0.5 (0.0006); no single T
> passes every check, and T is tuned. (c) The `q_pos` grid gains **2.0 m/√s**. The INS
> velocity bias (0.3 m/s per axis, constant per flight) moves the truth ~1.5 m per 5 s
> interval at 1σ, so a random-walk model needs q ≈ 1.5 / √5 = 0.67 at 1σ and 1.34 at 2σ; the
> registered maximum of 1.0 cannot cover a 2σ flight. This was noticed on the flight
> simulator's own bushcha example flight (a test village, but not a `sim-flights` route),
> where the PF at q = 0.3 lost the truth from its cloud. The grid is now
> T ∈ {0.25, 0.5, 1, 2} × q_pos ∈ {0.1, 0.3, 1.0, 2.0}.

> **2026-11-02 amendment 3 (after tuning, before any test run; the success criterion is
> unchanged; user's decisions).** Tuning chose A: T = 0.5, q = 2.0 (tuning mean end error
> 22.1 m) and B: T = 0.25, q = 1.0 (26.6 m), so **A decides the bar**. Window check (ii)
> **fails**: A's window held the truth on 96.8 % of tuning fixes (bar 99 %). A's best q sits
> on the grid edge; the user chose to run the test arms with these values rather than
> extend the grid. (2) **023 correlation head arms added (user), outside the bar:**
> `027.11` replace + head, `027.12` A + head, `027.13` B + head, normal start, test flights.
> Head = `023.3` (H, seed 42), as `matching="learned"`. Its z-scale differs from NCC, so T is
> re-tuned per filter on the tuning villages (same T grid), with q fixed at the NCC-tuned
> value. The head trained on 46 of these villages: only its 10 val villages and bushcha
> (excluded from its training) are clean, 5 tuning and 6 test villages, so head results are
> reported on all test flights **and** on the 12 clean test flights (`head_clean_*` metrics),
> and the head-vs-NCC comparison is read on the clean subset.

**What only you can provide.** Nothing. The INS defaults above are accepted as plausible
(user, 2026-11-01: heading error is small; the 2025 offsets were log sync).

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | `ua-village-multidate` (drone view `current`, map newest Wayback), routes from `sim-flights` |
| Split | by village: 10 tuning / 47 test, seed 42; birky and dubrova excluded |
| Seed | 42 (splits, bootstrap, particles); per-route seeds for routes and INS |
| Model | MobileNetV3-Large LR-ASPP, `model-lraspp-bn-satellite.pt`, both sides |
| Loss | none, no training |
| Hardware | Apple M1 Pro, **CPU-only processes** (MPS in the same process corrupts CPU conv2d, bug-121) |

## Known confounds, accepted up front

- **Not drone imagery.** The camera is a satellite crop of a different date (0.38 m/px, nadir).
- **Uncalibrated INS.** The drift model is plausible, not fitted to logs. The filters model it
  as a random walk while the simulator drifts with a correlated bias; `q_pos` is tuned to absorb
  the mismatch.
- **The filters get two tuned knobs; the replace arms reuse 007's values.**
- **One shared window per fix.** B scores its hypotheses from one window rather than one
  correlation each. That is the same evidence (the correlator is translation-invariant inside
  the window), but z-scoring over a bigger window changes the score scale; T absorbs it.
- **Cold start favours the filters** (the replace arms cannot reach 40 m), so it is secondary.
- **Random routes** cross featureless fields as often as the villages allow; weighted-class
  coverage is logged per flight.

## What would falsify this

- **Neither filter meets bar 1:** carrying the heatmap over time does not beat committing per
  frame on this imagery. Check the per-flight table: if the likelihood is the problem (the
  belief follows wrong peaks), the next step is a better per-frame score (023 head, 008's
  per-pixel likelihood), not a better filter.
- **Bar 1 met, bar 2 missed:** better on average but fails harder. It needs divergence handling
  (the "re-seed from peaks when lost" combination) before it is usable.
- **Bars 1–2 met, bar 3 missed:** right idea, too slow. Cap the window or cut N and re-time.
- **A and B equal in both scenarios:** the belief representation does not matter here; keep the
  cheaper one.
- **B worse than A in the cold start:** peak-seeding loses the true position at birth more often
  than it saves work. Look at how often the truth was outside every hypothesis.

## Related

- `008-particle-filter-sequence`: the per-pixel class likelihood with yaw in the state. 027 tests
  the cheaper heatmap likelihood first; 008 waits on its result. 027 reuses 008's flights, INS
  and bar shape.
- `builds/flight-simulator`: the equipment; the gated-flight divergence example.
- `007-search-prior-and-gate`: the gate values for `027.3`; τ does not transfer between places.
- `023-learned-correlation-head`: its heatmap is the natural next likelihood if 027 works
  (an optional arm, not part of this bar; it trained on 46 of these villages, so only its 10
  val villages would be clean).
- `003-pose-search`: heading search helped on the 2025 frames because of log sync; not used here.
