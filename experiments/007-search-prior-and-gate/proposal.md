# 007-search-prior-and-gate — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Two questions about the matcher. The masks stay as they are, because 006 showed
that perfect masks buy about 1 m.

- **A (search):** does a round search window sized to the INS uncertainty, with a
  Gaussian INS prior folded into the score, make a 15 m error correctable
  without the look-alike losses that a bigger window caused in `005.3`?
- **B (gate):** does a pre-registered decision (accept / correct across the
  ridge only / reject) built from the heatmap's shape cut the share of
  corrections that make position worse?

## Hypothesis

**A.** Today every cell in the search window is equally likely, and everything
outside it is impossible. At scale 1.7 the truth is out of reach in 90% of 15 m
draws (005). When it *is* reachable, the mean error is 4.8 m. Scale 3.0 made
the truth reachable, but it let far look-alikes win: on birky-15 a straight road
34 m away scored 0.87, and the truth scored 0.48. A Gaussian prior penalises
distance from the INS position. A look-alike 34 m away then has to beat the
truth by a wide margin, while a 15 m truth still has a chance. We expect the
prior plus a round 20 m window to beat `007.1` at 15 m without losing at 5 and
10 m.

**B.** A bad correction usually has a recognisable heatmap. It is a ridge along
a road (position known across the road, not along it), two or more competing
peaks, or a flat map from a frame with little structure. The posterior over
heatmap cells has a covariance that captures all three: a ridge gives one long
axis, competing peaks give a long axis between them, and a flat map gives two
long axes. Gating on that covariance should remove most harmful corrections
and keep most good ones.

## Method

All of it lives on the existing weighted-NCC heatmap `z` (z-scored, per heading-up
cell). Offsets `d` are metres from the prior, which is the patch centre.

1. **Round window.** Size the satellite patch so that the drone centre can move
   `R` metres in every direction, after the 3% border crop:
   `patch = footprint + 2R` per axis. Cells with `|d| > R` are **excluded**.
   The z-score is computed over the included cells only. `reachable` becomes
   `|d_true| ≤ R`.
2. **Posterior.** `log p(cell) = z / τ − |d|² / (2σ²)`, normalised over the
   included cells. `σ` is the INS uncertainty and is fixed at **10 m** at every
   injected level. It is never set to the draw's injected error, which would
   leak the truth. The corrected position is the posterior **argmax** (MAP).
   With `σ = ∞` it equals today's argmax.
3. **Uncertainty.** `μ, Σ` are the posterior mean and covariance of `d`.
   `s1 ≥ s2` are the square roots of `Σ`'s eigenvalues (metres), and `e1, e2`
   the matching eigenvectors.
4. **Gate**, applied in this order:
   - drone weighted-class coverage (the sum of `drone_class_shares` over classes
     with weight > 0) `< c_min` → **reject**
   - `s1 ≤ s_max` → **accept**: move to the MAP
   - `s2 ≤ s_max < s1` → **across-only**: move by the MAP offset projected on
     `e2`, and keep the prior along `e1`
   - otherwise → **reject**: keep the prior

A reject keeps the prior, so its error is the injected error. That is counted in
every mean. Nothing is dropped.

### Tuned values: leave-one-place-out, grid fixed now

| Symbol | Grid | Used in |
|---|---|---|
| `τ` (temperature) | 0.25, 0.5, 1, 2 | posterior (A and B) |
| `s_max` | 2, 3, 5 m | gate |
| `c_min` | 0.01, 0.03, 0.05 | gate |

For each held-out place (bushcha, birky), pick values on the **other** place
only, then score the held-out place. Reported numbers are the two held-out
halves concatenated: 40 frames, each scored by values chosen without it.

- `τ` objective: lowest mean error @ 10 m on the tuning place.
- Gate objective: lowest harm share @ 10 m subject to accepted + across-only
  ≥ 50% of draws.

Log the chosen values per fold in `runs.jsonl`.

> **Note, 2026-09-28 (build).** Three points the grid above leaves open, fixed
> before any logged run:
> 1. *Order.* `τ` has one objective, so it is chosen first on mean error @ 10 m
>    with no gate, and the gate values are then chosen at that `τ`. A gated run
>    therefore shares its ungated twin's `τ` (`007.7`/`007.9` share `007.4`'s).
> 2. *Flat prior.* Where `σ = ∞` (`007.8`) `τ` cannot move the argmax, so the
>    mean-error objective ties on every `τ`; the tie is broken by the gate
>    objective, i.e. `τ` is chosen jointly with the gate.
> 3. *No feasible gate.* If no grid point keeps accepted + across-only ≥ 50%,
>    the fold takes the point with the highest such share, then lowest harm,
>    then lowest mean error.
>
> A dry run of the `007.7` config (not logged) showed point 1 matters: the
> bushcha tuning place picks `τ = 2`, where `s1` exceeds every `s_max` in the
> grid, so that fold's gate rejects everything. The pre-registered runs keep
> point 1. `007.10` is an extra, **not pre-registered** run of `007.7` with `τ`
> chosen jointly with the gate; it is reported as a diagnostic and does not
> decide Rule B.

## Success criterion

Primary level: **10 m injected**. It is the operating regime, since corrections
run every few seconds and drift should not reach 15 m. It is also where 55% of
draws are reachable today, so it measures matching rather than window size.
15 m is the secondary level for question A, because A is about reach.

Every level (5 / 10 / 15 m) is reported for every run. The CIs are 95% paired
frame-bootstrap intervals over 40 frames, 10,000 resamples, seed 42, as in 006.

**Harm** = the share of draws with `error_m > prior_error_m + 2 m` (the
correction made position worse by more than 2 m).

**Metric:** `mean_position_error_m` and `harm_share`
**Baseline:** `007.1`, a re-run of `006.1` (9.29 m @ 15, 5.86 m @ 10, 4.59 m @ 5)

**Rule A (search) passes** if `007.4` (round R = 20 m + prior):
- is ≥ **2.0 m** better than `007.1` @ 15 m, CI clear of 0, **and**
- is not worse @ 10 m or @ 5 m: the upper CI of `007.4 − 007.1` is < +0.5 m at
  both levels.

**Rule B (gate) passes** if `007.7` (`007.4` + gate):
- has harm @ 10 m ≤ **half** of `007.4`'s harm @ 10 m, **and**
- has accepted + across-only ≥ 50% of draws @ 10 m, **and**
- has mean error @ 10 m no more than **0.5 m** above `007.4`'s. Rejections cost
  mean error, because a reject scores the full injected error. The gate is
  allowed to cost a little mean error, not a lot.

## Pre-registered runs

Everything not listed is fixed at `006.1` / `005.5`: MobileNet LR-ASPP satellite
BN on both sides, weighted NCC with weights `[0, 1, 1, 0.5, 0, 0.5]`,
`map_date: newest`, FOV 60°, 3% satellite border crop, 40 frames × 8 directions,
5 / 10 / 15 m, seed 42.

| Run | Window | Prior σ | Gate | Asks |
|---|---|---|---|---|
| `007.1` | rect, scale 1.7 | ∞ | none | baseline; must reproduce `006.1` bit-exactly |
| `007.2` | rect, scale 3.0 | ∞ | none | reproduces `005.3` (11.63 m @ 15) on current code |
| `007.3` | round, R = 20 m | ∞ | none | window shape alone |
| `007.4` | round, R = 20 m | 10 m | none | **Rule A** |
| `007.5` | rect, scale 1.7 | 10 m | none | prior alone, current window |
| `007.6` | round, R = 30 m | 10 m | none | does the prior make an even bigger window safe? |
| `007.7` | round, R = 20 m | 10 m | full | **Rule B** |
| `007.8` | rect, scale 1.7 | ∞ | full | gate alone, current window |
| `007.9` | round, R = 20 m | 10 m | accept / reject only | value of across-only |

`τ` enters only the runs with a finite `σ` or a gate. Where `σ = ∞` and there
is no gate, the MAP is the plain argmax and `τ` does nothing.

New config keys, so that `log_run` records them: `window` (`rect` | `round`),
`search_radius_m`, `prior_sigma_m` (`null` = ∞), `temperature` (per fold),
`gate` (`none` | `full` | `accept-reject`), `gate_s_max_m`, `gate_c_min`
(per fold).

## Per-draw records (for the report and later Kalman work)

In addition to today's columns: `s1_m`, `s2_m`, ridge angle (from `e1`, degrees in
north-up coordinates), `coverage`, `second_peak_ratio` (best z outside 5 m of
the peak ÷ peak z, logged as a diagnostic, not used by the gate), `decision`
(`accept` | `across` | `reject`), `map_offset_m`, `applied_offset_m`,
`truth_reachable`, and `ms_per_correction` (CPU, median of 3 standalone
timings per run, for the Raspberry Pi budget).

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.correlator`: round window, posterior with prior, covariance, gate (new `CorrelatorConfig` fields, defaults = today's behaviour) | to build (code equipment: every later matching experiment reuses it) | with the defaults, `007.1` reproduces `006.1`'s per-draw errors exactly (max abs diff 0.0); unit checks: `σ = ∞` → MAP = argmax; a synthetic single Gaussian bump gives `s1 ≈ s2` ≈ the bump's width; a synthetic line gives `s1 ≫ s2` with `e1` along the line |
| `uavloc.matching.benchmark`: new per-draw columns and `harm_share` in `error_stats` | to build | existing metric keys unchanged for old runs |
| Leave-one-place-out tuning (glue in this folder) | to build | each fold's chosen values come only from the other place's records; logged per fold |
| `village-crossdate-frames`, `ua-village-multidate` newest map, LR-ASPP checkpoint | exist | — |

Start from `006-segmentation-ceiling/train.ipynb`. Run `007.1` first and stop if
it does not reproduce `006.1`. Use the same device as 006 (MPS) for
reproduction. Time `ms_per_correction` on CPU.

**What only you can provide.** Nothing.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Figures

- A risk-coverage curve @ 10 and 15 m: accepted share vs mean error of the
  accepted draws, sweeping `s_max`, for `007.7` and `007.8`.
- A heatmap gallery: two draws each of accept, across-only and reject, with the
  `s1/s2` ellipse drawn. It must include birky-15 at 15 m.
- Error vs injected level for `007.1`–`007.6`.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames` (query), `ua-village-multidate` newest date (map) |
| Split | test only; tuning is leave-one-place-out across bushcha / birky |
| Seed | 42 |
| Model | MobileNetV3-Large LR-ASPP, `model-lraspp-bn-satellite.pt`, both sides |
| Loss | none, no training |
| Hardware | MPS for runs (reproducibility with 006), CPU for timing |

## Known confounds, accepted up front

- **Fixed-magnitude, random-direction noise with a fixed σ = 10 m.** Real INS
  drift is correlated, and its σ grows with time. With a 5 m injected error the
  prior pulls toward a point 5 m off, which is correct Bayesian behaviour but
  can look like a loss at 5 m.
- **Two places, overlapping birky frames.** Leave-one-place-out tunes on one
  place and tests on a very different one. That is honest, but noisy: a
  threshold that transfers badly will show up as a miss, not as a tuning bug.
- **Not drone imagery.** Satellite crops on both sides, as in 005 and 006. The
  heatmap shapes on real drone frames may differ.
- **Per-draw independence.** Each draw is corrected alone. In flight, a reject
  or across-only move is followed by the next frame, and that follow-up is what
  makes gating cheap. This benchmark cannot show it, so it understates the
  gate's value.

## What would falsify this

- **Rule A misses.** The prior does not stop look-alikes at this footprint. The
  ambiguity is in the scene (a 46 × 26 m frame is not unique enough), not in the
  search. Next: a larger footprint or a multi-frame map before touching the
  window again.
- **Rule B misses on harm.** Heatmap shape does not predict bad corrections.
  Confidence has to come from elsewhere: learned, or from consistency across
  consecutive frames.
- **Rule B misses only on the mean-error guard.** The gate rejects too many good
  corrections. Keep the covariance as a Kalman measurement noise instead of a
  hard gate.

## Related

- `005-village-crossdate`: window vs look-alikes (`005.3`, birky-15), reach 10% @ 15 m.
- `006-segmentation-ceiling`: masks are worth ~1 m, edges ~0.4 m; the ~4 m floor @ 5 m holds even with GT.
- 2025 thesis gap: heatmap confidence was computed but never used to reject corrections.
