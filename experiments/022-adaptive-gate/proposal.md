# 022-adaptive-gate — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a gate whose decision comes from the frame's own heatmap shape, with
the sharpness scale read at several temperatures and the accept threshold
set as a harm probability rather than in metres, cut harm as far as `007.8`'s
static gate does at a much smaller cost in mean error, and does that hold on a
place it was not fitted on?

## Hypothesis

`007.8` cut harm @ 10 m from 0.084 to 0.022 but cost +1.00 m of mean error. Its
`s_max` = 3 m and τ = 0.25 are constants in metres and in z units, and 007 showed
neither transfers: tuned on birky, bushcha gets τ = 0.25; tuned on bushcha,
birky gets τ = 2.0 (`007.7`). The follow-up in `builds/corrector-identity-bench`
(`gate-roc.ipynb`) adds that `s1` is the best single statistic (AUC 0.717), that
a logistic combination of five statistics reaches 0.767 with false alarms cut
from 36.5 % to 30.2 %, and that the 3 m threshold is near-optimal for `s1`
(retuning buys nothing).

Mechanism: a threshold in metres has to absorb the difference in heatmap scale
between scenes, so it rejects good sharp-but-wide matches and accepts nothing
that looks the same at a different scale. A probability of harm learned from
the shape at several τ (how fast `s1`/`s2` collapse as τ shrinks tells a real
peak from a plateau) is the same number in both places by construction. If it
is also calibrated across places, the threshold `p_max` is a single constant
that transfers and the cost of the gate falls.

What this does **not** claim to fix: a single confident wrong peak. 007 and the
gate-roc follow-up both show `p90` and `max` are identical with and without the
gate, because those draws have small `s1`. Whether shape at several τ sees them
is part of what is measured.

Search window and prior σ are **out of scope**. Per-draw injected error and
altitude are not inputs either. The real system does not know the injected error,
and every frame of `village-crossdate-frames` is at 40 m, so altitude cannot be
varied here. Adaptive window and σ from the filter's own covariance belong to
`008-particle-filter-sequence`.

## Method

Everything runs on the `007.1` / `007.8` setup: rect window, scale 1.7, flat
prior, weighted NCC, MobileNetV3 LR-ASPP satellite BN on both sides, map date
newest, 40 frames × 8 bearings × 5 / 10 / 15 m, seed 42. Surfaces are computed once
with `benchmark_draws`; every arm is scored from the same surfaces.

**Decision.** A draw whose heatmap is flat, or whose drone coverage is below
`c_min` = 0.01, keeps the prior, as in 007. Otherwise the arm decides
accept (move to the argmax) or reject (keep the prior). There is no across-only
move: `007.9` vs `007.7` found it adds 0.12 m @ 10 m.

**Features** (`uavloc.matching.risk.gate_features`). Computed from the
`SearchSurface` and the config only. None uses truth, place, noise level, bearing
or frame name.

| Feature | Taus |
|---|---|
| `log s1_m`, `log s2_m` of the posterior | 0.25 (all arms) and 0.5, 1, 2 (adaptive arm) |
| `peak_z` | — |
| `second_peak_ratio` | — |
| `coverage` | — |

**Risk model.** L2 logistic regression (C = 1, standardised features) predicting
`harm` = `error_m > prior_error_m + 2` (007's definition). It is fitted on the
other place's draws, all three noise levels pooled. The noise level is not a
feature.

**`p_max` is chosen on the fitting place only**, by 5-fold cross-validation grouped
by frame (so a frame's 24 draws never sit in both halves). Grid
`{0.05, 0.10, 0.20, 0.30}`. Objective: lowest mean error @ 10 m among the values
whose harm @ 10 m ≤ 0.042 and whose applied share ≥ 0.60; if none qualifies, the
highest applied share.

**Leave-one-place-out.** Bushcha is scored with a model fitted on birky, and birky
with one fitted on bushcha. The reported numbers are the two held-out halves
joined: 40 frames, each scored by values chosen without its place.

### Arms

> **Note, 2026-10-19 (build, before any run).** `007.8`'s gate values were tuned
> leave-one-place-out on the 007 grid: `s_max` 3 m for bushcha and 2 m for birky, not
> one 3 m. So `022.1` and `022.2` take their values from the same procedure
> (`uavloc.matching.tuning.tuned_config`, moved from 007's `lopo.py`; grid τ {0.25, 0.5, 1, 2} ×
> `s_max` {2, 3, 5} × `c_min` {0.01, 0.03, 0.05}, applied share ≥ 0.5, tuned @ 10 m)
> instead of the fixed 3 m written above. The success criterion is unchanged.


| Run | Gate | Taus | Asks |
|---|---|---|---|
| `022.1` | static `full`, values tuned leave-one-place-out as in 007, with across-only | 0.25 | must reproduce `007.8` per draw |
| `022.2` | static `accept-reject`, values tuned the same way, no across-only | 0.25 | like-for-like baseline for the risk arms |
| `022.3` | risk model, `p_max` | 0.25 | learned threshold alone, no adaptive τ |
| `022.4` | risk model, `p_max` | 0.25, 0.5, 1, 2 | **the idea**: sharpness read at several scales |

`007.1` (no gate) is the ungated reference and is re-scored from the same surfaces.

### Diagnostic, no bar

How far the static gate's best values differ by place and by injected level: the
`TuningGrid` of 007 (τ × `s_max` × `c_min`) scored per place and per level, to put
a number on "the optimum depends on the place" before the arms run. Reported in
the report as a table.

## Success criterion

Primary level 10 m injected, as in 007. All levels are reported. The CIs are
95% paired frame-bootstrap over 40 frames, 10,000 resamples, seed 42.

**Metric:** `mean_position_error_m` and `harm_share` @ 10 m, with the applied share
**Baseline:** `007.1` (5.861 m, harm 0.084, applied 0.75 ceiling) and `007.8`
(6.866 m, harm 0.022, applied 0.691), both from `runs.jsonl`

**Bar (`022.4`), all of:**
1. harm @ 10 m ≤ **0.042** (half of `007.1`'s, the same ratio as 007 Rule B).
2. applied (accept) share @ 10 m ≥ **60 %**.
3. mean error @ 10 m ≤ **6.36 m**, i.e. no more than +0.5 m over `007.1`.
4. `022.4 − 022.2` mean error @ 10 m has its upper CI < 0.

`022.3` against `022.4` isolates the adaptive τ, and `022.1` against `022.2` prices
across-only. Both are reported, neither decides the bar.

Also reported for every arm: `p90` and max error, the harm draws that carried
`s1 ≤ 3 m` (the confident wrong peaks), and each held-out place on its own.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.risk`: `gate_features(surface, config, taus)`, `RiskGate` (fit / `harm_probability`), `score_risk(draws, config, gate, p_max)` returning `score_draws`' columns | to build (reused by any later gate or filter measurement noise) | (a) `022.1` reproduces `007.8`'s per-draw errors exactly (max abs diff 0.0); (b) `gate_features` at τ = 0.25 equals `s1_m` / `s2_m` in `records-007.8.csv`; (c) its signature takes a surface and a config only; (d) synthetic bump gives `s1 ≈ s2`; a line gives `s1 ≫ s2` at τ ≤ 0.5 and its ratio falls as τ grows (amended 2026-10-19 at build: at large τ a ridge washes out in a bounded window, which is what the multi-τ features read) |
| Frame-grouped CV and LOPO folds | to build (glue in this folder) | each fold's `p_max` and model come only from the other place's draws; both logged per fold |
| `village-crossdate-frames`, `ua-village-multidate` newest map, LR-ASPP checkpoint, `benchmark_draws` / `score_draws` | exist | — |

Start from `007-search-prior-and-gate/train.ipynb`. Run `022.1` first and stop if
it does not reproduce `007.8`. Use the same device as 007 (MPS).

**What only you can provide.** Nothing.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames` (query), `ua-village-multidate` newest date (map) |
| Split | leave-one-place-out (bushcha, birky); `p_max` by 5-fold CV grouped by frame |
| Seed | 42 |
| Model | MobileNetV3-Large LR-ASPP satellite BN on both sides (frozen); risk model is logistic regression |
| Loss | none for the segmenter; log-loss for the logistic regression |
| Hardware | MPS for surfaces, CPU for the risk model |

The risk model is a logistic regression fitted in one shot, with no step axis.
There are no training curves or segmentation grids to put in TensorBoard.
The run ends with `log_run`, and the fitted weights per fold go in `artifacts/`.

## Known confounds, accepted up front

- **Two places, 40 frames.** A fold fits on 20 frames. The bootstrap interval is
  too narrow because the 8 bearings and 3 levels of one frame are correlated,
  and several birky frames overlap.
- **`p_max` is still a constant.** It is a probability rather than a length, which
  is the claim, but the claim is only tested on two folds.
- **Same frames as 007's tuning.** The static baseline's values (`007.8`) were
  tuned on these frames. The risk arms have not seen the test place, so the
  comparison favours the baseline, if anything.
- **Not drone imagery.** Query and map are both satellite crops of different dates,
  as in 005–007.
- **One model.** The seg models of 021 are not used here. Applying the fitted gate
  to their surfaces is a follow-up, not part of this bar.

## What would falsify this

- **`022.4` misses bar 1 or 3:** a single frame's shape does not separate good
  from bad accepts better than `s1` alone. Stop gating per frame; the covariance's
  only use left is as measurement noise in `008`.
- **`022.3` is within CI of `022.4` on mean error and harm:** several τ add nothing.
  The learned threshold is the whole effect, or there is none.
- **The bar is met but `p90` and max are unchanged and the harm draws are the
  `s1 ≤ 3 m` ones:** the gate still cannot see a confident wrong peak. Single-frame
  gating is exhausted; only a sequence can reject it.
- **One held-out place worse than ungated by more than 1 m @ 10 m:** the probability
  does not transfer between places, which is the property the idea rests on.

## Related

- `007-search-prior-and-gate`: the static gate (`007.8`), τ that does not transfer, the
  `007.10` joint-tuning diagnostic.
- `builds/corrector-identity-bench` (`gate-roc.ipynb`): AUC of `s1` 0.717 vs a logistic
  combination 0.767; `p90` and max unchanged by the gate; the 3 m threshold is
  near-optimal.
- `008-particle-filter-sequence`: adaptive window and σ from the filter's covariance;
  covariance as measurement noise instead of a gate.
- `006-segmentation-ceiling`, `021-building-loss-from-scratch`: masks are not the
  bottleneck; the frames and the model checkpoint.
