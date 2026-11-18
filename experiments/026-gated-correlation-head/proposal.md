# 026-gated-correlation-head — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.
>
> Approved by the author 2026-11-01 as drafted (holdout primary, bar, primary-bench guard).

## Question

Under the same static accept-reject gate, tuned by the same procedure, does 023's trained
correlation head place the drone closer to the truth than weighted NCC, without more harm?

## Hypothesis

023 found that the head ranks positions better where the query holds structure: on the 10
unseen val villages it is 1.38 / 1.14 m better than NCC on covered draws (`023.3` / `023.5`,
CIs clear of 0). It failed its bar because it never abstains: it applies 100 % of draws,
harm doubles (0.084 → 0.144 / 0.166 primary, 0.109 → 0.186 / 0.189 holdout), and p90 goes from
10 to 15–16 m.

Abstaining is what 022.2's static accept-reject gate does for NCC: it rejects frames
without weighted classes (`c_min`) and frames whose posterior is spread out (`s_max_m`),
which takes harm from 0.084 to 0.006 at a cost of +0.14 m mean. If the head's gain comes
from draws with structure and a clear peak, the gate keeps those draws and drops the ones
where the head was guessing. The gated head should then beat gated NCC by most of its
covered-draw gain, at gated-NCC harm.

What would kill it: the head's gain sits in exactly the ambiguous draws the gate rejects
(look-alikes on bushcha), or the head's posterior is sharp where it is wrong, so the gate
cannot tell its good moves from its bad ones.

## Method

Everything as `025.1` unless stated: `village-crossdate-frames` (primary bench, 40 frames)
and `village-crossdate-holdout` (200 frames, 10 villages), 5 / 10 / 15 m × 8 directions,
seed 42, rect ×1.7, flat prior, LR-ASPP satellite checkpoint, newest map date.

**Matchers.** Weighted NCC (hand weights) and the two trained 023 heads, loaded from
`023-learned-correlation-head/artifacts/head-023.3.pt` and `head-023.5.pt` through
`LearnedScorer`. No training in this experiment.

**Gate.** `accept-reject` (022.2), tuned with `uavloc.matching.tuning.tuned_config` on the
022.2 grid: temperature {0.25, 0.5, 1, 2}, `s_max_m` {2, 3, 5}, `c_min` {0.01, 0.03, 0.05},
tuned at 10 m, `min_applied_share` 0.5, `temperature_by` mean-error. The tuning procedure
is the same for every matcher:

- primary bench: leave-one-place-out over birky / bushcha (as 022.2);
- holdout bench: tuned on both primary places pooled (as 025.1). No value is tuned on a
  holdout village.

`c_min` measures the hand-weighted classes' share in the drone map, for every matcher, so
"blind" means the same thing in every arm.

### Runs

| Run | Matcher | Gate | Asks |
|---|---|---|---|
| `026.1` | weighted NCC | accept-reject, tuned as 025.1 | control: must reproduce `025.1` |
| `026.2` | head `023.3` (seed 42) | same | **the idea** |
| `026.3` | head `023.5` (seed 7) | same | seed check |

## Success criterion

**Primary bench: the holdout** (200 frames, 10 villages). 023 showed that the 40-frame
primary bench cannot resolve a gain of this size: its CIs are about ±0.8 m on covered draws.
The holdout gate is tuned without seeing a holdout village, so the bench stays clean.

CIs are 95 % paired frame bootstrap, 10,000 resamples, seed 42, against `026.1`'s per-draw
records.

**Metric:** `holdout_mean_position_error_m` @ 10 m injected, all draws
**Baseline:** `025.1` → **7.742 m**, harm 0.0344, accept 0.439 (`025-a-contrario-gate/runs.jsonl`), re-run as `026.1`
**Bar (`026.2` and `026.3`, each):**

1. Holdout: arm − `026.1` @ 10 m ≤ **−0.5 m** and its upper CI < 0.
2. Holdout: `harm_share` @ 10 m ≤ `026.1`'s (`025.1`: 0.0344).
3. Primary bench (LOPO gate), no contradiction: arm − `026.1` @ 10 m point estimate ≤ 0, and
   `harm_share` @ 10 m ≤ 0.0125 (4 of 320 draws; `026.1`: 0.00625, 2 of 320).

**Yes** needs both seeds.

`026.1` must reproduce `025.1`: primary mean @ 10 m 5.99884 m ± 0.001 with harm 0.00625, and
holdout mean 7.74157 m ± 0.001 with harm 0.03438. Stop if it does not.

Also reported: accept share, 5 / 15 m, covered / blind, bushcha / birky, the 10 holdout villages
one by one, p90 / max, the tuned gate values per fold, the ungated head (023) next to the gated
one, and `ms_per_correction`.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `PositionCorrelator` with `matching="learned"` plus a gate | to change: `surface()` sets `coverage` to NaN for the learned scorer (`correlator.py:484`), so `coverage < c_min` is never true and the gate would never reject a blind frame. Compute `weighted_coverage(drone, class_weights)` for it as for weighted NCC | (a) head with gate `none` reproduces 023.3's and 023.5's per-draw records exactly (ungated decisions ignore coverage); (b) `surface.coverage` is identical under NCC and the head for every draw |
| `uavloc.matching.tuning.tuned_config`, `benchmark_draws` / `score_draws`, `paired_frame_bootstrap` | exist | (c) `026.1` reproduces `025.1` |
| `LearnedScorer`, `head-023.3.pt`, `head-023.5.pt`, both benches | exist | — |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-holdout` (primary), `village-crossdate-frames` |
| Seed | bench 42; heads from 023 seeds 42 and 7 |
| Model | frozen LR-ASPP satellite BN + frozen 023 heads |
| Hardware | M1 Pro, one run at a time (keeps `ms_per_correction` clean) |

## Known confounds, accepted up front

- **The holdout villages were 023's training monitor.** Their pairs were logged as `val/*`
  during training; no checkpoint was selected on them. The heads have still never trained on
  them.
- **The gate grid was set for NCC surfaces.** The head's z-scored logits may want a temperature
  outside {0.25 … 2}. The grid is kept so the comparison is like-for-like; a tuned value at the
  grid edge is reported as such.
- **Gate tuned on 2 places** (bushcha / birky), as in 022 and 025. 022 showed the tuned values
  differ between them.
- **Same generator for training and test, not drone imagery** (as in 023).

## What would falsify this

- **Both seeds miss bar 1:** the head's ranking gain lives in the draws the gate rejects.
  Gated NCC is as good, and the head is not worth carrying. Go to `008-particle-filter-sequence`.
- **Bar 1 met, bar 2 missed:** the head is overconfident where it is wrong, and its posterior
  spread does not separate good moves from bad ones. A confidence output trained with the head
  comes next, not another gate.
- **The gain sits on 1–2 holdout villages:** it is about those scenes, not the matcher.

## Related

- `023-learned-correlation-head`: the heads, and the ungated result this builds on.
- `022-adaptive-gate` (`022.2`, the gate), `025-a-contrario-gate` (`025.1`, the holdout baseline;
  per-frame significance gates lose to the static gate).
- `024-class-compatibility-matching`: moving on blind frames hurts.
