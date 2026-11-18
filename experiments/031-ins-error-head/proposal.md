# 031-ins-error-head — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

With 028's INS-error particle filter, does scoring the window with the 023 correlation head
(`023.3`) instead of weighted NCC end closer to the truth on flights over villages the head
never trained on?

## Hypothesis

Yes. The two changes fix different things. 028 fixes the motion model; the head fixes the
per-frame score. 023 found the head ranks positions better than weighted NCC where there is
structure (holdout −0.83 m, CI clear of 0) but loses its single-frame bar because it moves on
every draw. A filter does not need a "keep the prior" decision, so the head's weakness goes
away and its better ranking should show. In 027's head tuning, the white-noise particle
filter's tuning error fell from 22.1 m (NCC) to 6.6 m (head), with half the tuning villages
seen by the head in training.

## Method

`028.1`'s setup (normal start, 94 test flights, INS-error filter, `q_pos` = 0.3), with
`matching = "learned"` through `023.3`'s head (the same `LearnedScorer` and checkpoint that
027.11–13 use). Only T is re-tuned (the head's z-scale differs from NCC's): T ∈ {0.25, 0.5, 1}
on the 20 tuning flights, lowest `mean_end_error_m`.

| Run | Likelihood | Compared with |
|---|---|---|
| `031.1` | 023.3 head | `028.1` (NCC), paired by route |

**Clean subset.** The head trained on 46 of these villages. Its 10 val villages and bushcha
are clean: 6 test villages, 12 flights. The bar is read on those 12 flights; all 94 are
reported.

## Success criterion

Unit: one flight; 95 % paired bootstrap over villages (both routes together), 10,000
resamples, seed 42.

**Metric:** `mean_end_error_m` on the 12 clean test flights, normal start
**Baseline:** `028.1` on the same 12 flights
**Bar, all of:**
1. Paired difference (031.1 − 028.1) on the clean flights < 0 with the CI's upper end < 0.
2. On all 94 test flights, `mean_end_error_m` no higher than `028.1`'s (point estimate).
3. Median per-fix time ≤ 1.0 s on the M1 CPU.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| 028 filter, `uavloc.simulation.arms.build_correlator(head=...)` | built (027 head arms use it) | 027.11–13 ran through it |
| `flights-028.1.csv` | written by 028.1 | 94 keys |

**What only you can provide.** Nothing.

## Setup

As 028; likelihood = 023.3 correlation head on LR-ASPP probability maps.

## Known confounds, accepted up front

- **Six clean villages.** The bootstrap over 6 villages is coarse; a real effect of ~1 m may
  not clear it. That is a power problem, recorded as such if bar 1 misses with a negative
  point estimate.
- **T is tuned on villages half seen by the head**, which favours the head in tuning only.
- The head was trained on 007.1-sized windows; the filter's windows are larger in blind
  stretches. Its z-scores there are out of its training distribution.

## What would falsify this

- **Point estimate ≥ 0 on the clean flights:** the head's ranking gain does not survive a
  sequence; NCC is enough once the motion model is right.
- **Bar 2 missed but bar 1 met:** the head helps where it generalises and hurts on its own
  training villages (overfit to their imagery), which would be strange and worth a look.

## Related

- `023-learned-correlation-head` (`023.3`), `028-ins-error-state` (`028.1`),
  `027-sequence-filters` (027.11–13: the head with the white-noise filter).
