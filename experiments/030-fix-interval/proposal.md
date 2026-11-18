# 030-fix-interval — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

How much does 028's INS-error particle filter lose when a fix comes every 10 s or 20 s
instead of every 5 s, and does it lose less than today's replace correction?

## Hypothesis

It loses little at 10 s and keeps a growing lead over replace. Two effects pull apart:

- **Fewer looks.** At 10 m/s a 46 × 26 m frame every 5 s already covers only ~half the
  route; at 20 s it covers ~13 %. Fewer frames means fewer junctions and buildings seen.
- **The drift between looks.** For the replace arm the drift between fixes is uncorrected
  INS drift and its 007.1 window reaches only ~±9 m across the frame, so a longer interval
  pushes the truth out of reach more often. The INS-error filter predicts the drift from
  the bias it has learned, so its cloud grows slowly and its window follows it.

The 2025 system corrected every ~10 s, and a Raspberry Pi gains compute from a longer interval.

## Method

`028.1`'s setup (normal start, the 94 test flights, the filter's T = 0.5 and `q_pos` = 0.3
tuned in 028, not re-tuned), with `correction_interval_s` changed. The INS realisation per
route is the same at every interval (the INS simulator draws its drift on its own stream).

| Run | Estimator | Interval |
|---|---|---|
| `030.1` | INS-error particles | 10 s |
| `030.2` | INS-error particles | 20 s |
| `030.3` | replace (007.1) | 10 s |
| `030.4` | replace (007.1) | 20 s |

The 5 s references are `028.1` (INS-error particles) and `027.2` (replace).

## Success criterion

Unit: one flight; 95 % paired bootstrap over villages, 10,000 resamples, seed 42.

**Metric:** `mean_end_error_m`, 94 test flights, normal start
**Baselines:** `028.1` (same filter at 5 s) and `030.3` (replace at 10 s)
**Bar (on `030.1`, the 10 s filter), all of:**
1. `mean_end_error_m` ≤ **1.25 ×** `028.1`'s.
2. `mean_end_error_m` ≤ **0.8 ×** `030.3`'s, with the CI of the paired difference clear of 0.

Reported, not barred: `030.2` against the same references; the filter's lead over replace at
5 / 10 / 20 s; truth-in-window share; divergence share.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| 027/028 sequence equipment, `sim-flights` | built | — |
| `FlightSetConfig.correction_interval_s` override in the notebook | trivial (`dataclasses.replace`) | the INS track per route is identical across intervals (checked in the run: `end_error_ins_m` equal to `028.1`'s per flight) |
| `flights-028.1.csv`, `flights-027.2.csv` | written by 027 / 028 test runs | same 94 keys |

**What only you can provide.** Nothing.

## Setup

As 028.

## Known confounds, accepted up front

- T and `q_pos` were tuned at 5 s. A longer interval might prefer a sharper T (each look
  must count for more). Not re-tuning keeps the comparison "same system, fewer looks".
- The replace arm's 007.1 window is fixed in size; at long intervals it is mostly a reach
  test. That is today's system as it is.

## What would falsify this

- **Bar 1 missed:** the filter needs frequent looks; the bias it learns does not carry it
  10 s. Then the Pi budget should go into a 5 s interval, not a slower but heavier matcher.
- **Bar 2 missed:** at a realistic interval the filter's advantage over replace is small;
  the 027/028 gains were partly the 5 s cadence.

## Related

- `028-ins-error-state` (`028.1`), `027-sequence-filters` (`027.2`).
- 2025 thesis: correction every ~10 s.
