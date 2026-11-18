# 029-ins-prior-mismatch — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does 028's INS-error particle filter keep beating 027's white-noise particle filter when
its IMU prior is wrong by a factor of two, too narrow or too wide?

## Hypothesis

Yes, with a smaller margin when the prior is too narrow. 028's filter starts from priors
that equal the simulator's own error parameters, which a real IMU datasheet never gives
exactly. Two failure modes are possible:

- **Too narrow (×0.5):** a flight whose true bias is 2σ under the real spec is a 4σ draw
  under the filter's prior. Few or no particles start near it, so the cloud must reach it
  through resampling and the small `q_pos` jitter. That is 027's failure in a milder form.
  Expected: worse than 028, still better than 027 on average.
- **Too wide (×2):** the cloud starts 2× broader in bias, so it spreads faster between fixes
  and blind stretches cost more. But the heatmap prunes wrong biases after a few fixes.
  Expected: close to 028.

The filter is not re-tuned for the wrong prior, because a deployed system is tuned once
against what it believes its IMU is.

## Method

Exactly `028.1` (normal start, test flights, T and `q_pos` as tuned in 028), except that the
filter's `InsErrorPrior` is scaled by `PRIOR_SCALE`: velocity bias std, velocity random walk,
heading bias std and speed-scale std are all multiplied by it. The simulator is unchanged.

| Run | `PRIOR_SCALE` | Compared with |
|---|---|---|
| `029.1` | 0.5 (prior 2× too narrow) | `027.4` (bar), `028.1` (reported) |
| `029.2` | 2.0 (prior 2× too wide) | `027.4` (bar), `028.1` (reported) |

## Success criterion

Unit: one flight; 95 % paired bootstrap over villages, 10,000 resamples, seed 42.

**Metric:** `mean_end_error_m`, 94 test flights, normal start
**Baseline:** `027.4` (027's particle filter, white-noise motion model)
**Bar, for both runs:**
1. `mean_end_error_m` ≤ **0.8 ×** `027.4`'s, with the CI of the paired difference clear of 0.
2. `divergence_share` no higher than `027.4`'s.

Reported, not barred: each run vs `028.1` (paired difference and CI), truth-in-window and
truth-in-ellipse share, per-fix time.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| 028's filter, tuned values and notebook | built in 028 | `029` with `PRIOR_SCALE = 1` would be `028.1` (same code path; checked by construction, the scale multiplies the prior only) |
| `flights-027.4.csv`, `flights-028.1.csv` | written by 027 / 028 test runs | same 94 (village, route) keys |

**What only you can provide.** Nothing.

## Setup

As 028: `sim-flights` test split, LR-ASPP satellite BN both sides, weighted NCC, M1 CPU-only.

## Known confounds, accepted up front

- Every component of the prior is scaled together. A real spec is usually wrong in one term
  (often the bias instability). A per-term sweep is the follow-up if the ×0.5 run fails.
- The form of the error model still matches the simulator exactly; only its scale is wrong.

## What would falsify this

- **029.1 misses the bar:** the INS-error filter's gain depends on knowing the IMU spec well.
  A deployed version would need an inflated prior, or the bias learned online in a wider way.
- **029.2 misses the bar:** a too-wide prior spreads the cloud enough to lose the gain, so the
  prior has to be calibrated, not just bounded.

## Related

- `028-ins-error-state`: the filter and its confound ("the prior is the simulator's own
  error model").
- `027-sequence-filters`: the baseline `027.4`.
