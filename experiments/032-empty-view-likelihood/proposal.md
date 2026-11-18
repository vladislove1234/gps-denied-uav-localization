# 032-empty-view-likelihood — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does 027.12's particle filter (023 head likelihood) end closer to the truth if blind frames
update it, instead of being skipped? A blind frame is one where the drone view has no road,
building, tree or water pixels. The update says "the drone is over a place the map shows as
empty".

## Hypothesis

Yes. Most of 027.12's large end errors come from drift during blind stretches, not from
losing the truth:

- 10 of 94 test flights end more than 20 m off. On 7 of them the window held the truth on
  every fix (`truth_in_window_share` = 1.00), and the ellipse held it on 88–100 % of fixes.
- 6 of the 10 skip 56–75 % of their fixes as blind (the test set as a whole: 28 %). Across a
  blind stretch the error grows by about 3–4 m per 5 s fix: INS drift with nothing to stop
  it (`fixes-027.12.csv`, e.g. rokety route 1: 6 → 48 m over 11 blind fixes). Three of the
  10 (railiv 1, rudnia-horodyshche 1, zarichanka 1) skip only 6–35 %, so this does not
  explain them.
- Only 2–3 flights really lose the truth: kosteiv route 0 (window 0.43) and ruda route 1
  (0.75). Re-seeding would fix only those, so it is not this experiment.

A blind view is still evidence. If the map under a particle's footprint has a road or a
house, that particle is unlikely to see an empty frame. The cloud then stays on the fields
and off the villages and roads, which narrows it at the edges of blind stretches. In robot
localisation this is called *negative information* (recalled, not checked here: Hoffmann et
al. 2005, Monte Carlo localisation with negative information).

## Method

**Empty-view likelihood.** On a blind frame (`weighted coverage < 0.01`, the existing 007
test), for every offset in the window that already covers the cloud:

1. `c_map` = share of satellite pixels under the drone footprint whose segmentation argmax
   is a weighted class (building, track, woody, water). This is the same count `coverage`
   makes on the drone frame, made on the map. One box filter per frame, on the heatmap's grid.
2. Each particle's weight is multiplied by `P(blind | c_map)^β`.
   - `P(blind | c_map)` is a logistic fit made once, on the tuning villages only, from
     frames whose true footprint is known (simulation).
   - It is floored at `MIN_BLIND_PROB = 0.01`, so one segmentation error can't zero a
     particle.
   - `β` makes up for consecutive frames not being independent.
3. ESS check and systematic resampling, exactly as for a normal update.

Frames that are blind because the heatmap is flat (coverage ≥ 0.01) are still skipped.
Everything else is 027.12: N = 2000, T = 0.5, q = 2.0, head `023.3`, same window, same
seeds.

**Tuning.** `β ∈ {0, 0.5, 1, 2}` on the 20 tuning flights; pick the lowest
`mean_end_error_m`. `β = 0` must reproduce 027.12's tuning number (check c).

| Run | Likelihood | Empty view | Compared with |
|---|---|---|---|
| `032.1` | 023.3 head | on (tuned β) | `027.12`, paired by route (**bar**) |
| `032.2` | weighted NCC | on (same β) | `027.4`, paired by route (reported) |

`032.2` repeats the test without the head. The head trained on 41 of the 47 test villages;
the empty-view model has no learned part. So if 032.2 shows the same gain, the gain is not
the head's.

## Checks (before tuning, on tuning villages only)

| | Check | Pass |
|---|---|---|
| a | `c_map` grid has the heatmap's shape, for NCC and for the head | equal on every frame tried |
| b | `c_map` at the true footprint predicts a blind frame | ROC AUC ≥ **0.75** |
| c | `β = 0` reproduces 027.12's tuning result | `mean_end_error_m` = 6.5997 m (from `runs.jsonl`), \|Δ\| < 1e-6 |

If (b) fails, an empty view says little about where the drone is in this simulator. The
experiment stops there and reports that.

## Success criterion

Unit: one flight. 95 % paired bootstrap over villages (both routes together), 10,000
resamples, seed 42.

**Metric:** `mean_end_error_m` over the 94 test flights, normal start
**Baseline:** `027.12` (PF + head): **10.027 m**, divergence 0.0106, truth in window 0.9845,
median fix 251 ms (`027-sequence-filters/runs.jsonl`)
**Bar (on `032.1`), all of:**
1. Paired difference (032.1 − 027.12) < 0, with the CI's upper end < 0.
2. `divergence_share` ≤ 027.12's (0.0106).
3. `truth_in_window_share` ≥ **0.98** (an overconfident empty-view update would push the
   cloud off the truth).
4. Median per-fix time ≤ 1.0 s on the M1 CPU.

Reported, not barred:
- 032.2 vs 027.4, with the same bootstrap.
- `mean_error_m` over the whole flight.
- Mean error on blind fixes only.
- 027.12's 10 worst flights, before and after.
- The 12 clean-village flights.
- `truth_in_ellipse_share`.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc/filtering/empty_view.py`: `map_coverage_grid`, `EmptyViewModel` (logistic + floor), `fit_empty_view` | to build | checks a, b; a figure of one blind frame next to its `c_map` grid and the truth |
| `ParticleFilter(empty_view=, empty_view_weight=)` | to build (optional arguments, default off) | check c: off = 027.12 exactly |
| `arms.estimator_factory` / `FilterParams` pass-through | to build | 032.1 config records β and the fitted coefficients |
| 027 head setup (`build_correlator(head=...)`), `sim-flights` | built | 027.12 ran through it |
| `flights-027.12.csv`, `flights-027.4.csv` | written by 027 | same 94 keys |

**What only you can provide.** Nothing.

## Setup

As 027.12:
- `sim-flights` test split; drone view = `current`, map = newest Wayback.
- LR-ASPP satellite BN on both sides; head `023.3`.
- M1 CPU only, at most 3 runs in parallel (bug-159).

## Known confounds, accepted up front

- **The simulator favours this idea.** Drone frames are rendered from another satellite
  date and segmented by the same satellite model as the map. So "empty on the drone" and
  "empty on the map" agree more often than they would with a real camera. A real-data check
  is the follow-up if this works.
- **β is tuned on villages half seen by the head.** The empty-view model is not learned, and
  β's effect is read against a baseline with the same head.
- **End error is one point in time.** A flight that ends in a big field can only be bounded
  to that field. `mean_error_m` is reported for that reason.
- **Independence.** Consecutive frames overlap little at 10 m/s every 5 s. β is the only
  correction for any dependence left.

## What would falsify this

- **Check b fails:** an empty view says little about where the drone is. Stop.
- **Point estimate ≥ 0:** the information is real but too weak against drift. Fields are
  too big, so the cloud is already inside the empty area. The next lever is then the motion
  model during blind stretches (028's idea, without its divergences) or route planning.
- **Bar 1 met, bar 3 missed:** the update is overconfident, and the gain comes with lost
  consistency. Read β's tuning curve.

## Related

- `027-sequence-filters` (`027.12` baseline, `027.4`): blind frames are skipped there.
- `028-ins-error-state`: the other attack on blind-stretch drift (missed its bar on divergence).
- `007-search-prior-and-gate`: the `coverage < 0.01` blind test.
- `023-learned-correlation-head` (`023.3`).
