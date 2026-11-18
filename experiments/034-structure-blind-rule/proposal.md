# 034-structure-blind-rule — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does the particle filter + head (`027.12` configuration) end within 10 m of the truth on more test flights
if a drone frame with no structure counts as blind and does not update the filter?

## Hypothesis

Today a frame is blind only when weighted classes cover < 1 % of it or the surface is flat. Woody has
weight 0.5, so a frame of solid forest passes as a full measurement. Its NCC surface has no real peak,
and the filter moves towards a random one.

Evidence from `033` (8,652 updates over four arms, `fixes-033.*.csv`):
- Updates on frames 90–100 % covered by weighted classes gain only +0.65 m per update net (2,816 m
  moved away, 3,947 m moved closer), against +2.0 to +2.6 m in every other coverage band.
- Harmful updates (error +5 m or more) have median coverage 0.73. Helpful ones have 0.41.
- `bushcha` route 1, `033.1`: the first loss is an update on a frame segmented as all woody
  (9.9 → 24.7 m at t = 20 s, `figures/videos/bushcha-1-control-t20s.jpg`). The filter never recovers
  (108 m at the end).

Skipping such frames should prevent some lock-ons on false peaks. The cost is fewer updates where a
forest edge or a clearing was a real cue, so the two rules below differ on exactly that.

## Rules (fixed now, not tuned)

Computed on the drone frame's class map (argmax shares), after the existing blind test:
- **R1 `one-class`:** blind if the most common class covers **≥ 0.95** of the frame. A forest–field
  edge stays a measurement.
- **R2 `no-anchors`:** blind if building + track + water cover **< 0.02** of the frame (≈ 24 m² of a
  46 × 26 m footprint, about a quarter of a house). A forest–field edge alone is blind.

Temperature stays at 0.5 (the value every `027`/`033` arm chose). It is not retuned, so the rule is
the only change.

## Success criterion

Three independent segmenter + head pairs, no training. The rule must help on all three, so the
seed-to-seed noise that sank `033` (3.24 m between two CE + Dice seeds) cannot pass as an effect:

| Replicate | Segmenter | Head | Baseline run |
|---|---|---|---|
| A | `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` | `023.3` | `027.12` |
| B | `033` control | `033` head-control | `033.1` |
| C | `033` noise | `033` head-noise | `033.4` |

Runs: `034.1`–`034.3` = no rule (A, B, C), `034.4`–`034.6` = R1, `034.7`–`034.9` = R2. Paired against
the same-replicate no-rule run of this experiment (same code, same thread count). 94 test flights of
`sim-flights`, normal start. CIs: 95 % paired village bootstrap, 10,000 resamples, seed 42.

**Metric:** `recovered_share`: share of flights ending ≤ 10 m from the truth.
**Baseline:** `034.1`–`034.3`. They must reproduce `027.12` / `033.1` / `033.4` (recovered_share 0.777 /
0.745 / 0.713), see acceptance check.
**Bar (per rule), all of:**
1. Mean over the three replicates of Δ `recovered_share` ≥ **+0.05** (about 5 of 94 flights).
2. Δ `recovered_share` ≥ 0 in **every** replicate.
3. Guard: `median_end_error_m` not higher than the replicate's no-rule run by more than **0.5 m**, in
   every replicate.

**Reported, not deciding:** mean end error, divergence share, skipped share, number of flights > 20 m,
paired village-bootstrap CIs of Δ recovered and Δ mean end error per replicate, the bushcha-1 and
kryve-1 end errors.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.filtering.belief.BlindRule`, `WindowConfig.blind_rule`; `FilterParams.blind_rule` in `uavloc.simulation.arms` | built (this proposal, before any run) | unit check: all-woody shares blind under both rules, a village mix under neither, a woody/field edge only under R2 (done); with `blind_rule=None` the code path is unchanged |
| `train.ipynb` (this folder): `033`'s run notebook with `REPLICATE`, `BLIND_RULE`, `BLIND_THRESHOLD`, fixed T | adapt | `034.1`–`034.3` reproduce their baseline runs' `flights-*.csv`: ≥ 90 of 94 flights within 0.01 m. Thread-count nondeterminism moved one `033` flight (kryve-1, structure) by 29 m, so a few chaotic flights may differ. More than 4 mismatches means the comparison is not paired and stops the experiment |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `sim-flights` test split, 94 flights / 47 villages |
| Split | as `027` |
| Seed | filter seed per route as `027`; bootstrap 42 |
| Model | three frozen segmenter + head pairs (table above), filter A, T 0.5, q 2.0 |
| Loss | — (no training) |
| Hardware | M1 Pro CPU, 2 threads per run (as `033`) |

## What would falsify this

- Both rules miss bar 1, or one replicate goes down: frames of one class are not what drives the
  failures, or skipping them costs as much as it saves.
- R2 helps less than R1, or hurts: forest edges carry real position, and "no buildings or roads" is too
  strict.
- Skipped share rises a lot while recovered share does not move: the filter loses more by waiting than
  by reading weak frames. The next step would then be 032's negative information rather than skipping.

## Related

- `033-class-loss-in-system`: the update-by-coverage analysis and the failure videos that motivate this.
- `027-sequence-filters` (`027.12`), `023-learned-correlation-head`: the system under test.
- `022-adaptive-gate`, `025-a-contrario-gate`: single-fix gates on the heatmap shape, both failed. This
  rule reads the drone frame's content instead of the surface.
- `032-empty-view-likelihood` (planned): the opposite treatment of empty frames, using them as
  negative evidence.
