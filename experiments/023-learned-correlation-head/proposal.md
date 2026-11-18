# 023-learned-correlation-head — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a small network that scores the drone class-probability map against the
satellite class-probability patch, trained on cross-date pairs from 46 villages
neither model has seen, place the drone closer to the truth than `007.1`'s
weighted NCC on the same maps, same frames and same window?

## Hypothesis

Weighted NCC sums six independent per-class correlations with weights
`0/1/1/0.5/0/0.5` that were set by hand in the 002 proposal and never tuned. It
cannot learn that a building edge survives seven years and a woody edge does
not, that field is useless alone but parcel edges are structure, or that a
thin track hallucinated on a dirt patch (`birky-13`, 021) should count less
than a track that lines up with a building. Its error lives on bushcha (7–8 m
vs 3 m on birky): look-alikes along a road and flat woody, where every class
correlates a little everywhere.

Mechanism: a head trained with cross-entropy over the true offset sees
thousands of pairs where the map date differs from the query date. It learns
which class patterns and which co-occurrences stay put across dates, and
scores those. Training pairs are free: `ua-village-multidate` dates are
co-registered to 1.5 cm median, so a crop of `current` at a known pose and a
patch of an older date around a shifted prior carry an exact offset. No labels
are needed.

Two things the gain could really be, and how they are kept apart:

- **Better weights, not better matching.** Arm `W` learns only the six class
  weights (and a temperature) inside the same NCC. `H − W` is the value of
  learned matching; `W − 023.1` is the value of tuned weights (roadmap idea 8
  on this bench).
- **Attempting the blind frames.** `007.1` keeps the prior on the 25% of draws
  whose query holds no weighted class. A learned arm also scores field and
  background and so moves on those. Every metric is reported on the `covered`
  and `blind` subsets (blind = no weighted class under the hand weights).

What this does **not** fix: window reach. At 10 m only 55% of draws have the
truth inside the rect ×1.7 window (`007.1` `reachable_share`). No matcher can
improve the rest, so −0.5 m on all draws is about −1 m on reachable ones.
`mean_error_reachable_m` is reported alongside.

## Method

Everything on the `007.1` setup unless stated: MobileNetV3 LR-ASPP satellite BN
on both sides (frozen), map date newest, rect window ×1.7, flat prior, no gate,
`sat_feature_crop_pct` 0.03, 0.385 m/px, 40 m altitude footprint 46 × 26 m.

**Villages.** The 56 `ua-village-multidate` locations other than birky, bushcha
and dubrova (the segmenter's ground). Split by location, seed 42: **46 train /
10 val**. The split is written to `artifacts/split.json`. No test-bench place
is in either.

**Training pairs.** Village uniform over train villages. Query = the `current`
RGB crop at a uniform pose (80 m margin, uniform heading), segmented exactly as
the drone path does (`SegmentationFeatures.drone`). Map = an Esri Wayback date
of that village other than `current`, uniform (69 dates over 56 villages,
median 1 per village). The prior is shifted from the truth by an offset uniform
over the surface cells. Patch = the map-date probability raster cropped at the
prior, as `SegmentationFeatures.satellite_patch` does. Target = the true cell.
A fixed bank of `N_TRAIN_PAIRS` = 20,000 train pairs and `N_VAL_PAIRS` = 2,000
val pairs, seeded.

**Arms.**

| Arm | Surface | Params |
|---|---|---|
| `ncc` | `007.1`'s weighted NCC, hand weights | 0 |
| `W` (`ClassWeightHead`) | `T · Σ w_c · NCC_c / Σ |w_c|` over the six per-class NCC maps of `class_ncc`; `w` initialised at the hand weights, unconstrained | 7 |
| `H` (`CorrelationHead`) | shared encoder 6→32→32→16 (3×3 conv, BN, ReLU) on query and patch; channel-mean NCC of the encoded maps, concatenated with the six raw per-class NCC maps → 3×3 conv 7→16→1 | ~15k |

`H` contains `W`'s inputs, so `ncc ⊂ W ⊂ H`. Every surface lives on the same
cell grid as `weighted_ncc_heatmap`, and `decide` is unchanged: argmax, flat
surface keeps the prior.

**Loss.** Cross-entropy over the surface cells against a Gaussian soft target
at the true offset, σ = 1 cell (0.385 m).

**Training.** AdamW, lr 1e-3, weight decay 1e-4, batch 32, `STEPS` = 4,000,
OneCycle. **The final step is the model.** No early stopping and no checkpoint
selection on val, so the val villages stay a fair second test. Log every 100
steps (`LOG_EVERY_STEPS`), step 0 before the first update. Train and val mirror
each other on fixed 512-pair subsets in eval mode: `loss`, `error_m` (argmax
to truth), `within_1m_share`. Plus `train/running_loss`, `train/lr`, and
heatmap grids of 8 fixed train and 8 fixed val pairs. `W` uses the same recipe.

**Benches.**

- **Primary:** `007.1`'s bench. `village-crossdate-frames`, 40 frames × 8
  bearings × 5 / 10 / 15 m, seed 42.
- **Holdout:** `village-crossdate-holdout`, the same generator
  (`village-crossdate-frames/build.ipynb` through papermill) on the 10 val
  villages, 20 frames each, map date newest. Both matchers are run on it.
  Metrics carry the `holdout_` prefix.

### Runs

| Run | Arm | Seed | Asks |
|---|---|---|---|
| `023.1` | `ncc` | 42 | must reproduce `007.1` on this machine (M1) |
| `023.2` | `W` | 42 | do tuned weights alone help? |
| `023.3` | `H` | 42 | **the idea** |
| `023.4` | `W` | 7 | seed check |
| `023.5` | `H` | 7 | seed check |

Run `023.1` first. Stop if it does not reproduce `007.1` (see Requires).

## Success criterion

Primary level 10 m injected, as in 007. All levels are reported. CIs are 95%
paired frame-bootstrap over the 40 frames, 10,000 resamples, seed 42
(`paired_frame_bootstrap`), against `023.1`'s per-draw records.

**Metric:** `mean_position_error_m` @ 10 m injected, all draws
**Baseline:** `007.1` → **5.861 m** (`runs.jsonl`; harm 0.084, reachable 0.55),
re-run as `023.1`
**Bar (`023.3` and `023.5`, each):**

1. `H − 023.1` @ 10 m ≤ **−0.5 m** and its upper CI < 0.
2. Guards: `H − 023.1` @ 5 m and @ 15 m upper CI < +0.5 m; `harm_share` @ 10 m
   ≤ 0.084.
3. Holdout bench: `H − ncc` @ 10 m point estimate < 0 (it must not contradict).

**Yes** needs both seeds. `W` is reported against the same bar, but it does not
decide the verdict.

Also reported for every arm: the `covered` / `blind` and bushcha / birky
subsets, `p90` and max, `mean_error_reachable_m`, the train–val gap of
`error_m` at the final step, and `ms_per_correction` on M1 CPU. Latency is not
in the bar. If `H` costs more than 2× `023.1`, the Pi cost goes in Next.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.matching.learned`: `ClassWeightHead`, `CorrelationHead`, `heatmap_loss`, `train_head(..., writer=)` (logs per the training-logging rule), `LearnedScorer` | to build | (a) `ClassWeightHead` at the hand weights gives the same argmax as `weighted_ncc_heatmap` on every covered `007.1` draw; (b) both heads overfit 64 pairs to mean `error_m` < 0.5 m; (c) surface shape equals the NCC surface shape for rect ×1.7 |
| `PositionCorrelator` with `matching="learned"` and a scorer | to build | `023.1` reproduces `007.1`: mean @ 10 m within ±0.05 m of 5.861 and ≥ 95% of draws within 0.01 m (M1 MPS vs the Radeon MPS that ran 007) |
| `uavloc.data.match_pairs`: `village_split`, pair bank, map-probability cache per (village, date) under `data/derived/` (gitignored) | to build | (a) same-date RGB on RGB: NCC argmax at the true cell ±1 cell on ≥ 99% of textured pairs (geometry); (b) offsets uniform over the cells (KS p > 0.01); (c) train and val villages disjoint, none of birky / bushcha / dubrova; (d) cached map probabilities equal `SegmentationFeatures`' on the two test places to 1e-4 |
| `datasets/village-crossdate-holdout` | to build (papermill of `village-crossdate-frames/build.ipynb`, the 10 val villages) | 200 frames, manifest, preview |
| Heatmap grid for TensorBoard in `uavloc.plotting.correlation` | to build | — |
| `frame_place` → `uavloc.data.drone_frames`; `median_ms_per_correction` → `uavloc.matching.benchmark` | to move (second use: 007, 018, 022) | 007's notebook gives the same output after the move |
| `village-crossdate-frames`, LR-ASPP checkpoint, `benchmark_draws` / `score_draws` / `summarise` / `paired_frame_bootstrap` | exist | — |

**What only you can provide.** Nothing. Imagery for all 59 villages is on disk.

> **Note, 2026-10-19 (build, before any run).** Four details fixed at build; the
> success criterion is unchanged.
> 1. **No separate map-probability cache.** Each pair is made by the bench's own
>    code (`SegmentationFeatures` on the map date, `PositionCorrelator.prepare`),
>    and the bank stores the resulting query and patch. Check (d) holds by
>    construction instead of by comparison.
> 2. **The overfit check is for `H` only.** `W` has 7 parameters and cannot
>    memorise 64 pairs; its check is that the loss falls on them.
> 3. **Correlation runs through the FFT on CPU.** torch 2.2 has no MPS FFT, and
>    a direct batched conv with a 67 × 120 kernel takes ~3 s per batch on MPS.
>    The FFT size is padded to a 5-smooth size (109 → 120), which leaves the valid
>    positions exact (2.5e-12 against `conv2d` in float64). Batched NCC matches
>    `class_ncc` to 8e-7. The encoder and refinement convs stay on MPS.
> 4. **OneCycle warm-up 10 %** of the steps (`WARMUP_FRAC`), the project's usual
>    value; the proposal named OneCycle without it.
>
> **Note, 2026-10-20 (acceptance checks, after `023.1`, before any trained run).**
> Two defects in the heads, found by the overfit check; the criterion is unchanged.
> 5. **`W`'s surface is `T · Σ w_c · NCC_c` over present classes, without
>    the division by `Σ |w_c|` written in Method.** With the division (and 007's
>    z-score) the loss jumps where a weight crosses 0: a pair whose only present
>    class has weight 0 goes from flat to full-strength NCC. The gradient at 0 was
>    −3e10, and Adam stepped onto a worse plateau (loss 7.42 → 7.59 on 64 pairs).
>    Without it the loss falls monotonically. At the hand weights the argmax is
>    still 007's, so `ncc ⊂ W` holds.
> 6. **`H`'s encoder uses GroupNorm (4 groups), not BatchNorm.** One BatchNorm
>    shared by query and patch normalised each with its own batch statistics in
>    training but with one mixed running average in eval, so eval error drifted
>    away from training. Learned-feature NCC also gets a small variance floor on
>    the kernel and window (`LEARNED_VAR_EPS`).
> 7. **The batched NCC differs from `class_ncc` by up to 0.16 at a few cells**
>    (1.25 % of pair-channels): windows whose satellite std is within 0.0499–0.054
>    of the 0.05 cut-off, where float rounding flips the threshold. Argmax parity
>    with 007 holds on all 720 covered draws.

> **Note, 2026-10-30 (acceptance checks, after the 5–6 fixes, before any trained
> run).** Check (b) now overfits on the first 64 **covered** val pairs (a class
> with nonzero hand weight present in the query), not the first 64 pairs. The bar
> is unchanged (`H` < 0.5 m, `W` loss falls). Of the first 64 pairs, 34 had no
> weighted class and under 1% non-background pixels: a blank class map carries no
> position, so no model can memorise it and the bar was unreachable for any head
> (re-run after the fixes: `H` 12.8 → 4.15 m, `W` 7.79 → 7.22). The overfit
> curves now go to TensorBoard under `023-learned-correlation-head/overfit-<arm>`.
> The success criterion is unchanged.
>
> **Note, 2026-10-30 (check (b) result, accepted by the author).** On the 64
> covered pairs `H` went 12.47 → 0.536 m in 400 steps (0.81 m at step 300, still
> falling); `W` loss 7.57 → 6.64. `H` misses the 0.5 m bar by 0.04 m, and only for
> want of steps. The author accepted this as a pass, so `checks.json` keeps
> `"pass": false` for `H`. Trained runs `023.2`–`023.5` start from here.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

## Setup

| | |
|---|---|
| Dataset | train/val: `ua-village-multidate` (46 / 10 villages); test: `village-crossdate-frames`, `village-crossdate-holdout` |
| Split | by location, seed 42; test places excluded from training |
| Seed | 42 and 7 for trained arms; bench seed 42 |
| Model | frozen MobileNetV3 LR-ASPP satellite BN (`pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt`) + head |
| Loss | CE over surface cells, Gaussian target σ = 1 cell |
| Hardware | M1 Pro MPS for training and surfaces, CPU for timing |

## Known confounds, accepted up front

- **Same generator for train and test.** Training pairs and both benches are
  satellite crops cut the same way (40 m, nadir, exact heading). The head may
  learn cues of the generator. Real flights differ in heading and scale (004).
- **Not drone imagery.** Query and map are both satellite, as in 005–022.
- **Primary bench is 40 frames on 2 places**, 20 of them overlapping on one
  strip. The holdout bench widens it to 10 more places but only as a
  no-contradiction check.
- **Training targets carry co-registration error.** A few old dates were
  misregistered by up to 4.16 m before correction. The residual is unknown;
  the σ = 1 cell target does not absorb it.
- **One segmenter.** The 021 U-Nets are not used. Swapping the input model is
  a follow-up.
- **Machine changed.** 007 ran on the Intel/Radeon Mac. `023.1` is the paired
  reference for that reason.

## What would falsify this

- **Both `H` seeds miss bar 1:** a learned scorer over the probability maps
  does not beat NCC with 46 villages of pairs. The matching function is not
  the lever at this scale. Go to multi-hypothesis tracking (roadmap 4).
- **`W` is within CI of `H` on mean error @ 10 m:** the gain is the weights.
  Keep learned weights (7 numbers, free on the Pi), drop the head.
- **Bar met on the primary bench but holdout Δ ≥ 0:** the win is a 2-place
  artefact.
- **The whole gain sits in the `blind` subset:** it is coverage, not matching
  quality. Report it as such; the covered draws decide whether matching
  improved.
- **Final val `error_m` > 2× train `error_m`:** the head memorises 46 villages.
  More villages before revisiting, not a bigger head.

## Related

- `007-search-prior-and-gate`: `007.1`, the baseline and its bench.
- `006-segmentation-ceiling`, `018-segfix-correction`, `021-building-loss-from-scratch`:
  perfect or sharper masks buy ≤ 1 m; the mask is not the bottleneck.
- `020-building-centroid-matching`: the other non-NCC matcher, never run.
- `022-adaptive-gate`: per-frame learned gating failed with 2 places; this
  trains on 46.
- `builds/corrector-identity-bench`: 0.17 m self-match floor, 20% blind draws.
- `src/docs/roadmap-2026-10-04.md`, idea 13; idea 8 (class-weight sweep) is
  arm `W`; idea 14 (backprop through NCC into the segmenter) comes after this.
