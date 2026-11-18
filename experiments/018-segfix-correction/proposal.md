# 018-segfix-correction — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does applying `014`'s SegFix boundary displacement to the LR-ASPP class maps,
on both the satellite map and the drone frame, lower the mean position error of
the `007.1` correction test at 10 m injected error?

## Hypothesis

Weighted NCC scores how well the class edges of the drone map line up with the
satellite map. LR-ASPP decides classes at stride 8 and smears its edges by
several pixels, and the smear differs between the two dates because the
appearance differs. SegFix moves those edges back towards the image edges
(`014`: Boundary IoU 0.123 → 0.140 at k8 s4, mIoU +0.006, best on unseen
bushcha). Sharper, better-placed edges on both sides should give a sharper and
more correct correlation peak.

The refiner moves whole probability vectors, not labels: each boundary pixel
takes the class probabilities found `offset_px` along its predicted direction.
Its argmax equals `014`'s refined label exactly (checked on a bushcha tile, 2.3%
of pixels change). The soft map is kept because the matcher uses probabilities.

Stated limits: a +0.017 BIoU gain is small next to the gap between the two
dates' appearance, so it may not show through the correlation. SegFix's
direction accuracy is only 0.36 (`014`).

## Success criterion

**Metric:** `mean_position_error_m` @ 10 m injected, all 40 frames.
**Baseline:** `018.1`, LR-ASPP without refiner, through this notebook. It must
reproduce `007.1` (5.861 m) to within 0.05 m. Otherwise the harness moved, and
that is reported first.
**Bar:** primary run `018.2` (SegFix k8, s = 4 px, seed-42 checkpoint), paired
with `018.1` over frames: mean difference ≤ −0.5 m **and** the 95 %
frame-bootstrap CI upper bound < 0.

**Guard:** at 5 m and 15 m injected, the paired difference's CI lower bound
must not be above +0.5 m (SegFix must not clearly hurt at either level).

**Also reported, not part of the bar:** harm share @ 10 m, bushcha vs birky
split, `ms_per_correction` on CPU with the refiner included.

## Registered runs

| Run | What | Role |
|---|---|---|
| `018.1` | LR-ASPP, no refiner (= `007.1`) | baseline |
| `018.2` | SegFix k8 seed 42, s = 4 | **primary** (`014`'s recommended config) |
| `018.3` | SegFix k8 seed 42, s = 8 | `014`'s best-BIoU config |
| `018.4` | SegFix k8 seed 43, s = 4 | checkpoint swing |

If `018.2` and `018.4` disagree on pass/fail, the verdict is *not resolved*.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `experiments/014-boundary-displacement/artifacts/segfix-k8-seed{42,43}.pt` | exists | loads into `SegFixNet(SegFixConfig())` |
| `uavloc.segmentation.refine.segfix.SegFixProbabilities`, `load_segfix` | added (this experiment) | argmax == `SegFixRefiner` label on a 512 tile (checked 2026-09-29) |
| `SegmentationFeatures(refine=...)` hook | added (this experiment) | `refine=None` reproduces `007.1` (run `018.1`) |
| `village-crossdate-frames`, `007`'s correlator setup | exists | — |

## Setup

| | |
|---|---|
| Dataset | `village-crossdate-frames`: 40 frames on bushcha and the birky val strip, map = newest date |
| Correlator | `007.1`: rect window ×1.7, flat prior, no gate, weighted NCC, class weights (0, 1, 1, 0.5, 0, 0.5) |
| Segmenter | LR-ASPP satellite BN (`pocs/mobilenet-lraspp-speed`), both sides, 0.385 m/px |
| Refiner | SegFix at 0.385 m/px (its training scale: z18 at ~50° N), threshold 0.5 |
| Draws | noise 5 / 10 / 15 m × 8 directions, seed 42 |
| Hardware | MPS for the run, CPU for timing |

SegFix never trained on the frame ground: bushcha is held out, and the birky
strip rows are banned from its training crops (as for LR-ASPP).

## What would falsify this

If `018.2` is within ±0.3 m of the baseline at 10 m, boundary sharpening at
this scale does not reach localisation, and further SegFix work (speed,
direction) is justified only by mask quality, not by correction error.

## Related

- `014-boundary-displacement`: the SegFix model and its mask gains.
- `007-search-prior-and-gate`: the baseline correlator setup (`007.1`).
- `006-segmentation-ceiling`: how much a perfect satellite map buys.
