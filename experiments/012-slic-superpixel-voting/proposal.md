# 012-slic-superpixel-voting — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does replacing each pixel's LR-ASPP class probabilities with their mean over an
RGB superpixel (SLIC, or a faster over-segmentation) raise the mean Boundary IoU
of the satellite masks over the three val sets, without costing mIoU?

## Hypothesis

The model predicts at stride 8 (3.1 m cells) and its edges are bilinear blur, so
a building edge in the mask can sit a few pixels off the real roof edge. An RGB
superpixel of ~6–16 px usually stays on one side of a strong colour edge. If
every pixel takes the class of its superpixel's mean probability, the mask edge
moves to the superpixel edge, which is the RGB edge. This should help large,
high-contrast objects (buildings, woody, water) most.

Known risk: tracks are 8–13 px wide and low-contrast against fields. A
superpixel of comparable size will straddle a track and vote it away. Hence an
**area-only** variant that votes only building / woody / water / background
(and field) and leaves the model's own track pixels exactly as they were.

The harness says most of the boundary loss is misplaced or missing edges, not
upsample blur (upsample blur alone is ~0.008 BIoU). Superpixel voting can only
move an edge that is already near a colour edge, and cannot create a missed
object, so the expected gain is small: we expect +0.005 to +0.015 BIoU.

## Success criterion

**Metric:** `boundary_iou_mean` — Boundary IoU (d = 4 px, fixed by the harness),
unweighted mean over birky-strip, birky-strip-google, bushcha.
**Baseline:** `012.1`, LR-ASPP bilinear argmax re-run through this notebook
(harness value 0.123).
**Bar:** `boundary_iou_mean` ≥ baseline + **0.010**.
**Guard:** `miou_mean` must not drop by more than **0.01** vs `012.1`.
**Latency budget:** refiner (superpixels + voting) ≤ **114 ms** median per
512 × 512 tile on CPU, 4 threads (= the model's own time; it may at most double
the pipeline). i7-9750H is a proxy for the Pi-class target. The machine is shared
with other jobs during this series, so the baseline argmax is timed in the same
session as a reference.

The config is chosen on **birky-strip only**; the bar is checked on the
3-set mean of that one chosen config. birky-strip-google and bushcha are
never used for selection.

## Pre-registered runs and grid

Selection rule (applied inside the notebook): among grid configs whose
birky-strip mIoU ≥ baseline birky-strip mIoU − 0.01, take the one with the
highest birky-strip Boundary IoU.

Grid (superpixel method × voting):

| Superpixel method | Settings |
|---|---|
| skimage SLIC (10 iterations, Lab) | cell ≈ 6, 10, 16 px (`n_segments = H·W / cell²`) × compactness 5, 10, 20 |
| skimage Felzenszwalb (speed alternative) | scale 25, 100; sigma 0.5; min size = 6² px |
| OpenCV watershed from a regular seed grid (speed alternative) | cell 6, 10, 16 px |

× vote `all` (every class) | `area` (track untouched) × alpha 1.0 (hard
superpixel mean) | 0.5 (`0.5·mean + 0.5·pixel probs`).

`cv2.ximgproc` SLIC/LSC is **not** in the grid: the env has `opencv-python`
without contrib, and swapping it for `opencv-contrib-python` replaces cv2 in an
environment other jobs are using. Felzenszwalb and grid watershed stand in as the
speed alternatives.

| Run | Grid | Asks |
|---|---|---|
| `012.1` | none (bilinear argmax) | baseline through this notebook |
| `012.2` | full grid | **primary**: the selected config vs the bar |
| `012.3` | vote = area only | secondary: best thin-track-safe variant |
| `012.4` | watershed only | secondary: best fast variant |

Only `012.2` decides the verdict. `012.3` and `012.4` are reported as
diagnostics. Every run reports the three sets, their mean, per-class IoU and
Boundary IoU (track and building especially), and CPU ms per 512 tile.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.segmentation.eval` / `latency` (`builds/seg-boundary-eval`) | exists | — |
| `uavloc.segmentation.refine.superpixels` | to build | segment mean of a constant map is that constant; `alpha = 0` reproduces baseline exactly; area vote leaves track pixels identical to baseline |
| `scikit-image` 0.24.0 | installed into `uavloc` env for this experiment | imports; numpy stays 1.26.4 |

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` via `load_val_sets()` |
| Split | tune: birky-strip; report: birky-strip, birky-strip-google, bushcha |
| Seed | 42 (SLIC and watershed are deterministic) |
| Model | `model-lraspp-bn-satellite.pt` (LR-ASPP, satellite BN), frozen |
| Loss | none (no training) |
| Hardware | CPU, i7-9750H; latency at 4 threads |

## What would falsify this

If the best birky-strip config, applied to all three sets, gains less than
0.010 mean Boundary IoU, superpixel voting is not worth pursuing on this model:
the edges it can move are not the ones that are wrong. If the only configs that
gain are ones whose track IoU collapses, that is a failure too, regardless of the
mean. If every config that clears the bar is over the latency budget, the idea is
not deployable as is.

## Related

- `builds/seg-boundary-eval` — harness, baseline and metric.
- Sibling mask-sharpening experiments 009–016 (run in parallel).
