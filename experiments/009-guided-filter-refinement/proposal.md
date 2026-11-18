# 009-guided-filter-refinement — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a training-free guided filter (He et al. 2013), using the satellite RGB as
guide, make the LR-ASPP class masks sharper — mean Boundary IoU over the three
val sets up by at least 0.010 — without costing mIoU and within a CPU budget?

## Hypothesis

LR-ASPP predicts at stride 8 and its boundaries are smooth blobs that ignore the
image. The guided filter models the output as a local linear function of the
guide in each window, `q = a·I + b`, so where the probabilities change inside a
window it snaps the transition to the strongest nearby image edge. Building
roofs and tracks have sharp, high-contrast edges against grass, so their class
boundaries should move by a few px toward the true edge. Woody (textured, low
contrast at its edge) should gain little and may get noisier.

Headroom is limited. The harness (`builds/seg-boundary-eval`) shows most of the
boundary loss is misplaced or *missing* edges, not upsample blur (bilinear vs
nearest is only +0.008 BIoU). A guided filter can move an edge within its radius
but cannot create an object the model missed. So we expect a small gain, larger
for building and track than for woody, and larger for the full-res filter
(variant a) than for joint upsampling from stride 8 (variant b), which cannot
move an edge by more than it moves the low-res coefficients.

## Success criterion

**Metric:** `boundary_iou_mean` — Boundary IoU (d = 4 px, fixed by the harness),
unweighted mean over `birky-strip`, `birky-strip-google`, `bushcha`.
**Baseline:** `009.1` (bilinear argmax re-run through this notebook); must
reproduce the harness value 0.123 (±0.001).
**Bar:** best variant `boundary_iou_mean` ≥ **0.133** (baseline + 0.010).
**Guard:** `miou_mean` must not drop by more than 0.01 vs `009.1` (≥ 0.466).
**Latency budget:** refiner median ≤ **30 ms** per 512×512 tile, CPU, 4 threads
(i7-9750H as Pi proxy; ~25% of the 114 ms model). A variant over budget can be
reported but cannot pass.
**Generalisation check:** the pass also requires `bushcha` BIoU > its `009.1`
value. bushcha is an unseen village and is never used for selection.

**Selection is on `birky-strip` only.** Within each run the grid config is
chosen as the max `birky-strip` BIoU among configs whose `birky-strip` mIoU is
within 0.01 of baseline. The best run among `009.2`–`009.5` is also chosen by
`birky-strip` BIoU. All three sets are reported for every grid point.
`birky-strip-google` is the same ground on another date, so it is only a
semi-independent check.

## Pre-registered runs

| Run | Mode | Guide | Grid |
|---|---|---|---|
| `009.1` | baseline: argmax of bilinear probs | — | — |
| `009.2` | (a) `full_res`: guided filter on full-res bilinear probs, per class | RGB | r ∈ {2, 4, 8} px × eps ∈ {1e-3, 1e-2, 1e-1} |
| `009.3` | (a) `full_res` | grey | same |
| `009.4` | (b) `fast`: joint guided upsampling from stride-8 softmax (Fast Guided Filter; coefficients fitted on 8×8 cell means, bilinearly upsampled, applied to the full-res guide) | RGB | r ∈ {1, 2, 4} cells (= 8, 16, 32 px) × eps ∈ {1e-3, 1e-2, 1e-1} |
| `009.5` | (b) `fast` | grey | same |

Guide is scaled to [0, 1]; eps is in intensity² units. Filtered probabilities
are argmaxed (not renormalised; argmax is unaffected). Each run logs the
selected config's metrics; the whole grid (all sets, per class) is saved to
`artifacts/grid-<run>.csv`. Per-class Boundary IoU is reported for the selected
config (water scores 0 on birky in the baseline).

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.{eval,boundary,metrics,latency}`) | exists | `009.1` reproduces 0.476 / 0.123 |
| `uavloc.segmentation.refine.guided_filter` | to build | constant input stays constant; eps → ∞ gives box-of-box mean; p = guide with eps → 0 returns p; cofactor 3×3 inverse matches `np.linalg.inv` |

`cv2.ximgproc` is **not** available in the `uavloc` env (OpenCV 4.11 without
contrib), so the box-filter guided filter is implemented with `cv2.boxFilter`.

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` val sets from the harness |
| Split | birky-strip (4 Esri dates, rows < 640), birky-strip-google, bushcha (both dates) |
| Seed | 42 (nothing stochastic) |
| Model | `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt`, frozen |
| Loss | none (training-free) |
| Hardware | CPU (i7-9750H), latency at 4 threads on bushcha tile (1024, 1024) |

## What would falsify this

If no variant raises mean Boundary IoU by at least **0.005** over `009.1`
(half the bar) at the selected config, or the gain on birky-strip does not
carry to bushcha (bushcha delta ≤ 0), then image-guided smoothing of this
model's probabilities is not a lever for sharper masks: the edges it would snap
to are not where the model's boundaries are wrong. We drop guided filtering
rather than tune it further, and it is not worth stacking under other refiners.

## Related

- `builds/seg-boundary-eval` — harness, baseline and the stride-8 oracle (0.44 / 0.68 BIoU).
- `006-segmentation-ceiling` — better masks did not reduce localisation error; this series targets mask quality itself (user's decision).
- Sibling mask-sharpening experiments run in parallel (CRF, etc.); guided filter is the cheapest candidate.

## Amendment 2026-09-28 (run ids only; criterion unchanged)

The first launch of the RGB full-res run crashed before `log_run` (numpy was
reinstalled in the shared env mid-run: `TypeError: expected np.ndarray (got
numpy.ndarray)`), so the grey run took the next id. Actual mapping:
`009.2` = full_res **grey**, `009.3` = full_res **RGB**, `009.4` = fast RGB,
`009.5` = fast grey. The two fast runs first failed because `-p RADII "[1, 2, 4]"`
passes a string; rerun with `-y`. The failed executions are kept in
`artifacts/failed/`; none reached `log_run`.
