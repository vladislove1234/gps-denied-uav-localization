# 011-uncertain-band-refinement — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

If the expensive refiners of 009, 010 and 012 run only in an uncertain band of
the LR-ASPP mask (pixels near a predicted class boundary, or with a small
top-1 minus top-2 probability margin), and every other pixel keeps the plain
argmax, does the best of them raise mean Boundary IoU by ≥ 0.010 over the
harness baseline without costing more than 0.01 mIoU?

## Hypothesis

The three siblings found small boundary gains that came with interior damage:

- 009 joint guided upsampling from stride 8 (fast, RGB guide, r = 1 cell,
  eps = 1e-3): +0.006 mean / +0.011 bushcha BIoU, but −0.0096 mIoU (guard failed
  on birky-strip).
- 010 dense CRF: every config below baseline BIoU; the compat-10 config raised
  BF1@1 but ate thin tracks and cost 0.03 mIoU on birky-strip. 1.25 s per tile.
- 012 grid watershed (6 px, vote all, α = 1): +0.0056 BIoU at −0.0038 mIoU.

Mechanism: Boundary IoU only looks at pixels within 4 px of a label or
predicted contour, while mIoU is dominated by interiors. A refiner's useful
work (moving an edge onto an image edge) happens within a few pixels of the
model's own contour; its harmful work (texture overwriting a class inside a
field, the CRF's Potts term deleting a whole track, a watershed cell swallowing
a thin structure) happens anywhere. Confining the refiner to a band of d px
around the predicted boundary should keep most of the gain and remove most of
the damage. Cost should fall roughly with the band's area, if the refiner can
run on the band's pixels rather than on the whole frame.

What it cannot do: the band is built from the model's own boundary, so an
object the model missed entirely has no band and cannot be recovered, and a
boundary misplaced by more than d px cannot move to the right place. 010 and
the harness say most of the remaining boundary error is of that kind, so the
realistic expectation is "gain ≈ the full-frame gain, mIoU loss ≈ 0",
i.e. +0.004 to +0.008 BIoU, below the bar. The run is worth it for the
latency and damage-control answers even if the bar is missed.

## Implementation

`uavloc.segmentation.refine.band.band_refiner(base_refine_fn, BandConfig)`:

- band = pixels within Euclidean `distance_px` of a predicted class boundary
  (both sides, 4-neighbour boundary of the bilinear argmax) and/or pixels with
  top-1 − top-2 probability < `margin`;
- band pixels take the base refiner's class, all others keep the argmax;
- three crop modes: `frame` (base refiner on the full frame, output masked:
  quality reference, no saving), `tiles` (base refiner on each 96-px tile that
  touches the band, with 24 px of context), `pixels` (the refiner works on the
  band pixels only).

A probe on real tiles before this proposal showed the band is spread over the
whole village: at d = 6 px it holds 27–30 % of pixels, but tiles that touch it
cover 78–93 % of the frame at any tile size from 48 to 192 px. Tile cropping
therefore cannot cut latency much, so the **deployable mode is `pixels`**, and
each refiner got a per-pixel variant:

- CRF: `dense_crf.crf_marginals_at` builds the dense CRF (same unary, same
  Gaussian and bilateral Potts kernels) over the band pixels only, via
  `pydensecrf.DenseCRF` with explicit features. Its cost scales with the node
  count. On an all-true mask it reproduces `DenseCRF2D` exactly.
- Guided upsampling: `guided_filter.refine_at` fits the stride-8 coefficients
  as before and evaluates `a · I + b` only at band pixels, with the model's
  bilinear x8 weights. Identical to full-frame on the band by construction.
- Watershed: `superpixels.masked_superpixel_refiner` pre-labels everything
  outside the band as one frozen region, so the flood runs inside the band only
  and superpixels stop at the band edge. Not identical to masking the full-frame
  watershed (0.05–2 % of pixels differ on the probe, most at d = 3).

## Success criterion

**Metric:** `boundary_iou_mean`, Boundary IoU (d = 4 px, harness), unweighted
mean over birky-strip, birky-strip-google, bushcha.
**Baseline:** `011.1`, bilinear argmax through this notebook (harness 0.1232).
**Bar:** `boundary_iou_mean` ≥ **0.1332** (baseline + 0.010).
**Guard:** `miou_mean` must not drop by more than **0.01** vs `011.1`.

Selection, fixed now: each refiner run tunes the band on **birky-strip only**
(among band configs whose birky-strip mIoU ≥ baseline − 0.01, the highest
birky-strip BIoU). The experiment's verdict uses the one refiner run whose
selected config has the highest birky-strip BIoU; its 3-set mean is checked
against the bar and the guard. birky-strip-google and bushcha are never used to
select.

**Also reported, per refiner** (not part of the bar): band vs full-frame deltas
in mean BIoU, mean mIoU and refine ms (each ms also as a ratio to argmax timed
in the same session, because the machine is shared); the band's pixel-area
fraction (mean over val images); per-class IoU and BIoU, track especially;
BF1@1/2; latency of every band config in `pixels` mode and in `tiles` mode.

**Latency sub-hypothesis:** "cost falls in proportion to band area" holds for a
refiner if its selected band's `pixels`-mode time, divided by the full-frame
time in the same session, is ≤ the band's area fraction on the latency tile +
0.15. Reported, not part of the bar. For Pi relevance the model itself is
113.8 ms per 512 tile in the harness (27× the harness argmax).

## Pre-registered runs and grid

Band grid (registered, `crop = pixels`, tile 96 / context 24 for the tiles
latency diagnostic):

| band | distance_px | margin |
|---|---|---|
| d3 | 3 | — |
| d6 | 6 | — |
| d12 | 12 | — |
| m0.2 | — | 0.2 |
| m0.5 | — | 0.5 |

| Run | Base refiner | Asks |
|---|---|---|
| `011.1` | none (bilinear argmax) | baseline through this notebook |
| `011.2` | 009 joint guided upsampling: fast, RGB, r = 1 cell, eps = 1e-3 | band grid |
| `011.3` | 010 best dense CRF: 5 it, bilateral sxy 20 / srgb 5 / compat 3, Gaussian 3 / 3 | band grid |
| `011.4` | 010 compat-10 dense CRF: 5 it, sxy 20 / srgb 5 / compat 10, Gaussian 3 / 3 | band grid |
| `011.5` | 012 grid watershed 6 px, vote all, α = 1 | band grid |

Each run also scores the base refiner on the full frame on all three sets (the
full-frame reference for the deltas) and the `frame`-masked band (same band,
full-frame refiner output) as a diagnostic of what per-pixel cropping changes.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.segmentation.eval` / `latency` (`builds/seg-boundary-eval`) | exists | — |
| `refine/guided_filter.py`, `refine/dense_crf.py`, `refine/superpixels.py` | exist (009, 010, 012) | — |
| `uavloc.segmentation.refine.band` | built for this experiment | empty band reproduces argmax exactly; all-pixel band in `frame` mode reproduces the full refiner exactly; argmax as base refiner in `tiles` mode reproduces argmax exactly (crop alignment) |
| per-pixel variants (`crf_marginals_at`, `guided_filter.refine_at`, `masked_superpixel_refiner`) | built for this experiment | all-pixel mask reproduces the full-frame refiner exactly (checked on a birky-strip image and the bushcha latency tile: 0 pixels differ for all three); guided `pixels` = guided `frame` on a d = 6 band exactly |

All checks above were run on real val images before this proposal was
written (2026-09-28) and passed.

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` via `load_val_sets()` |
| Split | tune: birky-strip; report: birky-strip, birky-strip-google, bushcha |
| Seed | 42 (every refiner is deterministic) |
| Model | `model-lraspp-bn-satellite.pt` (LR-ASPP, satellite BN), frozen |
| Loss | none (no training) |
| Hardware | CPU, i7-9750H, shared with other jobs; latency at 4 threads on the harness tile (bushcha, (1024, 1024)) |

## What would falsify this

- If no refiner's selected band beats its own full-frame BIoU gain while
  cutting its mIoU loss, the band does not isolate the useful part of the
  refinement, and "refine only where unsure" is not worth building on.
- If the best band config gains < +0.005 mean BIoU, band refinement cannot
  reach the bar on this model by tuning d or the margin; the missing boundary
  quality is in the model's probabilities (misplaced by more than the band or
  missing), not at its edges.
- If the per-pixel CRF on the band still deletes track pixels (track IoU
  below baseline by > 0.01 on any set), the band does not protect thin
  structures, because a track 8–13 px wide lies entirely inside a d ≥ 6 band.
  That is a known risk, so d = 3 is in the grid.

## Related

- `009-guided-filter-refinement`, `010-dense-crf-refinement`,
  `012-slic-superpixel-voting`: the refiners and configs reused here.
- `builds/seg-boundary-eval`: harness, baseline and metric.
