# 010-dense-crf-refinement — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a fully-connected CRF (Krähenbühl & Koltun 2011) over the LR-ASPP
full-res probabilities, with an RGB bilateral kernel and a Gaussian smoothness
kernel, raise the mean Boundary IoU (d = 4 px) of the satellite masks over the
three val sets, without losing mIoU, at an onboard-plausible CPU cost?

## Hypothesis

The baseline's boundaries are round blobs and fragmented tracks
(`builds/seg-boundary-eval`, `figures/zoom-baseline.png`): the stride-8 head
knows *roughly* where a building or track is, but not where its edge lies.
The imagery does: roofs, tracks and tree crowns are colour-homogeneous and
contrast with their surroundings. The bilateral term pulls a pixel's label
towards pixels that are both nearby and similar in colour, so class edges snap
to colour edges; the Gaussian term removes isolated pixels. Mean-field
inference over the fully-connected graph is tractable with the permutohedral
lattice. We expect the largest gain on `building` (sharp, high-contrast roof
edges) and some on `track`; `woody` edges are texture, not colour steps, and
may get worse.

The risk is the known dense-CRF failure: thin structures (tracks are 8–13 px
wide) whose probability is not dominant get absorbed by the surrounding class,
which would cost track IoU.

## Method

`uavloc/segmentation/refine/dense_crf.py`: `DenseCrfConfig` (iterations,
bilateral `sxy_px`, `srgb`, `compat`, Gaussian `sxy_px`, `compat`),
`crf_marginals(probs, rgb, config) -> (C, H, W)` and
`refine(output, rgb, config) -> (H, W) uint8`. Unary = `-log(max(p, 1e-5))` of
`ModelOutput.probs`. Potts compatibility. Library: `pydensecrf2` 1.1 (the
maintained fork of `pydensecrf`), built from source for this Python 3.11 /
osx-64 env. The CRF runs on the whole frame (experiment 011 will restrict it to
an uncertain band, and reuses `crf_marginals`).

### Registered grid (fixed now, 24 configs)

| Parameter | Values |
|---|---|
| iterations | 5, 10 |
| bilateral `sxy_px` | 20, 40, 80 |
| bilateral `srgb` | 5, 13 |
| bilateral `compat` | 3, 10 |
| Gaussian `sxy_px`, `compat` | 3, 3 (fixed; the paper's value) |

**Tuning on `birky-strip` only.** Selection rule: highest `birky-strip`
Boundary IoU among configs whose `birky-strip` mIoU is within 0.01 of the
baseline's `birky-strip` mIoU. Ties (±0.001) → the cheaper config (fewer
iterations, then smaller `sxy`). The selected config is then scored once on all
three sets. `birky-strip-google` is the same ground on another date and
`bushcha` is an unseen village, so bushcha is the only fully held-out number;
it is reported separately.

## Success criterion

**Metric:** `boundary_iou_mean` (Boundary IoU, d = 4 px, unweighted mean over
birky-strip, birky-strip-google, bushcha), for the config picked on birky-strip.
**Baseline:** harness `builds/seg-boundary-eval` → 0.123 (re-run in this
notebook as run `010.1`; `010.1` is the number used).
**Bar:** ≥ baseline + 0.010 (≈ 0.133), **and** bushcha Boundary IoU above the
baseline's bushcha Boundary IoU (the held-out set must move in the same
direction).
**Guard:** `miou_mean` must not drop by more than 0.01 vs `010.1`.
**Latency budget:** CRF alone ≤ 230 ms median per 512 × 512 tile on this CPU
(i7-9750H, 4-thread cap, `refiner_latency`) — i.e. refinement at most doubles
the ~114 ms model time. Reported, and a pass on accuracy with a miss on latency
is reported as "works, too slow at full frame" (the case 011 addresses).

Also reported: BF1@1, BF1@2, per-class Boundary IoU and IoU per set, and the
full 24-config grid on birky-strip.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.eval`, `.latency`) | exists | — |
| `pydensecrf2` in the `uavloc` env | to install (pin in `environment.yml`) | 5-iteration CRF on a real tile returns normalised marginals of the right shape |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` val sets (harness `load_val_sets()`) |
| Split | tune: birky-strip; report: all three; held out: bushcha |
| Seed | 42 (CRF inference is deterministic) |
| Model | LR-ASPP, satellite BN (`pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt`), frozen |
| Loss | none (no training) |
| Hardware | CPU (i7-9750H); latency at 4 threads |

## What would falsify this

- No grid config raises birky-strip Boundary IoU by ≥ 0.005 while keeping its
  mIoU within 0.01 → a whole-frame dense CRF on these probabilities does not
  sharpen these masks; the residual edge error is not colour-explainable at
  this resolution/label quality. Stop tuning CRF weights.
- The birky-strip gain appears but bushcha does not move (or falls) → the gain
  is tuning noise / site-specific colour statistics; do not carry it forward.
- Gain on building but a larger loss on track → CRF is the wrong tool for thin
  structures here; 011 (band restriction) would have to protect tracks
  explicitly.

## Related

- `builds/seg-boundary-eval` — harness, baseline, metric.
- 011 (planned) — the same CRF restricted to an uncertain band.
