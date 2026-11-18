# 013-refinement-cnn — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Can a tiny CNN, given full-res RGB and the frozen LR-ASPP's upsampled
log-probabilities and trained with plain cross-entropy, raise the mean Boundary
IoU of the satellite masks over the three val sets by at least 0.015, without
losing more than 0.01 mIoU and within 30% of the model's CPU time?

## Hypothesis

LR-ASPP decides classes at stride 8 (3.1 m cells) and its x8 bilinear upsample
never looks at the image again. The `seg-boundary-eval` build showed that most
boundary loss is misplaced or missing edges, not upsample blur. A large part of
that is local: a building roof ends at a sharp colour step a few pixels from
where the blurred probability crosses 0.5, and a track is a bright thin line the
coarse head smears into a blob or drops. A small network that sees both the
full-res pixels and the model's belief can move a class boundary onto the
nearest image edge and re-thicken a faded track. The residual form (output =
input log-probs + correction, correction head initialised to zero) starts from
the baseline exactly, so the network only has to learn *where to disagree*.

The mechanism has a limit that is stated now: the refiner cannot find an object
LR-ASPP has no evidence for more than ~half a receptive field away, and it is
trained on LR-ASPP's outputs on the images LR-ASPP was trained on, where those
outputs are better than on val (probe: sat-train mIoU 0.707 vs 0.45–0.52 on
val). It may learn to correct a milder error than the one it meets at test time.

## Success criterion

**Metric:** `boundary_iou_mean` — Boundary IoU (d = 4 px, fixed by the harness),
unweighted mean over `birky-strip`, `birky-strip-google`, `bushcha`, from
`uavloc.segmentation.eval.evaluate`.
**Baseline:** `013.1`, the harness baseline (LR-ASPP, argmax of the bilinear
upsample) re-run through this notebook. The build recorded 0.123 (mIoU 0.476,
BF1@1 0.137, BF1@2 0.210); `013.1` must reproduce it to within 0.001, or the
harness moved and that is reported first.
**Bar:** primary config (no edge input), seed 42: `boundary_iou_mean` ≥
baseline + 0.015 (≈ 0.138).
**Guard:** `miou_mean` ≥ baseline − 0.01. Reported per set and per class as
well; a gain carried by one set while another loses is flagged in the report.
**Latency budget:** refiner median CPU time on the harness's standard 512 tile
(bushcha 2024-06-06 at (1024, 1024), 4 threads) ≤ 30% of the LR-ASPP model
median **measured in the same session**. The machine is shared with other
jobs (load average 15–40 while this was written, model 172–475 ms against the
build's 114 ms), so the budget is a ratio measured back to back, not an
absolute number. Absolute ms are reported alongside. Pre-measurement on random
weights: the chosen size costs ~0.15–0.3 of the model under that load.

A second seed (43) of the primary config is run if time allows. The verdict is
made on seed 42; if seed 43 disagrees on pass/fail, the verdict is written as
*not resolved* and the swing is reported. With the LR-ASPP frozen, seed swing
here is refiner-only; the build's eval is deterministic.

## Registered runs

| Run | What | Role |
|---|---|---|
| `013.1` | baseline: bilinear argmax, no refiner | baseline |
| `013.2` | refiner, RGB + log-probs, seed 42 | **primary** |
| `013.3` | refiner, RGB + log-probs + Sobel edge magnitude, seed 42 | ablation: does an explicit edge map help |
| `013.4` | primary config, seed 43 | seed swing |
| `013.5` | edge config, seed 43 (if time allows) | seed swing |

The edge ablation is read as a delta against `013.2`; it only counts as an
effect if it exceeds the seed swing seen between `013.2` and `013.4`. No other
configs are run. No hyperparameter is chosen on the val sets.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.{eval,boundary,metrics,latency}`) | exists | `013.1` reproduces its baseline to 0.001 |
| `uavloc/segmentation/refine/refinement_cnn.py` | to build (this experiment) | zero-initialised head gives predictions identical to the baseline; tiled inference identical to single-pass (both checked on random weights before the first run) |
| `pocs/village-seg-baseline/village_seg.py` (`load_satellite`, `ban_holdouts`, `SatelliteSampler`) | exists | same train split as `probe.ipynb` (asserted against `results-lraspp.json`) |
| `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` | exists | — |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` satellite images; training crops from `SatelliteSampler` (512 px, rare-class share 0.3, the probe's photometric/rot/flip augmentation), exactly as LR-ASPP was trained |
| Split | train: everything except bushcha and birky rows `[:640 + 64]` (`ban_holdouts`, as probe/round 3). Val: the harness's three sets, never trained on. A 64-crop hold-out drawn from the *train* images is used only to watch the loss, not to select anything |
| Training data | 512 augmented crops, quantised to uint8, with the frozen LR-ASPP's stride-8 logits (satellite BN, eval mode) precomputed; each step takes random 256 px sub-crops on the stride-8 grid plus a shared rot90/flip |
| Model | `RefinementCNN`: pixel-unshuffle x4 of the image (+ edge channel), 3x3 stem (width 32) + 1x1 on the avg-pooled log-probs, 3 residual depthwise-separable blocks with dilations 1, 2, 4, 1x1 head to 6 x 16 channels, pixel-shuffle back to full res, added to the log-probs. ~21k params (~26k with edges). Receptive radius 36 px |
| Loss | cross-entropy, ignore 255 |
| Optimiser | AdamW lr 3e-3, wd 1e-4, one-cycle, 3000 steps x batch 16 |
| Seed | 42 (43 for the swing runs) |
| Hardware | training on MPS (Radeon Pro 5300M); eval and latency on CPU |

## What would falsify this

If the primary refiner at seed 42 gains less than +0.005 Boundary IoU (the
size of the whole bilinear-vs-nearest upsample effect in the build), a tiny
image-aware refiner trained on LR-ASPP's own training-set outputs does not fix
its boundaries, and this direction is dropped rather than widened. A bigger
refiner is a different experiment with a different latency argument. If the
hold-out CE improves but val Boundary IoU does not, that is the train/val logit
gap named in the hypothesis, and it is reported as the reason.

## Related

- `builds/seg-boundary-eval` — harness, baseline and the stride-8 oracle (0.44 nearest / 0.68 bilinear).
- `006-segmentation-ceiling` — why localisation error is not the metric here.
- Parallel mask-sharpening experiments 009–016 (guided filter, dense CRF, SLIC voting, SegFix-style displacement, boundary head, boundary loss) are scored on the same harness.
