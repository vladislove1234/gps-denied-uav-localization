# 017-gray-logit-refiner — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a small full-resolution residual CNN, which sees the grayscale image and
the frozen LR-ASPP's raw logits and is trained with CE + Dice to predict only a
logit correction (`Z' = Z + ΔZ`), raise the mean mIoU of the satellite masks
over the three val sets by at least 0.01?

## Hypothesis

LR-ASPP decides classes at stride 8 and its bilinear x8 upsample never looks at
the image again. The grayscale texture holds the local evidence (roof edges,
the bright line of a track) that the upsampled logits smear. Dilated depthwise
layers (1, 2, 4) let the refiner see a patch of the shape around an edge,
not just the edge pixel. The residual form keeps the refiner from competing
with the segmenter. A zero-initialised head starts from the baseline exactly,
so it only learns where to disagree. Raw logits keep the model's confidence
scale, which log-probabilities flatten.

Stated limits, from `013-refinement-cnn`, which asked a close question with RGB
and log-probs on a pixel-unshuffle x4 view (receptive radius 36 px). That
refiner gained +0.0025/+0.0046 Boundary IoU (seeds 42/43) and lost mIoU
(−0.0096/−0.0053). It trimmed and squared edges but did not recover missing
objects. This design is the user's, and it differs in four ways: grayscale,
raw logits, CE + Dice, and every layer at full resolution. Its receptive radius
is only 8 px (1 + 1 + 2 + 4). As in 013, the refiner trains on LR-ASPP's
outputs on LR-ASPP's own training images, where those outputs are better than
on val (sat-train mIoU 0.707 vs 0.45–0.52). It may learn to correct a milder
error than the one it meets at test time.

## Success criterion

**Metric:** `miou_mean`: mIoU, the unweighted mean over `birky-strip`,
`birky-strip-google` and `bushcha`, from `uavloc.segmentation.eval.evaluate`.
**Baseline:** `017.1`, LR-ASPP argmax of the bilinear upsample, re-run through
this notebook. It must reproduce `013.1` (`miou_mean` 0.4758,
`boundary_iou_mean` 0.1232) to within 0.001. Otherwise the harness moved, and
that is reported first.
**Bar:** primary config, seed 42: `miou_mean` ≥ baseline + 0.010 (≈ 0.486).

**Also reported, not part of the bar:**
- `mdice_mean`: mean Dice over classes, the second metric asked for. Per class,
  Dice = 2·IoU / (1 + IoU), so it moves with IoU. It is logged, not judged
  separately.
- `boundary_iou_mean`, BF1@1, BF1@2, for comparison with 013–016 on
  sharpness. 013's bar was +0.015 BIoU.
- Refiner CPU time on the harness's standard 512 tile, as a ratio to LR-ASPP
  measured in the same session. This is a diagnostic only: 013 found that
  full-res convolutions cost 60–70 % of the model time on this CPU.

The verdict is made on seed 42. If seed 43 disagrees on pass/fail, the verdict
is *not resolved* and the swing is reported. A gain carried by one val set
while another loses is flagged.

## Registered runs

IDs are assigned in the order the runs are logged. `run_all.sh` logs them in
this order.

| Run | What | Role |
|---|---|---|
| `017.1` | baseline: bilinear argmax, no refiner | baseline |
| `017.2` | refiner, gray + logits, CE + 1.0·Dice, seed 42 | **primary** |
| `017.3` | primary config, seed 43 | seed swing |

Ablations wait until the primary has a verdict: RGB vs gray, log-probs vs
logits, a boundary-loss term. They are new runs of this notebook (`IMAGE`,
`SCORES`, loss params) via `/exp-iterate`. No hyperparameter is chosen on the
val sets.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.{eval,boundary,metrics,latency}`) | exists | `017.1` reproduces `013.1` to 0.001 |
| `uavloc.segmentation.metrics`: Dice per class and `mdice` | added (this experiment) | hand-checked on a 2×2 case; per class, Dice = 2·IoU/(1+IoU) |
| `uavloc.segmentation.refine.residual_refiner.ResidualRefiner` | added (this experiment) | zero-init head gives predictions identical to the baseline; tiled inference identical to a single pass (both checked, 2026-09-29) |
| `uavloc.segmentation.refine.refinement_cnn` crop bank, loop and tiling with `as_logits`, `loss_fn` | extended (this experiment) | the 013 path with zero init still gives the baseline exactly (checked) |
| `uavloc.segmentation.losses.ce_dice_loss(dice_weight=...)` | extended | default 1.0 = the round-3 loss |
| `pocs/village-seg-baseline/village_seg.py`, `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` | exist | same train split as `probe.ipynb` (asserted) |

**What only you can provide.** You run the notebook yourself
(`run_all.sh`, or `train.ipynb` interactively, baseline first).

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` satellite images. The training crops are exactly 013's: `SatelliteSampler`, 512 px, rare-class share 0.3, the probe's augmentation |
| Split | the same as 013 and the probe. Train: everything except bushcha and birky rows `[:640 + 64]`. Val: the harness's three sets, never trained on. A 64-crop hold-out from the train images, used only to watch the loss |
| Training data | 512 crops with the frozen LR-ASPP's stride-8 logits (satellite BN, eval mode). Each step takes random 256 px sub-crops on the stride-8 grid plus a shared rot90/flip. The refiner sees the bilinear-upsampled logits |
| Model | `ResidualRefiner`: [Gray, Z₁..Z₆] → Conv 3×3 → 32, ReLU → DWConv 3×3 d = 1, ReLU → d = 2, ReLU → d = 4, ReLU → Conv 1×1 → 6 (zero-init) = ΔZ. Output Z + ΔZ. 3,206 params. Receptive radius 8 px. LR-ASPP is frozen |
| Loss | CE (ignore 255) + λ·soft Dice, λ = 1.0 (`ce_dice_loss`) |
| Optimiser | as 013: AdamW lr 3e-3, wd 1e-4, one-cycle (5 % warm-up), 3000 steps × batch 16 |
| Monitoring | every `LOG_EVERY` = 250 steps: train loss/CE of that window; hold-out CE (train images); **val** CE and CE + Dice over all val images cut into 512 px crops (drops all-ignore crops, pooled over the three sets), refiner vs LR-ASPP alone. Val is only watched for overfitting: the final step is always the evaluated model, with no early stopping and no step selection |
| Seed | 42 (43 for the swing run) |
| Hardware | training on MPS (Radeon Pro 5300M); eval and latency on CPU |

## What would falsify this

If the primary refiner at seed 42 gains less than +0.005 mIoU **and** less
than +0.005 Boundary IoU, a small residual refiner on LR-ASPP's own
training-set logits does not fix the masks. This then holds for both 013's and
this input and loss choice, so the ablations in *Registered runs* are dropped
rather than run. If the hold-out CE improves but val does not, that is the
train/val logit gap, reported as the reason. Its only test is cross-fitted
logits (013's *Next*).

## Related

- `013-refinement-cnn`: the same question with RGB + log-probs on a x4 unshuffled view and CE. Falsified: +0.0025 BIoU, −0.0096 mIoU.
- `014-boundary-displacement`: a SegFix-style refiner, +0.019 BIoU at k8 s8 (seed 42, unfinished).
- `016-boundary-loss`: a boundary CE term on the segmenter, +0.005 BIoU.
- `builds/seg-boundary-eval`: the harness and the baseline.
