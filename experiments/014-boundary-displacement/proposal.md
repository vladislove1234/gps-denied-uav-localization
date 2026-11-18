# 014-boundary-displacement — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does a SegFix-style boundary displacement refiner (Yuan et al., ECCV 2020), trained once
on the satellite training ground, raise the LR-ASPP satellite mask's mean Boundary IoU
(d = 4 px) by at least 0.010 without costing more than 0.01 mIoU or 40 ms CPU per 512 tile?

## Hypothesis

A small full-resolution network sees the RGB edges that the stride-8 LR-ASPP blurs. For
each pixel near a true class edge it predicts (a) that the pixel is a boundary pixel and
(b) which of 8 directions points into the interior of that pixel's own object. Replacing
the LR-ASPP label of a predicted boundary pixel with the LR-ASPP label `s` px along that
direction copies the (more reliable) interior prediction onto the edge. This fixes edge
pixels whose true class the model gets right a few px further in: blobby building
corners, bleed of one class into its neighbour.

Mechanism limits, stated now: it can only move labels the model already predicts
somewhere nearby. It cannot create a missed track or building (the harness shows most
boundary loss is misplaced or missing edges), and for thin tracks (8-13 px wide) a large
`s` crosses the object. The oracle run (GT boundary + GT direction applied to the
LR-ASPP map) prices that ceiling.

## Success criterion

**Metric:** `biou_mean` — Boundary IoU at d = 4 px, unweighted mean over birky-strip,
birky-strip-google and bushcha, from `uavloc.segmentation.eval.evaluate`.
**Baseline:** `014.1` — LR-ASPP bilinear argmax re-run through this notebook
(harness value 0.123).
**Bar:** `biou_mean` ≥ baseline + 0.010 for the best registered config, **and** the
same (k, s) config also clears baseline + 0.010 with the second seed.
**Guard:** `miou_mean` must not drop by more than 0.01 vs `014.1`.
**Latency budget:** refiner (SegFix net forward + offset gather, not the LR-ASPP model)
≤ 40 ms median per 512 × 512 tile, CPU, 4 threads (`uavloc.segmentation.latency`,
i7-9750H as a proxy; the model itself is ~114 ms on an idle machine). Other agents load
this machine (LR-ASPP measured 330-450 ms/tile while scoping), so the budget is checked
as a **ratio**: refiner median / LR-ASPP model median, both timed in the same cell, must be
≤ 0.35 (= 40 / 114). The absolute ms are reported alongside. A config that clears the bar
but not the budget is reported as "works, too slow".

Also reported (diagnostics, not criteria): BF1@1, BF1@2, per-class Boundary IoU and IoU,
direction accuracy and boundary precision/recall of the net on the val labels.

## Registered sweep

| knob | values | notes |
|---|---|---|
| boundary width k (training target) | 4, 8 px | one training per k |
| offset scale s (inference only) | 2, 4, 8 px | evaluated on the same trained net |
| boundary threshold | 0.5 | fixed |
| seed | 42; 43 for the best k | |
| input | RGB only | model-agnostic, as in SegFix; LR-ASPP logits are **not** used as input |

Selection of the best (k, s) is made on the same val sets the bar is scored on, so the
second-seed confirmation is part of the bar. Runs: `014.1` baseline, then one run per
(k, s, seed). Every run row also carries the oracle score for its (k, s).

**Oracle:** boundary = GT pixels within k px of a class edge, direction = GT direction,
applied to the LR-ASPP bilinear argmax with offset s. Reported as `oracle_biou_mean` etc.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.eval`, `.boundary`, `.metrics`, `.latency`) | exists | — |
| `uavloc.segmentation.refine.segfix` | to build (this experiment) | oracle run with s = 0 is identical to the baseline; targets look right on a real crop (figure) |
| `pocs/village-seg-baseline/village_seg.py` samplers + holdout ban | exists | same training ground as LR-ASPP (dubrova + birky below row 704) |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` satellite images, loaded like `probe.ipynb` |
| Split | train: every satellite image off the banned area (`ban_holdouts`: bushcha whole, birky rows < 704); val: the 3 harness sets |
| Seed | 42 (second seed 43) |
| Model | tiny encoder-decoder (16/32/64 ch, dilations 2 and 4 at stride 8, heads at stride 2), 0.14 M params; heads: 1 boundary logit (bilinear to full res) + 8 direction logits (argmax at stride 2, repeated) |
| Loss | BCE(boundary) on labelled pixels + CE(direction) on GT boundary pixels with a defined direction |
| Base segmenter | `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt`, frozen, bilinear argmax |
| Hardware | MPS training (~2500 steps, batch 8, 512 crops, ~10 min), CPU latency at 4 threads |

## What would falsify this

- If the oracle gains < 0.03 `biou_mean` over baseline at every (k, s): the LR-ASPP map
  has too little right-label-nearby for displacement to exploit; abandon the direction
  for this base model (the problem is missing edges, not misplaced labels).
- If every learned config gains < 0.005 while the oracle gains ≥ 0.03: the idea is
  sound but a small RGB-only net cannot predict boundary/direction on this imagery;
  record it and do not tune further here.

## Related

- `builds/seg-boundary-eval` — harness, baseline numbers, oracle gap.
- `006-segmentation-ceiling` — showed better masks barely move localisation; this series
  measures mask sharpness only, by the user's decision.
