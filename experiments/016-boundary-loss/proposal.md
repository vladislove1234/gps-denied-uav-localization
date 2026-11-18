# 016-boundary-loss — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does adding a boundary-band cross-entropy term to the LR-ASPP training loss
(`L = CE + Dice + λ · CE_band`) raise satellite Boundary IoU by at least 0.010
over the same fine-tune with λ = 0, without costing more than 0.01 mIoU?

## Hypothesis

The harness (`builds/seg-boundary-eval`) found that most of LR-ASPP's boundary
loss is misplaced or missing edges, not upsample blur: the model scores 0.123
Boundary IoU against a 0.44–0.68 stride-8 oracle. CE + Dice is dominated by
interior pixels (only ~15% of labelled pixels in a 512 training crop lie within
4 px of a label edge, measured on a real batch). Up-weighting the loss on that
band should push the stride-8 head to place class transitions where the label
has them. The architecture and the inference path do not change, so latency
cannot change.

## Success criterion

**Metric:** `boundary_iou_mean` — Boundary IoU (d = 4 px, fixed by the harness),
unweighted mean over `birky-strip`, `birky-strip-google`, `bushcha`, bilinear
argmax of the fine-tuned model.

**Baseline:** the **λ = 0 control run** of this experiment (same checkpoint,
steps, LR schedule, sampler stream and seed 42). Not the original checkpoint:
fine-tuning alone may move the numbers. Run `016.1` re-evaluates the original
checkpoint through this notebook as a reference (harness: 0.123 BIoU,
0.476 mIoU).

**Bar:** the best λ run at seed 42 has `boundary_iou_mean` ≥ control + **0.010**.

**Guard:** its `miou_mean` ≥ control − 0.01. A run that clears the bar but fails
the guard does not count.

**Replication:** if time allows, the control and the best λ are re-run at seed 7.
The claim is "confirmed" only if the seed-7 delta is also positive; with one seed
the verdict is reported as unreplicated. λ is picked on the val sets (three
values), so a single-seed pass carries selection bias — the seed-7 re-run is
the check on it.

**Latency budget:** model CPU median ≤ 125 ms per 512 tile (baseline 114 ms,
i7-9750H at 4 threads as a Pi-class proxy; the margin is measurement noise on a
shared machine). Unchanged by construction: identical architecture and
parameter count, which the notebook asserts.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.eval`, `.latency`) | exists | run `016.1` reproduces 0.123 BIoU / 0.476 mIoU for the original checkpoint |
| `uavloc/segmentation/losses.py` | to build | `ce_dice_loss` equals probe.ipynb's inline `seg_loss` (checked: max abs diff 0.0 on a real batch); `boundary_band` covers ~15% of labelled pixels at `band_px = 4`. The notebook sums `ce_dice_loss + λ · boundary_ce`, so λ = 0 gives exactly the CE + Dice gradient |
| `pocs/village-seg-baseline/village_seg.py` (sampler, holdout ban, BN re-estimation) | exists | — |
| `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` | exists | — |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated`: train = birky (rows below strip + 64 px gap, 5 dates) and dubrova (3 dates); val = the harness's three sets |
| Split | as `probe.ipynb` / round 3 (`vs.ban_holdouts`), unchanged |
| Seed | 42 (sampler RNG and torch); replication seed 7 |
| Model | LR-ASPP MobileNetV3-L, initialised from `model-lraspp-bn-satellite.pt` (satellite BN). Satellite-only fine-tune, so BN only ever sees satellite batches (cerebrum Do-Not-Repeat on mixed-domain BN does not apply); BN statistics re-estimated at the end with `vs.domain_bn` (60 batches, seed 123), as probe.ipynb did |
| Loss | `CE + Dice + λ · CE_band`, band = labelled pixels within 4 px (chessboard) of an edge between two labelled classes (edges against ignore are not edges, as in the metric). λ ∈ {0 (control), 0.3, 1, 3} |
| Training | 400 steps, batch 4 × 512 px crops, `SatelliteSampler(rare_p = 0.3)`, AdamW lr 1e-4 (OneCycle, 5% warm-up), wd 1e-4 |
| Hardware | training on MPS (AMD Radeon Pro 5300M, shared with other jobs: 1.2–6 s/step measured, ~4 s/step on CPU); evaluation and latency on CPU, as the harness baseline |

## What would falsify this

If every λ gives Δ BIoU < +0.005 over the control, or gains come only together
with an mIoU loss > 0.01, then re-weighting the loss toward edges does not
sharpen a stride-8 LR-ASPP on these labels, and the next step is on the
architecture side (higher-resolution head / skip features) rather than more
loss variants. Kervadec's distance-map boundary loss would then not be tried
either: it acts on the same band with the same labels.

A second, weaker failure: if the gain appears on birky (75% hand-drawn
buildings) but not on bushcha (100% source-layer labels), the loss is learning
source-model edges, not image edges, and the result does not generalise.

## Related

- `builds/seg-boundary-eval` — harness, metric, baseline numbers.
- `pocs/mobilenet-lraspp-speed` — the checkpoint and the CE + Dice loss.
- Label caveat: dubrova (training) and bushcha (val) labels are OSM /
  LandCover.ai / FLAIR source layers (~1.5 m residual), not hand-drawn.
