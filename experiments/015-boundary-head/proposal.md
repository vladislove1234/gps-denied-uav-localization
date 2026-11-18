# 015-boundary-head — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

Does adding a binary boundary head to LR-ASPP (multi-task: semantic mask + class-edge band,
read from stride-4 features) make the satellite mask sharper, (a) through the auxiliary task
alone, measured on the semantic argmax, and (b) by using the predicted boundary map at
inference to re-place the edges (boundary-guided watershed)?

## Hypothesis

(a) The boundary target forces the shared early backbone stages (stride 4 and 8, which also
feed LR-ASPP's `low` branch) to encode where class edges are. The `low` branch is the only
stride-8 input of the LR-ASPP classifier, so sharper low-level features should move the
semantic edges closer to the truth. Expected effect is small; the harness shows most
boundary loss is misplaced or missing edges, which a feature-level nudge can partly address.

(b) The head predicts the band at stride 4 from stride-4 features, i.e. twice the
resolution of the semantic output. Where it is right, flooding the band from the confident
interior (boundary prob < τ) with the boundary probability as elevation puts each edge on
the predicted ridge instead of where the ×8 bilinear upsample leaves it. Mechanism limit,
stated now: labels are only re-assigned inside the predicted band, from neighbouring seeds,
so a missed track or building cannot appear. The oracle run (GT band fed to the same fusion)
prices that ceiling.

## Success criterion

**Metric:** `biou_mean` — Boundary IoU at d = 4 px, unweighted mean over birky-strip,
birky-strip-google and bushcha, from `uavloc.segmentation.eval` (harness baseline 0.123).

**Baseline:** the **control fine-tune** of the same seed: the same model, init, data,
sampler, steps, LR schedule and seed, trained **without** the boundary head (`015.1` for
seed 42, `015.3` for seed 43). Not the original checkpoint, because extra training alone can
move the numbers. The original checkpoint is re-scored in every run (`ref_*` keys) as a
sanity check against the harness (0.476 / 0.123).

**Bar (a), auxiliary task:** multitask semantic argmax `biou_mean` ≥ control `biou_mean`
+ 0.010, at seed 42 **and** seed 43 (paired by seed).

**Bar (b), inference fusion:** watershed fusion at τ = 0.5 `biou_mean` ≥ the same run's
multitask semantic argmax `biou_mean` + 0.010, at seed 42 **and** seed 43.

**Guard (both):** `miou_mean` must not drop by more than 0.01 vs the thing it is compared to
(control for (a), multitask argmax for (b)).

**Latency budget (b):** boundary head + fusion ≤ 40 ms median per 512 × 512 tile, CPU,
4 threads (`uavloc.segmentation.latency`, i7-9750H proxy; LR-ASPP itself ~114 ms idle, so
budget ≈ 35% of the model, total ≤ ~155 ms). Other experiments load this machine
concurrently, so the budget is checked as a **ratio** timed in the same cell:
(multitask forward − LR-ASPP forward + fusion) / LR-ASPP forward ≤ 0.35. Absolute ms are
reported alongside. A config that clears the bar but not the budget is "works, too slow".

Diagnostics (not criteria): BF1@1, BF1@2, per-class IoU and Boundary IoU, pixel
precision/recall/F1 of the predicted band against the training target, fusion at τ = 0.3
and 0.7, oracle fusion (GT band as the boundary map, same watershed, on the multitask argmax).

## Registered runs

| run | arm | seed |
|---|---|---|
| 015.1 | control (no head) | 42 |
| 015.2 | multitask (head) | 42 |
| 015.3 | control | 43 |
| 015.4 | multitask | 43 |

Seed 43 runs only if compute allows; if they are not run, the verdict is "seed 42 only,
unconfirmed" and the bar is reported as not met (it requires both seeds).

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `builds/seg-boundary-eval` (`uavloc.segmentation.eval`, `.boundary`, `.metrics`, `.latency`) | exists | — |
| `uavloc.segmentation.refine.boundary_head` | to build (this experiment) | head-less model's `coarse_logits` identical to the checkpoint's (max abs diff 0); target band looks right on a real crop (figure); `ref_*` reproduces the harness |
| `pocs/village-seg-baseline/village_seg.py` samplers + holdout ban | exists | same training ground as LR-ASPP (dubrova + birky below row 704) |

**What only you can provide.** Nothing.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated` satellite images, loaded like `probe.ipynb` (`vs.load_satellite`, `vs.ban_holdouts`) |
| Split | train: every satellite image off the banned area (bushcha whole, birky rows < 704); val: the 3 harness sets |
| Sampler | `vs.SatelliteSampler(crop 512, rare_p 0.3)`, batch 4, satellite only (drone domain out of scope) |
| Seed | 42 (second seed 43) |
| Init | `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` (the probe's weights with satellite BN statistics) |
| BatchNorm | frozen on the satellite running statistics (BN modules in eval mode; affine params train). Single-domain batches cannot corrupt them, and train/eval BN are identical |
| Model | LR-ASPP/MobileNetV3-L unchanged + boundary head: 1×1 reduce of stride-4 (features[3], 24 ch) and stride-8 `low` (features[4], 40 ch, upsampled) to 16 ch each, concat, two 3×3 conv (16 ch, ReLU), 1×1 → 1 logit at stride 4, bilinear ×4. 8 k params, 0.12 GMAC per 512 tile (model 2.08) |
| Boundary target | pixels within 2 px (chessboard) of a class edge (3×3 label change, dilated by 1 px → a 4 px band); an edge needs two labelled classes; pixels within 3 px of ignored ones are not scored |
| Loss | CE + soft Dice on the ×8 bilinear semantic logits (round-3 loss) + 1.0 × class-balanced BCE (positives weighted n_neg / n_pos per batch) on the ×4 upsampled boundary logits. Control: CE + Dice only |
| Optimiser | AdamW, weight decay 1e-4; OneCycle, 5% warm-up; max LR 1e-4 for pretrained weights, 1e-3 for the new head |
| Steps | 1500 (≈ 0.7-0.9 s/step on MPS under the current shared load, ~20-25 min per run) |
| Fusion (b) | markers = semantic argmax where boundary prob < τ; `skimage.segmentation.watershed` on the boundary probability; τ = 0.5 registered |
| Hardware | MPS training and scoring; CPU (4 threads) latency |

## What would falsify this

- (a) If the multitask argmax gains < 0.005 `biou_mean` over its control at seed 42, or the
  two seeds disagree in sign: auxiliary boundary supervision does not sharpen this model's
  semantic edges; drop it as a training trick for LR-ASPP.
- (b) If fusion gains < 0.005 over the multitask argmax while the oracle fusion gains
  ≥ 0.03: the fusion is sound but the stride-4 head cannot find the edges on this imagery
  (check band F1). If the oracle itself gains < 0.03: re-assigning labels inside a band is
  not the lever, because the argmax is wrong far from the edges.

## Related

- `builds/seg-boundary-eval` — harness, baseline numbers, oracle gap.
- `006-segmentation-ceiling` — dropped boundary heads for localisation; this series measures
  mask sharpness only, by the user's decision (2026-09-28).
- Parallel mask-sharpening experiments 009-016 (e.g. `014-boundary-displacement`: a separate
  RGB-only SegFix net; `016-boundary-loss`: a loss on the semantic output). 015 differs in
  sharing the backbone with the segmenter.

## Amendments

- **2026-09-28, before any logged run.** Latency is measured as (boundary head timed alone on
  cached backbone features + fusion) / LR-ASPP forward, in 3 interleaved rounds (median),
  instead of (multitask forward − LR-ASPP forward). Under the shared load the difference of two
  ~300 ms timings came out negative in the smoke test, so it cannot price an ~10 ms head. The
  budget (ratio ≤ 0.35) is unchanged. The watershed now floods only the band plus a 1-px rim
  (`mask=`); checked bit-identical to the unmasked watershed on a 512² random case.
