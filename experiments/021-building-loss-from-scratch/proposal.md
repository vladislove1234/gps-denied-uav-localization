# 021-building-loss-from-scratch — proposal

> Slug kept from the first draft; the experiment covers tracks too (Q2, added 2026-09-29 before any run).

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

When U-Net/ResNet18, U-Net/ResNet34 and LR-ASPP/MobileNetV3-L are trained from
ImageNet initialisation on the satellite labels:

- **Q1 (buildings):** does a building-aware loss (gap weight + per-building
  weight + Tversky on building) lower both the share of missed buildings and the
  separable merge rate, compared with the same training on the current CE + Dice?
- **Q2 (tracks):** does adding track terms on top of it (skeleton recall on the
  track centreline + Tversky on track, the `structure` loss) break labelled
  tracks into fewer pieces and cover more of their centreline, compared with the
  building-aware loss alone?

The two questions have separate arms, so each change is measured against a run
that differs from it in that change only.

## Hypothesis

Plain CE barely sees small buildings or the gaps between them. A 200 px house or
a 3 m gap is a few hundred pixels against hundreds of thousands, so missing or
filling it costs almost nothing. Current LR-ASPP misses 24.8 % of val buildings,
while a perfect stride-8 map misses under 1 % (`pocs/building-merge-stride`).
So missed buildings are model errors, not a resolution limit.

`019` showed the model does respond to a pixel weight. Gap weight w₀ = 3 cut
separable merges by 35–42 % at two seeds, but the missed share rose by +0.025,
because the model opened gaps partly by dropping buildings. This experiment adds
two terms that make a miss expensive:

- **Per-building weight:** each building's pixels get extra CE weight
  `min(1046 / area_px, 10)`, so a small house counts about as much as a
  median one (1046 px, 155 m² in the training labels).
- **Tversky on building** (α = 0.3, β = 0.7): a missed building pixel costs more
  than a false one.

Training from scratch rather than fine-tuning lets the loss shape the model
from the start. `019` was 400 steps on a converged checkpoint. Three
architectures test whether the effect depends on the output stride: U-Nets
decode at full resolution, LR-ASPP at stride 8.

**Tracks (Q2).** For tracks the failure is broken connectivity, not missed
objects: a road predicted as dashes scores well on IoU but gives the matcher
fragments instead of a line, and in the correlator the track is often the only
structure in a frame. Skeleton recall (Kirchhoff et al., ECCV 2024) is `1 −`
the share of the labelled centreline, thickened by 1 px and kept inside the
track, that the model predicts as track. A one-pixel break costs as much as the
centreline it cuts, however wide the road. It also suits our labels: OSM tracks
are drawn 2–3× too wide and LandCover.ai over-claims track, but the centreline
is right even when the width is not. Tversky on track (α 0.3, β 0.7) adds the
same recall lean as for buildings.

Stated limits:
- **More false buildings.** β > α and larger weights push recall up and can pay
  for it in precision. Building IoU and mIoU are guarded for that reason.
- **Label errors look like misses.** bushcha's buildings come from OSM and MS
  footprints, not from hand-drawing. No loss fixes a building that is not in
  the image.
- **Crop edges.** Buildings cut by the 512 px crop edge look small and get a
  larger per-building weight.
- **Fewer fragments by predicting less track.** Fragments per km falls if the
  model drops tracks altogether. Centreline recall is in the bar and precision
  and track IoU are guarded for that reason.
- **Track label quality.** The skeleton follows the label; where a labelled
  track is not in the image, the loss pushes for a track that is not there.

## Success criterion

### Q1: buildings (`building` vs `current`)

**Metrics** (full val images; unweighted mean over `birky-strip`,
`birky-strip-google` and `bushcha`, definitions from `uavloc.segmentation.instances`):
- `building_missed_share_mean`: labelled buildings that no predicted component
  covers ≥ 20 %.
- `separable_merge_rate_mean`: among buildings a perfect stride-8 map keeps
  separate, the share the model merges (as in `019`).

**Baseline:** for each architecture and seed, the same run with `LOSS=current`
(CE + Dice). Same init, steps, schedule, sampler stream and seed.

**Bar**, per architecture and seed, building-aware vs current:
- `building_missed_share_mean` ≤ current − 0.03, **and**
- `separable_merge_rate_mean` ≤ 0.85 × current.

**Guards** (all three, or the pair does not count):
- `miou_mean` ≥ current − 0.02
- building IoU (mean over sets) ≥ current − 0.02
- `boundary_iou_mean` ≥ current − 0.01

### Q2: tracks (`structure` vs `building`)

**Metrics** (full val images, mean over the three sets, `uavloc.segmentation.topology`):
- `track_fragments_per_km_mean`: predicted track pieces (≥ 16 px) that touch a
  labelled track, per km of labelled centreline.
- `track_skeleton_recall_mean`: share of the labelled centreline with predicted
  track within 2 px.

**Baseline:** for each architecture and seed, the same run with `LOSS=building`.

**Bar**, per architecture and seed, structure vs building:
- `track_fragments_per_km_mean` ≤ 0.8 × building, **and**
- `track_skeleton_recall_mean` ≥ building + 0.03.

**Guards** (all four):
- track IoU (`track_iou_mean`) ≥ building − 0.02
- `track_skeleton_precision_mean` ≥ building − 0.05
- `miou_mean` ≥ building − 0.02
- the building result is kept: `building_missed_share_mean` ≤ building + 0.01
  and `separable_merge_rate_mean` ≤ building + 0.02

### Replication and verdict (both questions)

An architecture is **confirmed** for a question when it passes at both seed 42
and seed 7. A loss **works** when at least 2 of the 3 architectures are
confirmed. One architecture passing at one seed is reported as "not resolved",
not as a pass. Q2 is judged on its own: it can pass even if Q1 fails, since its
baseline is the building run either way.

**Also reported, not part of the bar:**
- merged / separate shares, the resolution-bound merge rate, and missed and
  merged share by building area and by gap to the nearest building
- mIoU, per-class IoU, mean Dice, Boundary IoU, BF1 at 1 and 2 px
- track clDice, centreline precision, and the label's own pieces per km (the floor
  for fragments)
- `structure` vs `current` on every metric (the whole bundle; descriptive)
- train and val curves of every loss term and metric (TensorBoard, `history-*.csv`)
- CPU latency on a 512 tile, on the machine that ran it (not comparable across
  machines)
- the architecture comparison on the current loss. It is descriptive only: one
  loss, two seeds, no bar.

## Registered runs

| Run | Architecture | Loss | Seed | Role |
|---|---|---|---|---|
| `021.1` / `021.2` / `021.3` | U-Net/ResNet18 | current / building / structure | 42 | triplet |
| `021.4` / `021.5` / `021.6` | U-Net/ResNet34 | current / building / structure | 42 | triplet |
| `021.7` / `021.8` / `021.9` | LR-ASPP/MobileNetV3-L | current / building / structure | 42 | triplet |
| `021.10` – `021.18` | the same three triplets | | 7 | replication |

> **2026-09-29, after the batch-4 pilot: steps and pilot runs.** `021.1` and `021.2`
> (U-Net/ResNet18, current and building loss) ran on the original batch 4 ×
> 10,000 recipe. They are kept as a **pilot pair** and are not part of the bar.
> 021.1's val mIoU on the monitor crops peaked at step 2,000 (8k crops) and fell
> after it while train mIoU kept rising: with 8 training images, long training
> overfits. The registered recipe is therefore batch 16 × **2,500** steps (40k
> crops per run), lr 1e-3, log every 50 steps, a checkpoint every 500 steps kept
> for inspection (the final step is still the one evaluated). Registered runs get
> ids from `021.3` on and are identified by architecture, loss and seed. The bar
> and guards are unchanged. Known weakness, not changed: the label itself has
> ~21 track pieces/km against the model's ~5, so `track_fragments_per_km` mostly
> measures label fragmentation; the report treats track centreline recall as
> the more telling Q2 number.

> **2026-09-29, before any run was logged: batch size.** Batch 4 × 10,000 steps
> (lr 5e-4) became batch 16 × 5,000 steps (lr 1e-3, square-root scaling), log
> every 125 steps. Batch 4 only mattered for matching the earlier checkpoints'
> recipe, and those are no longer compared here. The larger batch gives BatchNorm
> steadier statistics and uses the 24 GB GPU; each run sees 80k crops instead of
> 40k. Same recipe for all 18 runs; the bar and guards are unchanged.

> **2026-09-29, before any run: server and run ids.** All runs train on one Linux
> server (RTX 3090 Ti; env `requirements-server.txt`, steps in `server.md`). If
> architectures run in parallel (one tmux window each), `log_run` hands out ids
> in finishing order, so an id need not match the table below. A run is then
> identified by `config.architecture`, `config.loss` and `seed`, which the
> report uses instead of the id. Nothing else changes.

Q1 compares `building` with `current`, Q2 `structure` with `building`, within
each triplet. Every model is trained anew from ImageNet encoder weights with all
three losses. No
earlier trained checkpoint (round-3 U-Net, current LR-ASPP) is loaded, trained or
scored here; they stay untouched. Run ids are assigned by `log_run` in logging
order, so the order above is the order to run them in (`run_all.sh 42`, then
`run_all.sh 7`).

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.segmentation.losses.StructureLossConfig`, `structure_loss_terms`, `instance_weights`, `tversky_loss`, `class_skeleton`, `skeleton_recall_loss` | added (this experiment) | `checks.ipynb`, on real training crops: `CURRENT_LOSS` equals `ce_dice_loss` to float precision (< 1e-5; sum/n vs mean); the per-building weight is 0 off buildings and larger on small ones (figure); the gap band equals `019`'s `gap_band`; skeleton recall loss of a perfect prediction is 0; tubed centreline figure |
| `uavloc.segmentation.topology.track_scores` | added | `checks.ipynb`: the label scored against itself gives recall = precision = clDice = 1 and fragments = the label's pieces; cutting the label every 40 px raises fragments per km |
| `uavloc.features.segmentation.build_segmenter` (3 architectures, ImageNet init) | added | builds, forwards a 512 crop to `(B, 6, 512, 512)` |
| `uavloc.segmentation.eval.model_predictions`, `score_sets`; `instances.status_shares`, `status_by_area`; `latency.segmenter_latency`; `track.resolve_device` | added | smoke run of every architecture × loss completes and logs to TensorBoard |
| `monitor.crop_metrics` extended with Dice, Boundary IoU, BF1, per-class IoU, building shares, track scores | changed | the same smoke run |
| `ua-village-annotated`, `pocs/village-seg-baseline/village_seg.py` (sampler, BN recalibration) | exist | — |

**What only you can provide.**
- A go-ahead on the bar, the guards and the loss weights.
- The server run. It needs the repo, `data/derived/ua-village-annotated` and
  `ua-village-multidate` (the imagery the labels point to), an env with
  `segmentation-models-pytorch==0.3.4`, and internet for the ImageNet weights.

## Setup

| | |
|---|---|
| Dataset | `ua-village-annotated`: train = birky (rows below the strip plus a 64 px gap, all dates) and dubrova (all dates); val = the harness's three sets. Satellite only, no drone frames |
| Split | as round 3 and `019` (`vs.ban_holdouts`), unchanged |
| Seeds | 42, replication 7 (torch and sampler RNG) |
| Models | `unet-resnet18`, `unet-resnet34` (smp 0.3.4), `lraspp-mobilenetv3` (torchvision): ImageNet encoder weights, randomly initialised decoders and heads. None of our earlier trained checkpoints |
| Loss, current | CE + soft Dice (`CURRENT_LOSS`) |
| Loss, building | `mean(w · CE) + Dice + 1.0 · Tversky(building; α 0.3, β 0.7)`, `w = 1 + 3 · gap_band(r = 10 px) + 1.0 · min(1046 / area, 10)` on building pixels |
| Loss, structure | building + `1.0 · SkeletonRecall(track; centreline dilated 1 px, inside the track)` + `1.0 · Tversky(track; α 0.3, β 0.7)` |
| Training | 2,500 steps, batch 16 × 512 px (40k crops per run), `SatelliteSampler(rare_p = 0.3)`, AdamW lr 1e-3, OneCycle with 5 % warm-up, wd 1e-4. BN re-estimated from 60 training batches at the end. The last step is evaluated, never the best |
| Logging | every 50 steps and step 0: every loss term and metric on 90 fixed train crops and the 90 val crops, running loss, lr, preview grids (TensorBoard). Final: full val images, `runs.jsonl`, `artifacts/history-*.csv`, `artifacts/results-*.json` |
| Hardware | server GPU (device recorded per run). All 18 runs on the same machine, since RNG streams differ across devices |

## What would falsify this

- **Missed share not down by 0.03 on any architecture at seed 42** → reweighting
  the loss does not make LR-ASPP or a U-Net find the buildings it misses. The
  misses are then a data or label problem (bushcha's labels, too few villages),
  and the next step is labels, not losses.
- **Missed share down, merges up (or the reverse), on every architecture** →
  the terms fight each other. The report says so, and the weights are not tuned
  on the val sets to hide it.
- **The bar is met but a guard fails** → the loss trades buildings for false
  positives. That is reported as the result.
- **Q2: fragments not down by 20 % on any architecture at seed 42** → the track
  breaks are not a loss problem. With stride-8 LR-ASPP the likely cause is
  resolution; for the U-Nets, the labels. Topology losses (clDice, persistent
  homology) are not tried as a rescue.
- **Q2 passes only by dropping tracks** (recall or precision guard fails) →
  reported as the result.

## Related

- `019-building-gap-loss`: the gap weight; merges down, missed share up.
- `016-boundary-loss`: equal weight on every class edge; +0.005 BIoU.
- `010-dense-crf-refinement`: Potts smoothing eats thin tracks; connectivity is the fragile part.
- `001`/`002`/`005`: the track is often the only structure the correlator can use.
- Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures", ECCV 2024. Shit et al., "clDice", CVPR 2021.
- `pocs/building-merge-stride`: missed and merged shares, stride-8 oracle.
- `pocs/lraspp-stride4-decoder`: a finer decoder alone changes nothing without a loss that asks for gaps.
- `pocs/mobilenet-lraspp-speed`, `pocs/village-seg-baseline` (round 3): the from-scratch recipe (not their checkpoints).
