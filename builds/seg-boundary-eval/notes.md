# seg-boundary-eval

**Status:** ready for your review · **Unblocks:** `the 8 mask-sharpening experiments (boundary refiners for LR-ASPP)`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

Shared equipment for experiments that try to make the LR-ASPP satellite masks
sharper. It provides the three hand-labelled satellite val sets, the model's
output at two points (stride-8 logits and full-res probabilities) so a refiner
can plug in at either one, region and boundary metrics, a CPU latency helper,
and a single `evaluate(refine_fn, cases)` call. It also records the baseline
numbers that every experiment is measured against. Satellite only; the drone
set is out of scope.

## What I needed from you

Nothing.

## What I did

| Code | What |
|---|---|
| `uavloc/segmentation/eval.py` | `load_val_sets()`, `load_baseline_model(device)`, `lraspp_outputs(model, rgb, device) -> ModelOutput(logits, probs)`, `run_model(model, val_sets, device) -> cases`, `evaluate(refine_fn, cases, boundary_px=4, tolerances_px=(1, 2))`, `evaluate_predictions(preds, cases, ...)`, `results_table(results)`, `upsample_logits(logits, size, mode)`, refiners `bilinear_argmax` / `nearest_argmax`, oracle `label_at_stride(label, mode)` |
| `uavloc/segmentation/boundary.py` | `boundary_scores(preds, labels, boundary_px, tolerances_px)`: Boundary IoU and boundary F1, per class and as a mean. Also `inner_band`, `class_band`, `label_band` |
| `uavloc/segmentation/metrics.py` | `iou_scores` (moved from `village_seg.py`, same output), `segmentation_scores` (everything in one dict) |
| `uavloc/segmentation/latency.py` | `cpu_latency(fn, warmup=5, runs=30, threads=4) -> Latency(median_ms, p10_ms, p90_ms, runs, threads)`, `standard_tile(rgb, top, left, 512)`, `model_latency`, `refiner_latency` |
| `uavloc/segmentation/refine/__init__.py` | Empty. Each experiment adds its refiner module here |
| `uavloc/data/annotated.py` | `AnnotatedImage`, `uncovered`, `load_annotated_images(dataset)`, moved out of `village_seg.load_satellite` |
| `uavloc/features/segmentation.py` | `LRASPPLogits.coarse_logits(x)`: the head output before torchvision's ×8 bilinear upsample. `LRASPP_OUTPUT_STRIDE_PX = 8` |
| `uavloc/features/tiling.py` | `tiled_logits(..., out_stride=1)`: tiles a stride-8 model on the stride-8 grid. The default leaves existing behaviour unchanged |

`pocs/village-seg-baseline/village_seg.py` now imports `load_annotated_images`
and `iou_scores` from `uavloc` rather than defining them. Its signatures and
outputs are unchanged, and I re-ran and checked it.

**Val sets** match `pocs/mobilenet-lraspp-speed/probe.ipynb` exactly:
`birky-strip` is the four Esri dates of birky, rows `[:640]`. `birky-strip-google`
is the same strip on the Google date. `bushcha` is both dates of the whole
unseen village (2560×3328). The strip each date's alignment offset leaves
uncovered is ignored (255).

**`ModelOutput`**:
- `logits`: `(C, Hp/8, Wp/8)`, the head output on the image reflect-padded to multiples of 8 (32 for single-pass images). Cell `(i, j)` covers pixels `[8i, 8i+8) × [8j, 8j+8)`.
- `probs`: `(C, H, W)`, softmax of `upsample_logits(logits, (H, W))`. This is bit-identical to `model(x)` (checked: max abs difference 0.0 on a real tile).

Large images are tiled the same way as `class_probabilities` (512 px windows, step 384), but the average is taken over the stride-8 logits.

**Metrics** are accumulated per class over the whole set (sums of intersection
and union, or of matched and total boundary pixels), then averaged over the
classes that have labelled pixels in the set. This is the same class-presence
rule as mIoU. `field` never appears. `water` appears in the birky strip (0.1%)
and scores 0 there, which pulls those means down.
- **Boundary IoU** (Cheng et al. 2021): IoU of the inner bands `mask ∧ ¬erode(mask, 3×3, d times)` of the prediction and of the label, per class.
- **BF1@t**: F1 between 1-px inner contours. A contour pixel counts as matched when it lies within Euclidean distance t of the other contour (dilation with a disk of radius t).
- **Two departures from the reference code, both deliberate.** (1) The image edge is not a contour. The reference zero-pads, but the val images are crops. (2) A contour against ignored (255) pixels is not a contour in either the prediction or the label, and ignored pixels are never counted.
- **`mean`** in `evaluate()` is the unweighted mean of the three sets.

**Chosen d = 4 px (1.54 m at 0.385 m/px)**, with reasons:
- **Label precision.** The labels are OSM, LandCover.ai and FLAIR seeds, aligned per date and curated by hand. The best residual alignment measured here is about 1.5 m (Web Mercator OSM offset, 1.51 m). A band narrower than about 4 px would mostly score label noise.
- **It must see the stride-8 blur.** One stride-8 cell is 8 px (3.1 m). At d = 4 the band is half a cell, so staircase and rounding errors of a few pixels fall inside it. With a perfect stride-8 label and nearest upsample the score is only 0.44 (table below), so the metric is not saturated by the resolution we want to beat.
- **It still separates edge from interior.** Village tracks are about 3–5 m wide (8–13 px) and buildings 15–25 px. At d = 4 the band covers at most about half of a track, not the whole object.
- **The ranking does not depend on d.** In the sweep over d = 1…8 px (`figures/biou-vs-d.png`), the order bilinear > nearest and the oracle gap hold at every d. If an experiment's claim changes with d, report the sweep.

Run from `src/`:
`PYTORCH_ENABLE_MPS_FALLBACK=1 papermill builds/seg-boundary-eval/verify.ipynb builds/seg-boundary-eval/runs/verify.ipynb -k python3`
Takes about 2 min on CPU. The output folder `runs/` is gitignored.

## Does it work

**Baseline to cite.** The model is LR-ASPP with satellite BN
(`pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt`), run on CPU. The
metrics use d = 4 px and BF tolerances of 1 and 2 px. Source: `results.json`
and `runs/verify.ipynb`.

| prediction | set | mIoU | Boundary IoU | BF1@1 | BF1@2 |
|---|---|---|---|---|---|
| **LR-ASPP, argmax of bilinear (baseline)** | birky-strip | 0.460 | 0.126 | 0.138 | 0.212 |
| | birky-strip-google | 0.451 | 0.120 | 0.133 | 0.205 |
| | bushcha | 0.516 | 0.123 | 0.138 | 0.213 |
| | **mean** | **0.476** | **0.123** | **0.137** | **0.210** |
| LR-ASPP, stride-8 argmax, nearest | mean | 0.467 | 0.115 | 0.138 | 0.218 |
| oracle: label → stride 8, nearest | mean | 0.860 | 0.439 | 0.459 | 0.703 |
| oracle: label → stride 8, bilinear | mean | 0.930 | 0.677 | 0.792 | 0.902 |

The two oracles average the one-hot label over each 8×8 cell, upsample it, and
take the argmax. They show the best any stride-8 prediction can reach with that
upsample.

**CPU latency** is measured on one 512×512 tile (bushcha 2024-06-06 at
(1024, 1024)), i7-9750H, torch capped at 4 threads (Pi-class core count), 5
warm-up runs then 30 timed runs:

| | median ms | p10–p90 |
|---|---|---|
| model: normalise → stride-8 logits → bilinear → softmax | **113.8** | 103.5–127.9 |
| refiner: argmax of bilinear probs (baseline) | 4.2 | 4.0–4.5 |
| refiner: stride-8 nearest argmax | 4.9 | 4.8–5.2 |

Baseline end-to-end is about 118 ms per 512 tile on this CPU. This is not a
Raspberry Pi time.

What the numbers say:
- **Bilinear upsampling is worth only a little**: +0.009 mIoU and +0.008 Boundary IoU over stride-8 nearest. BF1@2 is actually 0.008 lower, because bilinear rounds corners.
- **The model sits far below the stride-8 ceiling.** It reaches 0.123 Boundary IoU, against 0.44 for a perfect stride-8 map with nearest upsampling and 0.68 with bilinear. Most of the boundary loss is misplaced or missed edges, not upsampling blur. In `figures/zoom-baseline.png` the buildings are round blobs and the tracks are fragmented and missing.
- Per-class Boundary IoU on the baseline:

  | set | background | building | track | woody | water |
  |---|---|---|---|---|---|
  | birky-strip | 0.19 | 0.21 | 0.19 | 0.04 | 0.00 |
  | birky-strip-google | 0.17 | 0.21 | 0.16 | 0.06 | 0.00 |
  | bushcha | 0.13 | 0.18 | 0.17 | 0.09 | 0.04 |

  Woody is the worst class. The full table is in `results.json` → `runs`.
- The mIoU reproduces `probe.ipynb` to within 0.0004: 0.4597 / 0.4514 / 0.5162 against 0.4596 / 0.4510 / 0.5161. The notebook asserts this. The small gap comes from averaging stride-8 logits across tiles instead of full-res logits, and from CPU vs MPS.

Metric sanity checks (asserted in `verify.ipynb`, figure `figures/synthetic-shift.png`):
- A mask scored against itself gives 1.0 on every metric.
- Randomising the pixels under ignore also gives 1.0.
- Shifting the prediction k px diagonally makes Boundary IoU fall strictly and every other metric fall or stay flat, up to k = 4 on a thin-structure "village" scene and up to k = 12 on a large-block scene.
- On the large-block scene, a 2 px shift costs mIoU 0.037 but costs Boundary IoU 0.612.
- **Beyond about half the width of a thin structure, the metrics are not monotone.** A 12 px shift of a 12 px track lines its top edge up with the old bottom edge, and BF1 rises again. This is a property of the metric, not a bug. Refiners move edges by 1–4 px, so it does not affect them.

**Label source (question 7).** The LR-ASPP satellite branch was trained on
`ua-village-annotated`, the annotator's hand-corrected export. It was **not**
trained on the OSM `label.png` from `ua-village-multidate`. The evidence:
`probe.ipynb` loads `vs.load_satellite(DERIVED / "ua-village-annotated", ...)`
(cell 6, the same as `round3.ipynb`), and `samples.jsonl` pairs each
`ua-village-multidate/<loc>/<date>.jpg` with `ua-village-annotated/<loc>/label-<date>.png`.
The labels follow each date's image (per-date `image_offset_px`, uncovered strip
ignored), so they are aligned to the imagery. The known 2–6 m misalignment of
the OSM `label.png` does not apply.

There is one caveat for boundary losses. "Hand-corrected" does not mean
hand-drawn. I read each record's `provenance` field as the share of the class's
pixels covered by each sublayer; this is my reading of the field, not checked
against the annotator code. On that reading:
- birky buildings are about 75% manual.
- birky woody is 100% the OSM sublayer.
- dubrova (training ground) and bushcha (val) are 100% source sublayers: OSM buildings, LandCover.ai track and FLAIR woody, with almost no manual pixels (except dubrova's Google date).

So the track and woody boundaries on dubrova and bushcha are LandCover.ai and
FLAIR edges, aligned and curated but not redrawn. A boundary loss trained on
them learns those models' edges, and Boundary IoU on bushcha partly scores
agreement with them.

**Not verified:**
- Latency on a real Raspberry Pi. ARM kernels for hardswish/SE may differ.
- An independent measurement of label boundary precision. There is no second labelling of the same ground, so d rests on the alignment residuals above.
- The metrics against the official `boundary-iou-api`. The code follows the paper's definition, with the two departures above.

## Your call

- Look at `figures/zoom-baseline.png`, `figures/synthetic-shift.png` and `figures/biou-vs-d.png`.
- **Decision: is d = 4 px acceptable** as the fixed dilation for all 8 experiments?
- The oracle gap suggests that refiners working only on the upsample (guided or joint-bilateral upsampling, CRF on stride-8 logits) have limited headroom from this model. Their ceiling is 0.44–0.68 Boundary IoU, but the model's stride-8 map itself only scores about 0.12. Worth keeping in mind when ranking the 8 ideas.
