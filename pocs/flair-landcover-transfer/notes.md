# flair-landcover-transfer

**Status:** parked — transfers, not usable as-is · **Question:** `Does IGN's FLAIR land-cover model, trained on French aerial imagery at 0.2 m, transfer to Ukrainian Esri tiles at 0.386 m/px?`

## The question

`IGNF/FLAIR-INC_rgb_15cl_resnet34-fpn`: FPN + ResNet-34, **RGB only** (Esri tiles
carry no NIR and no elevation, so the better `rgbie` variant is unusable),
trained by IGN on 174,700 photo-interpreted 512x512 patches of French BD ORTHO.
Etalab 2.0. Reported IoU: Water 83.1, Building 76.3, Brushwood 29.9, mIoU 59.3.

**This probe gates every aerial-pretrained-model route**, including
`landcoverai-finetune`. It costs no training.

It is **self-falsifying** — the property probe 1 lacked and probe 4 had.
`Building` and `Water` are exactly the classes where OSM is measured good, so a
model that misses buildings OSM proves are there cannot be trusted on `woody` or
`track`.

### The two risks, named before running

1. **GSD 1.93x.** Model wants 0.2 m, tiles are 0.386 m/px. Upsampling fabricates
   detail — the exact failure in `pocs/drone-frame-boundaries`, where a 131 px
   crop presented at 512 px returned nothing. Feeding native halves apparent
   object size. Both are run, one variable, same tiles.
2. **Sensor.** BD ORTHO is aircraft; Esri is a satellite mosaic. The model card
   states generalisation issues occur at other resolutions and sensors.

### Class mapping

| ours | FLAIR | note |
|---|---|---|
| building | Building | IoU 76.3 |
| water | Water + Swimming pool | IoU 83.1, its best |
| woody | Coniferous + Deciduous + Brushwood | Brushwood 29.9 — its **worst**, and treelines are exactly Brushwood |
| track | Impervious surface | **FLAIR has no road class**; a dirt track lands in Bare soil or Pervious surface. `track` is not solved here. |

## Verdict

**It transfers on `building`, and it is not usable as a label source as-is.**
Run 2026-09-23 over 9 locations, `runs/probe-2026-09-23.ipynb`.

| class | native IoU | null | recall | claims | OSM actual |
|---|---|---|---|---|---|
| building | **0.200** | 0.004 | 0.256 | 0.6% | 1.8% |
| water | 0.000 | 0.000 | 0.067 | 2.2% | 0.2% |
| woody | 0.001 | 0.000 | 0.927 | 25.5% | 0.03% |
| track | 0.011 | 0.001 | 0.011 | 0.15% | 2.88% |

**`building` is a real positional signal** — 50x its null, **9/9** locations
beat 2x null, and it *under*-predicts (0.6% claimed against 1.8% actual). That
is the opposite of YOLOE, which reached recall by painting 55.7% of a tile where
4.09% was building. In the figure the masks follow roofs, treelines and lanes,
with **no tile-aligned rectangles** — this is real segmentation, and that
qualitative difference is why the modest IoU is believable.

**The GSD question is answered: feed native, 8/9.** Median IoU 0.200 native vs
0.083 upsampled. This **contradicts the model card's own prior** — FLAIR was
trained at a fixed 0.2 m with no scale augmentation, so upsampling to match
should have won. It loses badly, because upsampling fabricates detail and the
model then labels the fabrication. Same mechanism as the satellite arm of
`pocs/drone-frame-boundaries`, measured here from the other side.

**`water` failed outright.** Native claims 2.2% where 0.2% exists; upsampled
claims **29.7%**, painting a third of the tile blue over dry farmland. Visible
immediately in `figures/flair-overview.png`.

**`track` failed, and the failure is the expected one.** Median IoU 0.011
against OSM road centrelines, recall 0.011, claiming 0.15% where OSM maps 2.88%.
The single exception is `smykivtsi` (IoU 0.145, recall 0.204) — the most
built-up of the nine and the only one with real paved roads. FLAIR's
`impervious surface` finds tarmac and nothing else, which is exactly the
paved-road bias the literature reports for unpaved tracks.

### The finding that decides it: cross-date instability

Two dates of the same co-registered ground, no ground truth needed:

| mode | pixel agreement | building IoU between dates |
|---|---|---|
| native | 44-57% | 0.22-0.46 |
| upsampled | 32-52% | 0.03-0.45 |

**The model disagrees with itself on more than half the ground when only the
date changes.** For a dataset whose entire purpose is invariance to season and
capture date, that disqualifies FLAIR as a drop-in label source. No OSM-based
score revealed this; the cross-date check did, at the cost of one extra
inference pass.

### What this unblocks

The gate on `landcoverai-finetune` was *"if FLAIR fails on building and water at
both GSDs, stop."* It did **not** fail on building. A Western-European
land-cover model does cross to Ukrainian Esri tiles, so LandCover.ai training is
worth its measured 1.3 hours.

### What is *not* established

- **`woody` and `water` were unscoreable here.** OSM `wood` covers a median
  0.03% of these tiles with **4/9 at exactly zero**, and water a median 0.23%.
  The recall of 0.927 for woody is against an empty truth and means nothing.
  The probe was self-falsifying for `building` only; for the two sparse classes
  the control cannot referee, which is the same sparsity that made them need a
  new source. **Design flaw in this probe, not a result about FLAIR.**
- **The null test loses power when both truth and prediction are sparse.**
  "9/9 beats 2x null" holds for `track` while its IoU is 0.011, because the
  null is 0.001. Read the null as a floor, never as a pass mark.
- **No fine-tuning or domain adaptation was tried.** The card recommends exactly
  that for other sensors. This measures the zero-shot transfer only.
- **9 locations, not 59.**

## What I ran

- `probe.ipynb` — weight download, tiled inference at both GSDs, scoring against
  offset-corrected OSM with a null baseline, plus the cross-date consistency
  check below.

**Cross-date consistency** is the part with no precedent in this repo: dates are
co-registered to 1.5 cm and share a label, so running the model on two dates of
the same ground scores it with **no ground truth at all**. Disagreement between
dates is pure source error, and it measures the invariance the dataset exists
for. A source that scores well against OSM but flips between dates is keying on
appearance.

**Install first** (under the existing pin, or it drags torch forward and breaks
`ultralytics`):

```
printf 'torch==2.2.2\ntorchvision==0.17.2\n' > /tmp/pin.txt
/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/pip install \
    "segmentation-models-pytorch==0.3.4" -c /tmp/pin.txt
```

**Do not name a timm version** — smp 0.3.4 hard-pins `timm==0.9.7` with `==`,
so any other value makes the resolve impossible (`bug-026`). Done 2026-09-23:
torch stayed 2.2.2, torchvision 0.17.2, `ultralytics` 8.4.150 still imports.

### Two silent failures caught before the first run

- **`bug-025`: checkpoint keys are prefixed `model.seg_model.`**, not `model.`,
  and the file also carries a non-network `criterion.weight`. Stripping only
  `model.` under `strict=False` yields a **randomly-initialised network that
  still runs and still returns plausible masks**. Load with `strict=True`.
- **`bug-024`: the head emits 19 logits, not 15.** "15cl" counts *active*
  classes; clear cut/mixed/ligneous/other were trained at weight 0. Verified
  against `criterion.weight` = `[1]*14+[0,0,0,1,0]`. The README states the dead
  classes must never win the argmax, which the notebook now asserts as a load
  check.

## Evidence

- `figures/flair-overview.png` — imagery, both GSDs, OSM control. The check that
  matters: do masks follow buildings and treelines, or the 512 px inference grid?
  Grid-aligned rectangles is what YOLOE produced and it is visible instantly.

## Cost

Measured 2026-09-23 on one location (3328x2560, 0.384 m/px), 6 CPU threads:

| mode | wall | peak RSS |
|---|---|---|
| native (0.386 m/px) | **25 s** | 1.8 GB |
| upsampled (to 0.2 m) | **89 s** | 5.6 GB |

So 9 locations x both modes is ~17 min; all 59 is ~1 h 50. Weights are 
cached by the Hub after the first pull. `upsampled` at 5.6 GB on a 16 GB machine
is the memory ceiling — one process at a time.

## If it graduates

Nothing moves yet — a pretrained model used as a label source belongs behind a
`uavloc/data/` interface shared with whatever wins, and that shape is not known
until all three probes report.
