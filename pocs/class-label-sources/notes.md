# PoC — label sources for the classes OSM cannot provide

Six classes were settled with the user on 2026-09-15:

    0 background   1 field   2 building   3 track   4 woody   5 water

`building` and `water` come from OSM and are measured good. The other three do
not: farmland is 0.25% median coverage, the flight ground's dirt track is
unmapped (0/207 frames), and only 136 `tree_row` ways exist across 499 cached
Overpass responses.

The user chose **plan A — per-class specialists** over a single land-cover
model, and accepted a model run over the satellite tile as a label source.

This folder answers one question per class. Each gets its own verdict; a `failed`
verdict is a finished result, not a reason to keep patching.

Run everything with `/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python`
from `src/`.

---

## Probe 1 — `field` from Delineate Anything

**Status: failed.** Recall looked good; the boundaries do not survive
inspection. See *Verdict* below — the numbers in this section are all still
correct, and that is the point worth keeping.

### The question

`pocs/drone-frame-boundaries/` marked DA **failed**. That verdict is about
labelling a *drone frame*: a 50×26 m footprint is 11% of a median 1.14 ha
parcel over Gonchary, so there is no parcel to find. A location tile here is
1284×988 m. The geometry that killed that PoC does not apply, so DA was re-asked
as a *satellite-tile* labeller.

### What was run

`field_da.py` over 9 of the 59 built locations — the 6 with the most OSM
farmland (a yardstick to score against) and 3 built-up ones (the failure mode
the handoff predicted). Tile resampled from 0.385 to 0.5 m/px, tiled inference
at 1024 with 128 overlap, `conf=0.15`, CPU, polygons scaled back to full
resolution. 41–81 s per location.

Reused from `pocs/map-derived-labels/boundaries.py` unchanged: tiled inference,
seam dedup by centroid, and the black-padding guard.

### Numbers

| location | OSM farm % | polys | median m² | DA cover % | recall on farmland | leak onto buildings |
|---|---|---|---|---|---|---|
| maidan | 51.9 | 275 | 2,177 | 58.0 | **77.6%** | 1.0% |
| ploske | 46.2 | 401 | 1,847 | 69.0 | 94.3% | 0.3% |
| kryve | 37.4 | 177 | 3,181 | 58.5 | 82.5% | 0.4% |
| kosteiv | 34.9 | 151 | 6,124 | 79.2 | 91.7% | 0.3% |
| ruda | 34.4 | 112 | 1,960 | 54.4 | **97.6%** | 0.1% |
| ripchytsi | 31.0 | 112 | 2,633 | 82.5 | 91.3% | 2.7% |
| lapaivka | 0.7 | 1,101 | 181 | 79.2 | 89.1% | 9.0% |
| birky | 0.0 | 1,060 | 140 | 50.3 | — | 11.7% |
| pidhaichyky | 0.0 | 490 | 706 | 35.5 | — | 4.6% |

**Rural (6 locations, 1,228 polygons):** median parcel 2,233 m², p10 627, p90
19,404. Recall on OSM farmland **78–98%, median 91%**. Leak onto buildings
median **0.4%**, max 2.7%.

**Built-up (3 locations, 2,651 polygons):** median parcel **196 m²**, leak onto
buildings median **9.0%**, max 11.7%.

### Verdict — failed

Recall on OSM farmland is 78-98%, median 91%, and that number is real. It is
also not the question. Zoomed to native 0.385 m/px in the inspector, the parcel
boundaries are wrong over many areas; the user rejected the source on sight.

**An aggregate agreement score hid per-parcel quality.** A filled DA mask can
cover the right ground while drawing the wrong shapes on it, and coverage
recall cannot see the difference. Any future probe here gets looked at
full-resolution before a verdict is written.

**The deeper reason, and the reusable part:** DA is an *instance* segmentation
model. The class set settled on 2026-09-15 makes `field` a filled surface class
with parcel merging explicitly accepted, so instances are not needed. This was
a semantic land-cover task handed to an instance model. Wrong tool class, not a
badly tuned one.

The two populations separate cleanly by parcel size, about 10× apart
(`figures/field-da-distributions.png`, left). That separation is what a filter
would key on, and it is measured rather than guessed.

### What is *not* established

- **The large DA-only area is unscored.** Red in the overview figure is ground
  DA called field and OSM never mapped. Visually it is field — at ploske,
  kosteiv and ripchytsi it is obviously so — but that is a **visual judgement,
  not a measurement**, and OSM cannot confirm it because sparsity is exactly
  why OSM was rejected for this class. Recall is measured; precision is not.
- **Built-up tiles need arbitration, not a verdict.** 196 m² median is a
  household garden plot. Whether that is `field` or `background` is a class
  definition question, unanswered. Untouched, it puts ~10% of DA's area on
  buildings.
- **`conf=0.15` and `TARGET_MPP=0.5` were inherited, not tuned.**

### DA claims buildings as field -- measured after the user asked

`leak_onto_buildings` above divides by DA's field area, which is large, so it
reads low even when every building is swallowed. Wrong denominator. Asked the
other way -- of OSM *building* pixels, how many fall inside a DA parcel:

| location | farm % | parcels touching a building | buildings claimed | old `leak` |
|---|---|---|---|---|
| maidan | 51.9 | 15.3% | 23.2% | 1.0% |
| ploske | 46.2 | 9.7% | 10.1% | 0.3% |
| kryve | 37.4 | 12.4% | 26.5% | 0.4% |
| kosteiv | 34.9 | 7.3% | 34.8% | 0.3% |
| ruda | 34.4 | 3.6% | 9.0% | 0.1% |
| ripchytsi | 31.0 | 21.4% | 77.2% | 2.7% |
| lapaivka | 0.7 | 59.0% | 84.9% | 9.0% |
| birky | 0.0 | 67.0% | 56.2% | 11.7% |
| pidhaichyky | 0.0 | 40.8% | 18.5% | 4.6% |

Two distinct behaviours: rural parcels swallow isolated farmsteads (median
building fraction per parcel is 0.0%, yet 77% of ripchytsi's buildings sit
inside a "field"), and built-up tiles get segmented into house plots outright
(50.8% of birky's parcels are >10% building).

Carries forward past this probe: **whatever supplies `field`, OSM `building`
must override it.** That is no longer an arbitration preference, it is required.

### A measurement that contradicts an earlier one

The drone-frame PoC measured a median parcel of **1.14 ha over Gonchary**. Over
these six village-adjacent locations the median is **0.22 ha** — 5× smaller.
Both are correct; they are different terrain (Gonchary is scrubby pasture, these
are strip fields beside villages). Noted because the footprint arithmetic in
that PoC is specific to Gonchary and should not be quoted as a general figure.

### Evidence

- `figures/field-da-overview.png` — imagery, DA outlines, DA-vs-OSM agreement
- `figures/field-da-distributions.png` — parcel size by terrain; recall and leak
- `results-field.json` — per-polygon statistics
- `data/derived/class-label-sources/field-polygons/*.npz` — the polygons
  (gitignored; `field_da.py` regenerates them)

---

## Probe 4 — all four classes from an open-vocabulary prompt (YOLOE)

**Status: failed.** Not "needs tuning" — measured against a null baseline, three
of four classes carry no positional information at all, and the fourth only
appears to because it paints half the tile.

### The question

The user's: skip training and just *name* the classes. `ultralytics 8.4.150`
ships `YOLOE`, a YOLO11 segmentation model with a CLIP text encoder, so
`set_classes(names, get_text_pe(names))` returns masks for concepts it was never
trained to segment. Also `YOLOWorld` (boxes only, so unusable here) and
`SAM`/`FastSAM` (geometry prompts, no semantics).

Worth running despite a bad prior because the cost of an answer was an
afternoon — no labelling, no training — and because `building` and `water` have
trusted OSM labels, which makes the probe **self-falsifying**: a model that
cannot find buildings OSM proves are there cannot be trusted on `track` or
`woody`. Probe 1 lacked exactly that property.

### What was run

`yoloe_probe.py` over the same 9 locations as probe 1. `yoloe-11l-seg.pt`,
tiled 1024 with 128 overlap at native 0.386 m/px, `conf=0.05`, CPU, seam dedup
by centroid (reused from `boundaries.predict`). The OSM control is `label.png`
shifted by the per-location `label_offset` the build recorded, so the dataset's
known 2.84 m label error is not charged to the detector.

Prompt phrasing was swept first and it matters more than anything else: bare
nouns (`"building"`, `"trees"`) return **zero** masks at any scale or model
size. The `"... in satellite image"` suffix is what makes the text encoder fire.
The reported runs all use that phrasing — i.e. the most favourable one found.

### The null baseline — the part worth keeping

Instance recall here is deliberately generous: an OSM blob counts as found if
*any* predicted pixel touches it. The first result read 0.57 recall on 355
buildings at maidan, which looks usable until you notice it came from 27 masks
covering **55% of the tile**.

So every score is paired with a **null**: the same prediction randomly
circular-shifted, twenty times, and rescored. That preserves blob count, shape
and total coverage exactly and destroys only *where* the blobs are. Anything
that does not beat its null is an artifact of how much of the tile got painted.

### Numbers — text prompts, 9 locations

| class | dets | tile covered | OSM actual | recall | null | IoU |
|---|---|---|---|---|---|---|
| building | 183 | 55.7% | 4.09% | 0.77 | 0.61 | 0.046 |
| track | 10 | 2.9% | 3.47% | 0.09 | **0.10** | 0.005 |
| woody | 38 | 13.5% | 6.57% | 0.27 | **0.26** | 0.090 |
| water | 71 | 21.1% | 0.76% | 0.25 | **0.33** | 0.002 |

`track`, `woody` and `water` are at or **below** their null. `building` is the
only class above it, and it gets there by claiming 55.7% of the ground when
buildings occupy 4.09% — IoU 0.046. Peak confidence anywhere was 0.45; most
detections sit near 0.2.

`track` produced **10 detections across all nine locations**. That is the class
the flight ground most needs.

### Numbers — visual prompts

Exemplar = the largest OSM building blob per location, passed as a bbox with
`YOLOEVPSegPredictor`. Only `building` was prompted, so the other three rows are
zero **by construction and prove nothing**.

| class | dets | tile covered | recall | null | IoU |
|---|---|---|---|---|---|
| building | 643 | 21.7% | 0.26 | 0.25 | 0.046 |

Recall equals its null. Worse, it is unstable: detections per location run
**70, 10, 0, 1, 60, 0, 403, 2, 97** for the same class and the same kind of
exemplar. Two locations return nothing; one returns 403 masks over 80% of the
tile. The mode I rated *above* text prompting for `track` beforehand is the one
that degenerated hardest.

### The most diagnostic single result

Prompt-free mode (`yoloe-11l-seg-pf.pt`) lets the model name what it sees
without any prompt. Over a Ukrainian village at 0.386 m/px it returned:

    fish (6), scenery (5), pollution (3), aircraft carrier (2), skylight (2),
    icicle (2), beam (2), cloud forest, plume, boat house, shipwreck, waiting hall

Not one correct label. The model is not mislabelling our classes — it does not
recognise the scene as land seen from above at all, and it parses it as
something aquatic. That is the whole result in one line, and it explains why
prompt wording dominated every other knob.

### Verdict — failed

The vocabulary these models learned comes from ground-level photographs (LVIS,
Objects365, GoldG). Nadir imagery at 0.386 m/px is outside it. A prompt can only
retrieve a concept the encoder already holds; it cannot teach one. Nothing here
is a tuning problem, and no amount of prompt engineering reaches a concept that
is absent.

Look at `figures/yoloe-overview.png` before re-opening this: the predicted masks
are **tile-aligned rectangles** spanning whole agricultural strips, labelled
`building` across fields and `water` over dry farmland. They follow the
inference grid, not the imagery.

This also closes the question permanently rather than leaving it as a maybe,
which was the stated reason to spend the afternoon.

### What is *not* established

- **Remote-sensing-pretrained open-vocabulary models were not tested.** The
  verdict is about YOLOE/YOLO-World checkpoints trained on natural images. It
  does not cover geospatial foundation models, which are a different prior.
- **Visual prompting was only asked for `building`.** Its collapse is real, but
  `track` and `woody` were never prompted visually.
- **`conf=0.05` is a floor, not a tuned value.** Raising it only removes
  detections, and the failure is that the detections are everywhere, so this
  cannot rescue the result — but it was not swept.

### Two silent failures recorded

- **`bug-018`: the kwarg is `visual_prompts`, not `prompts`.** Passing `prompts=`
  is not rejected — it is swallowed into overrides, VP never activates, and every
  image returns 0 detections. The first visual run produced a clean, plausible,
  entirely fake "0 dets everywhere" across two locations before a control on
  ultralytics' own `bus.jpg` (→ 1 mask, conf 0.931) exposed it. **Run the
  natural-image control before trusting any VP number.**
- **`bug-015`: upsampling a crop before `predict(imgsz=N)` does nothing.** The
  call resizes back to `imgsz`, so an x1/x2/x4 sweep varied nothing and returned
  byte-identical results. To change apparent object size, change the crop's
  ground extent.

### Probe 4b — the mask is the frame (the user's reading, tested)

The verdict above first shipped with an open caveat: maybe the classes failed
only because a 10 m building is 26 px in a 391 m frame — 2.6% of frame width —
and the model needed the object bigger. The evidence against that was four data
points from one crop at one location, which was too thin to carry it.

The user looked at `yoloe-overview.png` and read it differently and better: the
masks are *tile-sized rectangles*, so the model may simply be unable to draw a
small box regardless of what is in frame. That is a testable claim and a
different one.

`yoloe_smallcrop.py`: 9 locations x 8 sampled OSM buildings x 4 ground extents
(25/50/100/200 m), each crop centred on one known building, all resized to
`imgsz=640` so the only variable is how much ground is in frame. 288 crops.

| extent | largest mask / frame | building / frame | hit% | best IoU |
|---|---|---|---|---|
| 25 m | **0.997** | 0.184 | 100% | 0.193 |
| 50 m | **0.991** | 0.047 | 99% | 0.067 |
| 100 m | **0.985** | 0.012 | 100% | 0.046 |
| 200 m | **0.914** | 0.003 | 94% | 0.003 |

**The user was right about the largest mask, and then caught the next mistake
too.** The table above is `frame_frac` = the *largest* mask per crop, and the
figure originally drew only that mask. Asked why only the largest was shown,
re-ran recording **every** mask (`yoloe_allmasks.py`, 417 masks):

| extent | masks | median frac, all masks | under 30% of frame | best IoU |
|---|---|---|---|---|
| 25 m | 74 | 0.997 | 2.7% | 0.611 |
| 50 m | 95 | 0.984 | 20.0% | 0.628 |
| 100 m | 141 | 0.573 | 41.8% | 0.542 |
| 200 m | 107 | 0.770 | 31.8% | 0.019 |

**"YOLOE cannot draw a small box" was wrong and is retracted.** It can: 114 of
417 masks (27.3%) cover under 30% of the frame, the proportion rises with
context, and the single best mask anywhere reaches IoU 0.61. Plotting only the
largest mask manufactured the appearance of a model stuck at frame scale.

The correct statement is narrower and survives the full data: **the masks it
draws, small or large, are almost never on the target.** Median IoU of the small
masks is **0.000**, and only **7 of 417** masks (1.7%) are both small
(<30% of frame) and well-placed (IoU > 0.3). In the figure the small masks trace
roads and field edges at 100-200 m while the target building sits untouched.

So the degeneracy claim is downgraded: at 25 m the output really is frame-sized
(2.7% small), but by 100 m it is not, and the failure there is misplacement
rather than blanket coverage.

The `hit%` column near 100 is worthless and shows why the null baseline was
needed: a mask covering the whole frame overlaps everything in it. The proof is
the ratio **IoU / building_frac = 1.04, 1.03, 1.36, 0.98** across the four
extents — the mask scores exactly what a constant "select everything" mask
scores by construction. The detection carries no positional information at all.

**This still closes the scale hypothesis, for a reason that survives the
correction.** More ground context does make the masks smaller and more
structure-following — but not more *correct*: best IoU falls from 0.61 at 25 m
to 0.019 at 200 m, and mean best-IoU per crop falls monotonically
(0.193 / 0.067 / 0.046 / 0.003). Both ends fail, in opposite ways. There is no
extent at which this becomes a label source, so retiling cannot reach it.

`figures/yoloe-smallcrop.png` now draws **every** mask — red above 50% of frame,
orange below — after the first version showed only the largest and was replaced.
Green is the OSM building. At 25-50 m an isolated building fills the frame with
"building"; at 100-200 m small masks appear and follow roads and field edges
while the target building sits untouched, best IoU 0.00-0.01.

### Evidence

- `figures/yoloe-overview.png` — imagery / prediction / OSM control / native-res zoom
- `figures/yoloe-scores.png` — recall vs null, and coverage claimed vs actual
- `figures/yoloe-smallcrop.png` — every mask at four ground extents
- `results-yoloe-allmasks.json`, `yoloe_allmasks.py` — per-mask size and IoU,
  the data that retracted the "cannot draw small" claim
- `results-yoloe-text-satellite.json`, `results-yoloe-visual-satellite.json`,
  `results-yoloe-smallcrop.json`

---

## Probe 2 — `woody` from ESA WorldCover

**Status: exploring — code and numbers exist, no verdict written.**

This heading previously read "not started", which was stale: `worldcover.py`,
`woody_wc.py` and `export_woody.py` were written in an earlier session and
`results-woody.json` covers all 59 locations. Corrected on 2026-09-18 while
committing probe 4; **the verdict is still open and is not mine to write.**

What is on disk, stated as fact and not as a conclusion: WorldCover woody covers
a median 28.7% of a location, and recall on OSM woody is median 0.856 across the
35 locations that have any OSM woody at all (min 0.011, max 1.000). `woody_wc.py`
says outright that this recall is a *minimum bar, not evidence of correctness*,
because OSM woody is sparse-but-sharp — and that the probe "ends at a viewer
rather than at a number". The viewer export exists at
`data/derived/class-label-sources/viewer`. Nobody has looked at it yet.

That unfinished inspection is the next action for this class, and it is exactly
the step probe 1 skipped.

## Probe 3 — `track` from road extraction

**Status: not started.**
