# landcoverai-finetune

**Status:** partial — `track` yes, everything else no · **Question:** `Can a model trained on LandCover.ai (rural Poland, 0.25 m, with a road class) supply the track class over Ukrainian Esri tiles?`

## The question

LandCover.ai is the closest annotated dataset that exists to this problem: rural
**Poland**, 0.25 m (33 orthophotos) and 0.5 m (8), 216.27 km², manually
fine-annotated, 10,674 tiles of 512x512 (7,470 / 1,602 / 1,602). Classes
**building, woodland, water, road** — four of our five, and the **only candidate
with a road class**, which nothing else supplies.

Class areas: woodland 72.02, water 13.15, road 3.50, building 1.85 km². Roads are
1.6% of labelled area — the class we most need is the rarest, so it is handled in
the loss rather than discovered afterwards.

**Licence CC-BY-NC-SA 4.0.** Non-commercial *and* share-alike; the share-alike
may attach to weights trained on it. Fine for offline label generation and
academic work, but it belongs in the dataset card.

### Gated on `flair-landcover-transfer`

Both ask the same underlying question — does a Western-European land-cover model
transfer to Ukrainian Esri tiles — and FLAIR answers it with no training.
**If FLAIR fails on `building` and `water` at both GSDs, stop here.**

### Compute reality, stated before starting

This machine is an **Intel i7-9750H, 6 cores, 16 GB, no CUDA, no MPS**.
Measured 2026-09-23 with ResNet34-UNet, 5 classes, 6 threads:

| crop | batch | s/batch | 2k subset | full 7,470 |
|---|---|---|---|---|
| 256 | 8 | 3.15 | 13 min/ep — **6 ep = 1.3 h** | 49 min/ep — 30 ep = 24.5 h |
| 512 | 4 | 6.00 | 50 min/ep — 6 ep = 5.0 h | 187 min/ep — 30 ep = **93 h** |

**The default probe config (256 px, 2,000 tiles, 6 epochs) is 1.3 hours** — an
afternoon, not the multi-day run a naive full-resolution setup would be. The
93 h figure is what makes 512 px over the full split infeasible here, and it is
why the probe is deliberately small. Stage 0b re-measures at run time so the
number is never stale. Training stays opt-in (`train=False`).

The probe model is deliberately not a contender for the paper's 85.6% mIoU —
256 px crops, a 2,000-tile subset, 6 epochs. The question is whether `road` is
learnable and whether it survives the border, not whether we match the benchmark.

## Verdict

**Two verdicts, as the question demanded.**

### 1. Did `road` learn on Polish data? Yes, weakly.

Validation IoU on LandCover.ai's own split, 6 epochs, 2,000-tile subset at 256 px:

| ep | background | building | track | woody | water |
|---|---|---|---|---|---|
| 0 | 0.337 | 0.032 | 0.086 | 0.662 | 0.359 |
| 1 | 0.481 | 0.040 | 0.101 | 0.827 | 0.439 |
| 2 | 0.666 | 0.270 | 0.121 | 0.856 | 0.567 |
| 3 | 0.638 | **0.334** | **0.179** | 0.835 | 0.531 |
| 4 | 0.709 | 0.157 | 0.175 | 0.856 | 0.618 |
| 5 | 0.713 | 0.247 | 0.165 | 0.840 | **0.692** |

`track` climbs monotonically to 0.179 then plateaus. The common classes learn
well (woody 0.84, water 0.69); the two rare ones — building 0.87% and track
1.52% of pixels — stay low and **building is unstable**, swinging 0.334 -> 0.157
-> 0.247 across the last three epochs under median-frequency weighting.

This is a probe model, not a contender for the paper's 85.56% mIoU: 27% of the
train split, 256 px instead of 512, 6 epochs instead of a converged run.

### 2. Did it survive the move to Ukraine? For `track`, qualitatively yes.

**`figures/landcoverai-transfer.png` is the result.** The yellow `track` masks
form a **connected road network that follows the actual lanes** — the
north-south lane at kosteiv, the diagonal main road plus side lanes at sosnivka,
the whole street grid at smykivtsi — and it includes **unpaved village lanes**,
not just tarmac.

That is categorically different from FLAIR, which returned 0.15% of the tile and
found only paved surfaces. **This is the first thing in the project that
produces a usable-looking `track` signal at all.**

But it **over-claims by ~3.2x**:

| | LandCover.ai | FLAIR | OSM road |
|---|---|---|---|
| median tile claimed as track | **9.19%** | 0.15% | 2.88% |

Some of that excess is legitimate — OSM does not map the field tracks the
project needs, so a correct source *must* exceed OSM. But the bands are visibly
wider than the lanes, and yellow bleeds into field edges. It is not calibrated.

### Everything else: no.

| class | IoU | null | recall | claims | OSM actual |
|---|---|---|---|---|---|
| building | 0.149 | 0.009 | 0.588 | 4.11% | 1.8% |
| woody | 0.001 | 0.000 | 0.802 | 24.99% | 0.03% |
| water | 0.002 | 0.002 | 0.164 | **19.53%** | 0.2% |

**`water` failed catastrophically** — 19.53% claimed against 0.2% actual. In the
figure whole ploughed field strips are painted blue. Polish water is 7.1% of the
training pixels and Ukrainian bare/ploughed soil evidently sits in that part of
the feature space.

**`building` is the worst of the three sources, 0 wins out of 9:**

| | LandCover.ai | FLAIR native | MS footprints |
|---|---|---|---|
| median IoU | 0.149 | 0.200 | **0.204** |
| per-location wins | **0/9** | 4/9 | 5/9 |

So LandCover.ai buys exactly one thing: `track`. It should not be used for
building, water or woody.

### What is *not* established

- **The evaluated checkpoint is not the best one.** Stage 1 saves every epoch
  and keeps the last, so transfer was measured on ep5 (building val 0.247)
  rather than ep3 (0.334). Building transfer is therefore measured pessimistically
  by an unknown margin. Defect, not a finding (`bug-032`).
- **`predict_ua` tiles with no overlap** (`step = crop`), unlike the FLAIR probe
  which used 64 px. Grid seams are visible in the native-resolution zooms
  (`bug-033`). The class fractions are affected by an unmeasured amount.
- **`track` has no quantitative control and still does not.** The 9.19% vs 2.88%
  comparison uses OSM road, whose widths are themselves guessed and known to be
  2-3x too wide. The figure is the evidence; a real number needs hand-labelled
  ground.
- **No cross-date consistency check was run on this model.** FLAIR failed on
  exactly that axis and it is the check that matters most for this dataset.
  Untested here.
- **6 epochs on 27% of the data.** The plateau in `track` may be the subset, not
  the method.

## What I ran

- `probe.ipynb`, three stages: **0** download/tile + compute budget, **1**
  fine-tune (opt-in), **2** transfer eval over Ukrainian tiles.

`track` has **no control** — OSM does not map the tracks we care about, which is
the whole reason for the class — so it is scored by cross-date consistency and by
eye at full resolution. A `track` number without one of those is not evidence.

## Evidence

- `figures/landcoverai-transfer.png` — imagery, prediction, native-resolution
  zoom. For `track` this figure *is* the evidence: are the predicted tracks on
  actual lanes, or on field edges and shadows?

## Cost

Download is **1.43 GB** (verified by range request 2026-09-23), extracting to
roughly 4-5 GB with tiles. **25 GB free**, so disk is not a constraint. Training
cost as measured above; stage 0b re-checks before anything commits.

## If it graduates

Nothing yet. Reported **two** verdicts — did `road` learn on Polish data, and did
it survive the move to Ukraine. Yes/no is the paved-road-bias result and would
close the `track`-from-a-model route.
