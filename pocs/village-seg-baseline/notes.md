# village-seg-baseline

**Status:** worked (satellite) · drone transfer failed · **Question:** `Can a U-Net/ResNet-34 learn our classes from one hand-corrected village well enough to beat the reference source it was seeded from, on ground it did not train on?`

## The question

The first `ua-village-annotated` export is one location (birky) on four dates.
Before annotating more, check that the labels are learnable and pick the
initialisation to carry forward: the LandCover.ai probe from
`pocs/landcoverai-finetune` (aerial, rural Poland) vs ImageNet.

## Verdict

**Yes on satellite imagery; no on drone frames.** On the held-out strip
(birky, all 4 dates) both initialisations reach mIoU ~0.47 against 0.240 for the
LandCover.ai probe zero-shot, and predictions agree across dates on 0.81-0.82 of
pixels (labels: 0.875; zero-shot: 0.49). **The LandCover.ai start buys nothing**
-- ImageNet matches or beats it at every eval, so carry ImageNet forward. Both
curves are flat after step ~400-600: the limit is one village of data, not
training length.

| | zero-shot | landcoverai | imagenet |
|---|---|---|---|
| background | 0.252 | 0.795 | 0.796 |
| building | 0.356 | 0.600 | 0.600 |
| track | 0.456* | 0.525 | 0.527 |
| woody | 0.133 | 0.391 | 0.438 |
| water | 0.003 | 0.000 | 0.000 |
| **mIoU** | **0.240** | **0.462** | **0.472** |
| cross-date agreement | 0.488 | 0.818 | 0.812 |

\* inflated: `track` labels were seeded from this model. Water in the strip is
~7k px — not measurable. mIoU curve (steps 0,200..1200): landcoverai
0.223/0.412/0.449/0.455/0.465/0.463/0.462; imagenet
0.023/0.444/0.458/0.476/0.486/0.482/0.472.

Buildings come out as blobs rather than rectangles (73% recall, IoU capped ~0.6);
woody is the weakest class, and its labels vary 4%-10% between dates.

## What I ran

`probe.ipynb` — one notebook, one run per `init`:

- **Hold-out:** the top quarter of the image (rows 0:640), held out on **all
  four dates**, 64 px gap to any train crop. Chosen by rule from the 8 quarter
  strips (no class >35% held out; maximise the worst-served class). It holds
  25% building, 23% track, 19% woody, 8% water — water is only ~7k px there.
- **Model:** `smp.Unet("resnet34")`, 6 classes. `landcoverai` init copies the
  probe's weights and head rows for the 5 shared classes; `field` random.
- **Training:** 512 px crops at native 0.385 m/px, batch 4, 1,200 steps, AdamW
  3e-4 one-cycle, CE (ignore 255) + soft Dice. 30% of crops centred on a pixel of
  a random present class (water is 0.2%). rot90/flip + brightness/tint/contrast.
  MPS, 2.46 s/step measured.
- **Baselines:** the LandCover.ai probe zero-shot, run as the annotator adapter
  runs it (0.25 m, 256 px); label agreement across dates as the ceiling for
  prediction agreement.
- The **last** step is reported, not the best — the strip is the only held-out
  data.

Re-run from `src/`:
`PYTORCH_ENABLE_MPS_FALLBACK=1 papermill pocs/village-seg-baseline/probe.ipynb pocs/village-seg-baseline/runs/probe-<init>.ipynb -p init <init>`

## Evidence

- `figures/<init>-curve.png` — loss and per-class hold-out IoU over steps
- `figures/<init>-holdout.png` — whole strip, every date: image/label/prediction/errors
- `figures/<init>-zoom.png` — densest-building window, native resolution
- `results-<init>.json` — every number above, plus confusion and per-date IoU

**Not established by construction:** anything about other villages. One
location means the score is fit to birky. `field` has no labels and is not
scored. The zero-shot `track` IoU is inflated — `track` was seeded from that
model.

## Drone frames (`drone.ipynb`)

**The satellite-trained model does not transfer to the 2025 drone frames.**
8 frames over the Gonchary pasture, chosen where the satellite-side prediction
puts a track in the footprint; every one visibly shows a dirt track.

- **From the satellite, in-domain, it works:** the model finds the unmapped dirt
  tracks over the flight area at 0.386 m/px (`figures/drone-landcoverai-overview.png`).
- **From the drone, at every GSD tried (0.385 / 0.19 / 0.096 m/px), it does not.**
  Median predicted track share is 0.0 / 0.0 / 0.1%; track IoU against the
  satellite-side prediction is 0.00 on all 8 frames. Instead it calls large
  patches of pale dry grass **building** (up to ~80% of a frame).
- Likely cause is appearance, not scale: the frames are grey-brown dead grass
  (early April), the Esri imagery is green summer, and no training date looks
  like the drone camera. Scale alone does not rescue it — finer GSDs only break
  the false buildings into smaller blobs.
- Registration caveat: in frames 1 and 3 the track sits on the opposite side of
  the satellite crop from the drone frame, so the pose-derived crop may be
  mirrored/rotated or offset for some headings. It does not change the verdict
  (no track is predicted on the drone side at all), but the drone-vs-satellite
  agreement numbers are not trustworthy until that is checked.

The ImageNet model fails the same way: median track share 0.0 / 0.03 / 0.01%
across the three GSDs, track IoU vs satellite 0.00 on 7 of 8 frames (0.04 on
one), fewer false buildings than landcoverai but still up to 46% of a frame.
Its frame set is chosen from its own satellite prediction, so it may differ
slightly from landcoverai's.

Evidence: `figures/drone-<init>-frames.png`, `results-drone-<init>.json`.

## Round 2 (`round2.ipynb`) — 3 villages + 22 drone frames, from `model-imagenet.pt`

**The weights learned both domains; one shared set of BatchNorm statistics
cannot serve both.** Training alternated all-satellite and all-drone batches, so
BN's running statistics at eval time reflect whichever domain ran last. The
logged curve swung accordingly (bushcha 0.453 -> 0.532 -> 0.427 -> 0.500 ->
0.309 -> 0.337 -> 0.453) while satellite train loss fell steadily. Re-estimating
BN statistics per domain on the final weights, no retraining
(`runs/round2-bncheck.ipynb`, `results-round2-bncheck.json`):

| BN statistics from | birky-strip | bushcha (unseen) | drone-val |
|---|---|---|---|
| round-1 model | 0.476 | 0.453 | 0.318 |
| round 2, as trained | 0.409 | 0.453 | 0.666 |
| round 2, **satellite** | **0.473** | **0.573** | 0.454 |
| round 2, **drone** | 0.196 | 0.126 | **0.786** |
| round 2, mixed | 0.400 | 0.443 | 0.648 |

- **Unseen village (bushcha): 0.453 -> 0.573** with satellite BN. Woody 0.82,
  track 0.45, building 0.43, water 0.43 (water measurable here: 125k px).
- **Drone: 0.318 -> 0.786** with drone BN; track IoU 0.07 -> 0.83, woody 0.56.
  In `figures/round2-drone-val.png` (as-trained weights) the track is found cleanly
  on every block-0 frame; bright frosted shrubs are still called **building** — no
  drone label has a building, so nothing in the drone data corrects it. Block 4
  (2 frames, a different place) is still poor.
- birky-strip is flat (0.476 -> 0.473): no new birky labels, so none expected.
- Mixed statistics do not help: the domains need separate normalisation.
- **Domain is always known at inference** (the drone camera vs the satellite map),
  so domain-specific BN is legitimate here, not a trick.

Caveats: drone-val is effectively two places (block 0 = one pass, block 4 = 2
frames); BN was re-estimated from *training* images of each domain, never
held-out ones; the logged per-step curve is not interpretable as learning
progress for the reason above. `figures/round2-*.png` use the as-trained BN.

Training 52 min on MPS (1,500 steps). BN check ~5 min.

Visual example (`bn-example.ipynb`, 2026-09-30): the saved round-2 checkpoints
(`model-round2.pt` as trained vs `model-round2-bn-{satellite,drone}.pt`) on the
same bushcha dates and drone-val frames, with a fixed/broken map (own BN vs shared).
Set mIoU: bushcha 0.453 -> 0.588, drone-val 0.666 -> 0.783; own-domain BN beats
shared BN on every one of the 18 images. These `-bn-` checkpoints come from round 3's
`domain_bn`, not the bncheck recalibration, hence 0.588 vs 0.573.
Figures: `figures/round2-bn-example-{drone,bushcha}.png`.

## Round 3 (`round3.ipynb`) — + Google imagery for birky and dubrova

Round-2 recipe unchanged (init `model-imagenet.pt`, 1,500 steps, seed 0); the
only new data is `label-google.png` for birky and dubrova. Every eval uses
per-domain BN (satellite BN for satellite sets, drone BN for drone-val), and the
"before" is `model-round2-bn-{satellite,drone}.pt`. Helpers moved out of the
round-2 notebook into `village_seg.py`; the drone split is asserted identical.

| set | round 2 | round 3 |
|---|---|---|
| birky-strip, Google date (held out) | 0.257 | **0.429** |
| birky-strip, 4 Esri dates | 0.474 | 0.471 |
| bushcha (unseen) | 0.588 | 0.588 |
| drone-val | 0.783 | 0.779 |

- Google strip: woody 0.01 -> 0.42, track 0.17 -> 0.43, building 0.40 -> 0.53.
  The image is low-sun with long shadows; round 2 misses the street grid.
- Nothing else regressed.
- 6 unseen villages with `google.jpg` (evenly spaced from the sorted list, no
  labels): round 3 changes little — a few fewer false buildings / track
  fragments. Google vs Esri predictions of the same village still differ
  (woody share by 0.1-0.2 in several).

Caveats: the Google score is one 640-row strip; it swung 0.30-0.43 across evals.
No labelled Google image of an unseen village exists yet.

Evidence: `figures/round3-{curve,birky-strip-google,google-unseen}.png`,
`results-round3.json`. Training 79 min on MPS.

## Cost

Training 62 min (landcoverai) and 67 min (imagenet) on MPS, ~3 s/step with evals. Zero-shot baseline ~30 s. Drone notebook ~2 min.

## If it graduates

**Code moved (2026-11-03):** `village_seg.py` now lives in `uavloc/segmentation/village_seg.py` (its third use, experiment 033); the file here is a shim re-exporting the same names.
