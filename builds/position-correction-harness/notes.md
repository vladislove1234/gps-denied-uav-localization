# position-correction-harness

**Status:** ready for your review · **Unblocks:** `001-seg-map-correction`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

The 2025 position-correction benchmark rebuilt as equipment. It injects a known
error, crops a heading-up satellite patch around the wrong position, correlates
it with the drone frame and measures the metres left over. Any feature
extractor can be plugged in: the 2025 VGG11, the segmentation model's class
maps, or anything later.

## What I needed from you

Nothing. Every input was already on disk.

## What I did

| Module | What |
|---|---|
| `uavloc/geo.py` | `LatLon`, haversine, metric offsets, camera footprint, heading-up ↔ north-up rotation. One sphere: WGS84 a = 6,378,137 m, which is the sphere the 2025 per-degree constants imply. |
| `uavloc/data/satellite.py` | `GeoImage`: a north-up raster with corner lat/lons, `resampled(mpp)` and `crop_rotated(center, yaw, w_m, h_m)`. Handles non-square pixels: the orthophoto is 0.0665 × 0.0679 m/px, and the 2025 code averaged the two. |
| `uavloc/data/drone_frames.py` | `DroneFrame` and `load_frames(dataset)` from `data/derived/<dataset>/*/meta.json`. |
| `uavloc/data/seg_crops.py` | Round 2's crop sampler and photometric jitter, moved here on its second use. |
| `uavloc/features/imagenet.py` | ImageNet normalisation constants. |
| `uavloc/features/tiling.py` | `tiled_logits`, moved from `tools/annotator/refs/_infer.py`, which now re-exports it. |
| `uavloc/features/vgg11.py` | The 2025 `VGG11FeatureNet` (conv1 to conv5_1, same state-dict keys), plus `Vgg11Features`: both inputs are squashed to squares, as in 2025. |
| `uavloc/features/segmentation.py` | `load_unet`, `recalibrate_batchnorm`, `class_probabilities`, and `SegmentationFeatures`. The drone frame is segmented at 0.10 m/px with drone BN. The satellite is segmented **once, whole** at 0.385 m/px with satellite BN. Both are compared on one 0.385 m/px grid. |
| `uavloc/matching/correlator.py` | `PositionCorrelator`: the 2025 geometry and correlation, with feature extraction injected. It also reports whether the truth was inside the search window. |
| `uavloc/matching/benchmark.py` | `run_benchmark` (seeded, N evenly spaced directions per frame) and `summarise`. |

**BN checkpoints**, gitignored: `pocs/village-seg-baseline/model-round2-bn-satellite.pt`
and `model-round2-bn-drone.pt`. Each is `model-round2.pt` with its BatchNorm
statistics re-averaged over 60 batches drawn with round 2's own sampler, seed 42.

**Two deliberate departures from the 2025 code:**
- **Border-crop offset fixed.** The 2025 correlator took the argmax on the
  border-cropped heatmap but converted it with the full map's size. With
  `sat_feature_crop_pct = 0.03` that biases every answer by one feature cell,
  about 2 m toward the patch's top-left. The port adds the cropped cells back.
- **Non-square pixels respected**, as described in the `GeoImage` row above.

## Does it work

Run: `papermill builds/position-correction-harness/verify.ipynb builds/position-correction-harness/runs/verify.ipynb --cwd .`
(about 3 min on MPS). Numbers are from that run's outputs.

- **Geo:** lat/lon → px → lat/lon closes to 0.0 m on every frame. A 15 m
  injection measures 15.000 m. The 2025 haversine (R = 6,371 km) would call it
  14.983 m, so errors here read 0.11% larger than 2025 errors for the same
  answer.
- **Regression against the archived 2025 class** (5 frames × 4 directions,
  `sat_feature_crop_pct = 0`, with the 2025 averaged pixel size forced): heatmap
  correlation r = 0.994 median (min 0.909). The corrected position differs by
  0.27 m median. But in 3 of 20 draws the port lands on a different peak:
  8.4, 12.0 and 22.3 m away. The heatmaps agree, and the argmax still flips
  between near-equal peaks when resampling moves by a fraction of a pixel. That
  is a finding about the 2025 method rather than a porting bug: **its answer is
  not stable under sub-pixel perturbation of the input.**
  `figures/regression-vs-2025.png`.
- **Identity:** a satellite crop at the true pose was used as the "drone" input
  (5 frames × 8 directions, 15 m). When the truth is inside the window:
  segmentation 0.14 m median (max 0.27), VGG11 1.14 m median (max 2.03). VGG11
  stays within one of its ~2.1 m feature cells. **Only 16 of 40 draws had the
  truth inside the window**: at 15 m the 2025 window leaves ±9.9 m along the
  frame's height.
- **Segmentation reproduces round 2:** drone-val mIoU 0.783 (round 2: 0.786),
  bushcha 0.588 (round 2: 0.573). Bushcha is 0.015 **above** round 2, just
  outside the ±0.01 tolerance but on the favourable side. My first,
  simplified sampler (plain crops, no jitter) got drone-val 0.717. BN statistics
  need the training crop distribution, which is why the sampler moved into
  `uavloc/data/seg_crops.py`.
- `figures/pose-drone-vs-satellite.png`: drone frame, satellite crop at the
  logged pose, and both class maps for 6 frames. **Yaw convention:** in the
  three frames with a clear track (rows 1–3) the faint path in the satellite
  crop runs where the drone sees the track, so the 2025 rotation holds. Rows
  4–6 are ambiguous.
- `figures/satellite-class-map.png`: **the satellite class map misses the
  pasture dirt track the frames are centred on.** It finds the tarmac road,
  the village buildings and the trees, but the faint summer path through the
  field comes out as background. The drone side segments the same track
  cleanly. So on this orthophoto the class maps disagree on the one structure
  in view. The round-2 notes found this track on the Esri z18 imagery; the
  2025 orthophoto is a different source.

Not verified: that the logged altitude is the height above ground. The archive
calibration applied a −13 m offset that the filenames may not include. Both
methods take the same altitude, so the comparison is fair, but the footprint
scale may be off for both.

## Your call

Look at `figures/satellite-class-map.png`. The pre-registered first run uses
this orthophoto, as you chose, and I'm running it as is. If segmentation
loses, the most likely reason is visible there: the satellite side does not
see the track.
