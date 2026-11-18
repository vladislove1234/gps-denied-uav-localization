# 001-seg-map-correction — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

On the 2025 benchmark frames at 15 m injected error, does correlating the
segmentation model's class maps give a lower mean position error than the 2025
VGG11 feature matching, with everything else held the same?

## Hypothesis

It will. The drone camera (grey-brown April grass) and the orthophoto (a
different season and sensor) disagree in appearance. They agree on *where the
track and the shrubs are*. VGG11 was trained contrastively on raw drone and
satellite pixel pairs, so its features still carry appearance. The segmentation
model's output is by construction a map of classes. Once each domain is
normalised with its own BatchNorm statistics, the class maps from both sides
should line up where the raw features do not. Round-2 drone-val mIoU of 0.786
(drone BN) and unseen-village satellite mIoU of 0.573 (satellite BN) say both
sides are segmented well enough for that.

## Success criterion

The metric that matters here is **metres of position error**, because the
system corrects an inertial prior rather than retrieving from a global map.

The 2025 results cannot serve as the baseline. They come from one unseeded draw
of noise direction per frame, on folders that have since lost 11 frames to
0-byte files. The best 2025 config at 15 m logged 8.21 m (17 frames,
`images_2_v3_renamed_small_easy`) and 15.08 m (20 frames,
`images_v6_renamed_easy`). So **run `001.1` re-runs VGG11 in this harness**,
with the 2025 config unchanged, and its number is the bar.

Equipment: `builds/position-correction-harness/` (verified 2026-09-27).

**Metric:** `mean_position_error_m` (@ 15 m injected; 44 frames × 8 directions)
**Baseline:** `001.1` → VGG11, 2025 config (`scale 1.7`, drone 448 px,
`sat_feature_crop 0.03`, no centre bias). No number until it runs.
**Bar:** a segmentation run with `mean_position_error_m` < `001.1`'s

*Decided with the user, 2026-09-27:* beating VGG11 at all counts. No margin
and no absolute ceiling are required.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.geo` — haversine, metric offset of a `LatLon`, `LatLon` ↔ pixel on a north-up image | built | Round-trip lat/lon → px → lat/lon closes to < 0.01 m on the orthophoto. Haversine matches the 2025 function to 1e-6 m. |
| `uavloc.matching.correlator` — port of `PositionCorrelator` that takes one feature function per domain | built | **Regression:** same frame, same noisy prior and VGG11 → matches the archived class to < 0.01 m on 5 frames. **Identity:** a satellite crop at the true pose, used as the "drone" image, is recovered to < 1 m at 8 yaws. This checks the rotation convention `drone.ipynb` flagged as possibly mirrored. |
| `uavloc.matching.benchmark` — seeded noise injection over N directions, returns one record per (frame, direction) | built | Injected offsets measure exactly 15.00 m by haversine. The same seed gives the same offsets. |
| `uavloc.features.vgg11` — 2025 `VGG11FeatureNet` + loader for `vgg11_drop_margin08_adamw_model_20250521_034900.pth` | built | Loads with `strict=True`. Output shape matches the archive's on one image. |
| `uavloc.features.segmentation` — U-Net/ResNet34 → softmax class maps at a given m/px | built | Drone-val mIoU with drone BN reproduces 0.786 ± 0.01 and bushcha with satellite BN 0.573 ± 0.01 (`results-round2-bncheck.json`). |
| Per-domain BN checkpoints of `model-round2.pt` → `model-round2-bn-{satellite,drone}.pt` | built | Same as the row above. |
| `uavloc.data.drone_frames` — `DroneFrame` + `load_frames(dataset)` with pose, mpp, `picked_from` | built | 44 frames load. Every pose equals its 2025 filename. |
| `uavloc.data.satellite.GeoImage` — north-up image + corner `LatLon`s, `mpp`, crop/resample | built | `mpp` ≈ 0.067 on the orthophoto. Uses `uavloc.geo` for all conversions. |
| `datasets/drone-2025-easy` — the 44 frames, the union of both 2025 folders, with 0-byte files recovered | exists | — |
| 2025 orthophoto `archive/uav_localization_2025/data/satelites/satellite_gonchary.jpg` (8192×6026, ~0.067 m/px) | exists | — |

**What only you can provide.** Nothing. All inputs are on disk.

> Building must not move the success criterion above. If the build reveals the
> metric is unmeasurable as promised, amend this file with a dated note saying
> what changed and why. Do not quietly swap it.

**Amended 2026-09-27, during the build.** The success criterion is unchanged.
Three acceptance checks were reworded, because as first written they could
not pass even for a correct port:

- **Regression:** "matches the archived class to < 0.01 m" became "heatmap
  r ≥ 0.9 on every draw". The correlation surfaces agree (r = 0.994 median),
  but the argmax flips between near-equal peaks under sub-pixel resampling
  differences. In 3 of 20 draws that moved the answer by 8–22 m. That is a
  property of the 2025 method, recorded in `builds/position-correction-harness/notes.md`.
- **Identity:** "< 1 m" became "< 1 m for segmentation, < one feature cell
  (~2.1 m) for VGG11". VGG11's conv5 stride of 16 px cannot resolve finer.
- **Haversine:** the 2025 function used R = 6,371 km while its planar maths
  implied 6,378 km. One sphere (6,378,137 m) is used throughout, so errors read
  0.11% larger than the 2025 function would report.

Also decided during the build: the orthophoto is segmented once, whole, at
0.385 m/px. Per-patch segmentation is not used.

## Setup

| | |
|---|---|
| Dataset | `drone-2025-easy`: 44 frames, the union of the two 2025 "easy" folders (decided with user 2026-09-27) |
| Satellite | 2025 orthophoto for both methods, 2025 corner coordinates. The segmentation model sees it downsampled to 0.385 m/px, its training scale. |
| Noise | 15 m primary. 5 and 10 m are also reported. 8 directions per frame, 45° apart with a seeded random phase. |
| Search window | 2025 geometry unchanged: patch = 1.7 × footprint, rotated by yaw |
| Pose | lat/lon/alt/yaw from the 2025 filenames, the same inputs the 2025 run used |
| Seed | 42 |
| Model A (`001.1`) | VGG11FeatureNet, `out_level c3`, 2025 weights |
| Model B (`001.2`) | `model-round2.pt`: drone BN on drone frames, satellite BN on the orthophoto. All 6 class probabilities, matched at 0.385 m/px. |
| Matching | 2025 correlator for both: channel-wise L2 normalisation, `conv2d`, z-score, argmax |
| Hardware | MPS (AMD Radeon Pro 5300M). Every run on the same device. |

**Diagnostics** (logged, never the criterion):
- median and p90 error
- share of draws that ended closer than the injected 15 m
- share of draws whose true position lies inside the search window
- error per subset: 2025 "train" folder, 2025 "val" folder, segmentation held-out blocks 0+4 (16 frames)

## Known confounds, accepted up front

- **The segmentation model trained on 22 of the 44 frames** (blocks 2+3), and
  6 more touch its training footprints. Its number over all 44 frames is
  optimistic. The 16-frame held-out subset is logged separately so the size of
  that optimism can be seen. VGG11 was likely trained on pairs from the same
  flight too, so it is not clean either.
- **The 2025 search window cannot recover every 15 m error.** At 51 m altitude
  the patch is 100 × 56 m against a 59 × 33 m footprint. That leaves about
  ±20 m of search along the frame's width and ±11.5 m along its height, and it
  limits both methods alike. The in-window share is logged so the reachable
  error is known.
- **VGG11's config was tuned on these frames in 2025. Model B's first config is
  not tuned.** Later seg iterations (e.g. dropping background, NCC) tune on the
  same frames. Report how many configs were tried.
- **Frames 1 s apart overlap about 90%.** The 44 frames are about 5 places, so
  the mean has fewer degrees of freedom than 352 draws suggest.

## What would falsify this

At its first, untuned config, class-map correlation lands at or above
`001.1`'s mean error. It is **abandoned** if it is still there after trying
the two obvious matching variants: background dropped, and zero-mean NCC
instead of per-pixel L2. That result would say the class maps carry less
localising signal than VGG11 features on this ground, even with the benefit of
having trained on these frames. The same holds if the model beats VGG11 only on
the frames it trained on and not on the held-out 16. That would be leakage,
not signal.

## Related

- 2025 benchmark: `archive/uav_localization_2025/image_matching/PositionCorrelatorTest.ipynb`
  (cells 29–30, 44) and `position_correlator.py`
- `pocs/village-seg-baseline/notes.md`: round-2 model, per-domain BN result
- No prior experiments in `experiments/`
