# 005-village-crossdate — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

With exact ground truth, does class-map matching with the MobileNet LR-ASPP
segmentation model beat VGG11 in the same correlator test at 15 m injected
error? The "drone" frames are the newest satellite date and the map is an older
date of the same val ground.

## Hypothesis

It will. In 001–004 every method was scored against the logged 2025 pose, which
disagrees with the map by several metres (`004-reposed-frames` gallery). The
frames also mostly showed a single straight track. Here the truth is exact by
construction: all dates are co-registered to 1.5 cm median (`ua-village-multidate`).
Village ground has buildings, roads and tree lines in several directions, so a
class map should pin position in 2D. VGG11 was fine-tuned on Gonchary
drone/satellite pairs only and has no training on this ground or on stale-date
changes.

## Success criterion

**Metric:** `mean_position_error_m` (@ 15 m injected; 40 frames × 8 directions, seed 42)
**Baseline:** `005.1` → VGG11 (2025 config, cosine) on the same frames, run first
**Bar:** `005.2` reaches `mean_position_error_m` **<** `005.1`'s

### Pre-registered runs

| Run | Method | Model | Matching |
|---|---|---|---|
| `005.1` | `vgg11` | 2025 VGG11 checkpoint | `cosine` |
| `005.2` | `segmentation` | MobileNetV3 LR-ASPP, satellite BN, **both sides** | `weighted-ncc` |

Both sides of `005.2` are satellite imagery, so both use
`model-lraspp-bn-satellite.pt` at the tile's native ~0.38 m/px
(`drone_seg_gsd_m = match_gsd_m = 0.385`). Weights are unchanged from `002.2`:
background 0, building 1, track 1, woody 0.5, field 0, water 0.5. Correlator
geometry is unchanged from 001–004: scale 1.7, FOV 60°, 3% satellite-border
crop, 5/10/15 m injected, 8 directions.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `village-crossdate-frames`: 40 "drone" frames, the newest Esri date cropped at random poses (40 m altitude, 46 × 26 m footprint, random heading), plus the oldest Esri date of the same village as the map | built | 20 on bushcha (map 2024-06-06), 20 on the birky strip (map 2018-03-28); every search patch stays inside rows 0–704 of birky; `preview.png` |
| `uavloc.features.segmentation.load_segmenter(..., "lraspp")` + `seg_architecture` config key | built | the LR-ASPP checkpoint loads strictly; default `unet` keeps every earlier run's config unchanged |
| Val ground: bushcha (whole village) and the birky strip (rows 0–640 + 64 gap) | exists | the same held-out ground as `pocs/village-seg-baseline` round 3 and `pocs/mobilenet-lraspp-speed` |

## Known confounds, accepted up front

- **Not drone imagery.** The "drone" view is a 0.38 m/px satellite crop, 120 × 68 px.
  It has no perspective, no tilt, and none of the drone's 3–4 cm detail. The test is
  matching across time, not across sensors.
- **VGG11 is out of its training domain** (drone vs satellite, one field). That
  is also true of any real deployment on new ground.
- **Two places, very different date gaps.** Bushcha's gap is 2 years. Birky's is
  7 years, a March map against current summer imagery. Per-place means are logged.
- **The birky strip is narrow** (~270 m). Its 20 frames overlap and are not
  independent.
- Random positions include featureless ground. Nothing was hand-picked.

## What would falsify this

`005.2` ≥ `005.1`. Then even with exact truth and structured ground, MobileNet
class-map matching does not beat VGG11. The 2025 data would not be the whole
explanation for 001–004.

## Related

- `004-reposed-frames`: gallery showing the logged 2025 truth off the map.
- `pocs/mobilenet-lraspp-speed`: the model and its val sets.
- `002-perfect-satellite-map`: weighted NCC and its weights.
