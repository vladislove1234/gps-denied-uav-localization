# perfect-map-matching

**Status:** ready for your review · **Unblocks:** `002-perfect-satellite-map`

> Status is one of: `planning`, `blocked`, `ready for your review`,
> `accepted`, `parked`. It is parsed from this line — keep the format.

## What this is

Two additions to the position-correction harness:
- A hand-labelled satellite map can stand in for the segmentation model's
  satellite side.
- A class-weighted per-class NCC matching mode sits next to the 2025
  per-pixel cosine.

## What I needed from you

- [x] Gonchary satellite labels exported (arrived as `ua-ortho-annotated/gonchary-2025`, drawn on the orthophoto) by the annotator → `data/derived/ua-village-annotated/samples.jsonl`, `location` starting `gonchary`, covering 49.761905, 24.167825 → 49.759486, 24.172788, drawn from satellite imagery only (see `experiments/002-perfect-satellite-map/proposal.md`)

## What I did

| Code | What |
|---|---|
| `uavloc/data/annotated.py` | `label_records`, `load_label_map(location, date)`: an exported label as a `GeoImage` on its Web Mercator box. Only reference-grid dates (`image_offset_px == [0, 0]`) are accepted, because other dates' labels follow their own shifted imagery. |
| `uavloc/features/segmentation.py` | `label_probabilities(labels, mpp)`: one-hot, area-averaged to the matching grid, with the ignore label mapped to no class. `SegmentationFeatures.from_satellite_probabilities(...)`. |
| `uavloc/matching/correlator.py` | `CorrelatorConfig.matching` (`cosine` \| `weighted-ncc`) and `class_weights`. `class_ncc`: zero-mean NCC per class, 0 where the satellite window's std < 0.05. `weighted_ncc_heatmap`: weighted mean over classes that are the drone argmax on ≥ 1% of pixels. A flat heatmap keeps the prior. |
| `uavloc/matching/identity.py` | `SatelliteAsDrone`, moved out of the harness verify notebook on its second use. |
| `uavloc/features/factory.py` | `build_features(config, satellite, device)` and `correlator_config(config)` from the logged run config. `001`/`002` `train.ipynb` use them, so what is logged is what ran. |

## Does it work

Run: `papermill builds/perfect-map-matching/verify.ipynb builds/perfect-map-matching/runs/verify.ipynb --cwd .` (~1 min).

- **`cosine` unchanged:** 12 sampled `001.3` draws replay with a max
  difference of 2.4e-10 m.
- **Label loader:** `birky-49p9105-23p9192 / current` loads as 3328 × 2560 at
  0.385 m/px, the same shape as its image. The overlay puts class borders on
  the imagery (`figures/label-map-overlay.png`).
- **Identity on a label map** (6 synthetic poses over birky, 8 directions,
  15 m): `cosine` and `weighted-ncc` are both 0.17 m median, 0.22 m max, on the
  22 reachable draws.
- **Identity on round 3's Gonchary map:** `cosine` 0.14 m median.
  `weighted-ncc` recovers 0.1–0.2 m on frames where the satellite map at the
  true spot has woody or building. On frames where it is pure background it
  keeps the prior (15.0 m). This is the rule as designed: with nothing but
  background there is nothing to match. It also previews `001.4`.
- `figures/cosine-vs-weighted-ncc.png`: both heatmaps on the worst reachable
  `001.3` draw.

- **Gonchary labels:** `load_label_map("gonchary-2025", "ortho", "ua-ortho-annotated")`
  has exactly the orthophoto's corners. Inside the required box: background
  94.9%, track 1.2%, woody 4.0%. `figures/gonchary-labels.png` shows the
  labels over the orthophoto with the frames. The track along the hedge line
  and the nearby shrubs are labelled. The tarmac road, the houses and most of
  the southern tree cover are not.

## Your call

Look at `figures/gonchary-labels.png`. If the unlabelled road, houses and
southern trees should be in the map, a relabel is a new run in 002 with a note.
