# 004-reposed-frames — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

If each frame's heading and footprint scale are corrected (the logged position
kept), does weighted class-map matching on the hand-labelled map beat VGG11 on
the 001/002 benchmark at 15 m injected error?

## Hypothesis

It will. `002.2` tied VGG11 (12.11 vs 11.63 m) and lost the tie on blocks 1–2,
where `pocs/pose-reregistration` found the logged heading off by +25–26°. A
linear feature matched at the wrong angle scores low everywhere, and the weighted
NCC keys almost entirely on the track, so it should suffer more from a wrong
heading than VGG11's texture features do. Fix the heading and scale, and the
method that depends on the track's geometry should gain more.

## Success criterion

**Metric:** `mean_position_error_m` (@ 15 m injected; 44 frames × 8 directions, seed 42, same draws as 001/002)
**Baseline:** `004.1` → VGG11 (2025 config, cosine) on the same reposed frames, run first
**Bar:** `004.2` reaches `mean_position_error_m` **<** `004.1`'s

The user asked to compare against VGG11 on the same pairs, so the baseline is
re-measured on the reposed frames rather than taken from `001.1`. Both methods
see the same corrected pose. How much each moved from its logged-pose run
(`001.1` 11.634 m, `002.2` 12.113 m) is reported as a diagnostic, not the
criterion.

### Pre-registered runs

| Run | Method | Matching | Frames |
|---|---|---|---|
| `004.1` | `vgg11` | `cosine` | `drone-2025-reposed` |
| `004.2` | `segmentation-labels` | `weighted-ncc` | `drone-2025-reposed` |

Everything else is identical to `001.1` / `002.2`: correlator geometry (scale
1.7, FOV 60°, 3% satellite-border crop), VGG11 checkpoint, round-3 drone
checkpoint, label map `ua-ortho-annotated/gonchary-2025` / `ortho`, weights
background 0, building 1, track 1, woody 0.5, field 0, water 0.5.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `drone-2025-reposed`: the 44 frames with `yaw_deg` = corrected heading and `alt_m` = logged × footprint scale, logged lat/lon | built (`datasets/drone-2025-reposed`) | 44 frames; lat/lon unchanged (0.0° difference); orthophoto and label crops at the reposed pose line up with the drone track (`preview.png`) |
| `GeoImage.crop_rotated(..., nearest=True)` for class-id crops | built | used by the dataset build only; default behaviour unchanged |
| `002`'s harness, checkpoints and label map | exists | — |

## Known confounds, accepted up front

- **Circular for `004.2`.** The heading and scale were fitted by maximising the
  *same* weighted NCC on the *same* hand map. `004.2` is therefore evaluated at
  the pose its own score likes best. VGG11 gets no such favour. A win here is an
  upper bound on what pose correction can give class-map matching, not evidence
  that it beats VGG11. That fair comparison is `003-pose-search`, which searches
  pose at test time.
- **Position truth is still the log.** The PoC's corrected positions are ambiguous
  along the track (median 36 m) and are not used. Any error in the logged
  position is shared by both methods.
- **Scale is weakly determined.** 7/44 frames sit at the smallest scale tried,
  0.5. Where the scale is wrong, both methods see the same wrong footprint.
- **The satellite search window shrinks with the scale** (it is 1.7 × the
  footprint). At median scale 0.6 the window is ~0.6× as wide, so more 15 m
  draws put the truth outside it. `reachable_share` is logged for both runs.
  This alone can raise the error of both methods versus 001/002.

## What would falsify this

`004.2` ≥ `004.1`. Then even with the pose tuned in its favour, class-map
matching does not beat VGG11 on these frames, and pose search (`003`) cannot be
expected to make it win either.

## Related

- `pocs/pose-reregistration`: where the corrections come from.
- `002-perfect-satellite-map`: `002.2`, the same method at the logged pose.
- `001-seg-map-correction`: `001.1`, VGG11 at the logged pose.
- `003-pose-search`: the non-circular version of this question.
