# pose-reregistration

**Status:** worked · **Question:** `Are the logged poses of the 2025 drone frames (and so the 2025 drone/satellite pairs) wrong, and by how much?`

## The question

The `002` walkthrough (step 2b) showed drone frames whose track runs at a
different angle from the satellite at the logged pose. This PoC measures it on
all 44 frames by re-registering each frame against the hand-labelled Gonchary
map.

## Verdict

**Yes. The logged heading is wrong on most frames, and the logged altitude
overstates the footprint.**
- Registered frames: 44/44.
- Heading correction: median |error| **10°**, half the frames beyond 10°.
- Footprint scale: median **0.6**.
- Match score: median **0.10** at the logged pose, **0.95** after
  re-registration.

The heading error is a **constant offset per pass** that changes between
passes:

| block | frames | median heading correction | median scale |
|---|---|---|---|
| 0 | 14 | −9° | 0.68 |
| 1 | 6 | **+26.5°** | 0.60 |
| 2 | 5 | **+25°** | 0.60 |
| 3 | 17 | +8° | 0.55 |
| 4 | 2 | −3° | 0.62 |

Blocks 1 and 2 are exactly where `002.2` lost to VGG11. The 2025 training
pairs were cut at the same logged poses (`002` walkthrough step 2b: our crop
equals the `image_pairs_v2` patch), so **VGG11 was trained on pairs rotated by
these amounts**.

**Position is not settled.** Median shift 12.9 m, but the peak slides a median
**36 m along the track**. One straight track fixes heading and across-track
offset, not the position along it. Do not use these positions as ground
truth.

## What I ran

`reregister.ipynb` (papermill, ~20 min on MPS for 44 frames):
`papermill pocs/pose-reregistration/reregister.ipynb pocs/pose-reregistration/runs/reregister.ipynb --cwd .`

For each frame:
1. The drone class map (round-3 model, drone BN) is resampled to the 0.385 m/px
   grid for each footprint scale 0.50–1.10, in steps of 0.05.
2. The hand-labelled map (`ua-ortho-annotated/gonchary-2025`) is cut around the
   logged position, turned to logged heading + offset, 30 m wider than the
   footprint on every side.
3. The score is the same weighted per-class NCC as `002.2`
   (`uavloc.matching.correlator.class_ncc_maps`, same weights), before
   z-scoring, so it is comparable across headings and scales.
4. The search is a coarse heading sweep ±45° in 3° steps at scale 1, then the
   scale at the best heading, then a fine heading sweep ±3° in 1° steps.
5. Ambiguity is the long-axis length of the region scoring ≥ 95% of the peak.

Outputs: `results.csv` (per frame), `results.json` (summary).

## Evidence

- `figures/example-sweeps.png`: one frame (block 2). The score against heading
  offset has a single clear peak; the score against scale is flat-ish and
  keeps rising toward small scales.
- `figures/example-before-after.png` and `figures/before-after-{1..4}.png`:
  every frame as drone frame | orthophoto at logged pose | orthophoto at
  re-registered pose | class overlay before | after. In the overlays the
  drone's track (yellow) crosses the map's (cyan) before and coincides with it
  (white) after.
- `figures/heading-corrections.png`: logged heading and the correction per
  frame over time, coloured by block.
- `figures/position-shifts.png`: logged → re-registered positions. Read with
  the ambiguity caveat.

## What this does not establish

- **The cause of the heading error.** It is not a single compass bias, since
  the sign changes between passes. It is not a simple timing lag either:
  block 1 held its heading within 2.5° yet needs +26°. Candidates are a
  camera/gimbal yaw that is not the airframe yaw, or magnetometer error that
  varies by pass. Nothing here separates them.
- **The scale.** The score varies little with scale, and 7/44 frames (low
  altitude, block 3) sit at the smallest scale tried, 0.5. The direction
  (logged altitude too high) agrees with the archive's −13 m calibration,
  which would hurt low altitudes most, but the value is weakly determined.
- **Circularity.** The corrections come from the same hand map `002` scores
  against. Using them to re-run `002` would grade the method on poses fitted to
  its own reference. They are fine as a diagnosis, not as benchmark truth.

## Cost

~20 min on MPS for 44 frames; about 25 s per frame, dominated by
large-kernel NCC on the CPU.

## If it graduates

This does not graduate as-is. What it argues for:
1. **A rotation (and scale) search inside the correlator.** That is
   legitimate at test time and would help both methods. It is the next
   experiment.
2. **Rebuilding the training pairs** with per-pass heading corrections
   estimated on frames that are *not* used for evaluation.
