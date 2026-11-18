# 033-class-loss-in-system — proposal

> Written **before** the first run. The success criterion below does not move
> once runs begin. If nine runs all miss it, the honest report says the
> approach failed.

## Question

When the onboard segmenter (MobileNetV3-L LR-ASPP, round-3 data, per-domain BN) is retrained with
the 021 class-specific loss terms, and the 023 correlation head is retrained on its outputs, does
the final system (dense particle filter + head, the `027.12` configuration) end closer to the truth
on the 94 test flights than the same pipeline retrained with plain CE + Dice?

## Hypothesis

The building terms (gap weight, per-building weight, Tversky on building) recover small buildings,
and the track terms (skeleton recall + Tversky on track) recover more of the track centreline. Both
classes carry the structure the matcher locks onto, so a map with more of them should give sharper,
better-placed heatmap peaks and a filter that drifts less between fixes.

The evidence so far points the other way, which is why this is worth one controlled test rather
than an assumption:
- In `021`, models trained on 8 satellite images and scored by single-fix weighted NCC (40 frames),
  the building loss moved position error by +0.24 / −0.20 m (CIs across 0), and the structure loss
  made it worse (+0.67 m @ 10 m on ResNet34, CI > 0), probably through extra false track
  (skeleton precision −0.12).
- `021` never tested the onboard segmenter, the learned head or the filter. The head learns which
  class evidence to trust, and the filter averages over fixes, so either could absorb the false
  track or make use of the recovered buildings.

## Success criterion

Unit of analysis: one **flight**. CIs are 95 % paired bootstraps over **villages** (both routes of a
village resampled together), 10,000 resamples, seed 42, as in `027`.

Arms (one segmenter + its own head + the filter each):

| Run | Segmenter loss | Role |
|---|---|---|
| `033.1` | `current` (CE + Dice), seed 0 | **control** |
| `033.2` | `building`, seed 0 | loss arm |
| `033.3` | `structure`, seed 0 | loss arm (the loss of novelty claim 1) |
| `033.4` | `current`, seed 1 | noise floor: retraining alone |

Loss weights are the `021` ones, unchanged (`config.loss_config` of `021.2` / `021.7`).

**Metric:** `mean_end_error_m` over the 94 test flights of `sim-flights`, normal start, filter A
(particles) with the arm's own head.
**Baseline:** `033.1` (control, same code and seed). `027.12` (10.027 m) is the reference only.
**Bar (each loss arm against `033.1`), all of:**
1. `mean_end_error_m` ≤ **0.9 ×** the control's, and the CI of the paired difference is clear of 0.
2. The improvement is larger than the retraining noise: `033.1 − arm` > `|033.1 − 033.4|`.
3. `divergence_share` no higher than the control's.
4. Guard, the 12 flights over villages the heads never trained on: `head_clean_mean_end_error_m`
   no higher than the control's.

**Precondition (does the loss still do its job in this training setup?).** Scored on the satellite
val sets of round 3 (birky-strip, birky-strip-google, bushcha), satellite BN, by the `021` metric
code:
- `033.2`: `building_missed_share_mean` ≤ control − 0.03.
- `033.3`: `building_missed_share_mean` ≤ control − 0.03 and `track_skeleton_recall_mean` ≥
  control + 0.03.

An arm that misses its precondition is reported as *"the loss did not reproduce its segmentation
gain on round-3 data"*. Its system number is still reported, but it is not read as an answer to the
question.

**Reported, not deciding:**
- Reproduction: `033.1` against `027.12` (10.027 m). A gap larger than `|033.1 − 033.4|` is flagged.
- Single-fix head results per arm on `village-crossdate-holdout` (200 frames) @ 10 m, from the head
  training notebook.
- Per arm: drone-val mIoU (drone BN) and satellite mIoU, separable merge rate, track skeleton
  precision; median end error, p90 and max error, share of skipped fixes, per-fix time.

## Requires

| Equipment | Status | Acceptance check |
|---|---|---|
| `uavloc.segmentation.village_seg`: the round-3 data, samplers, per-domain BN and eval helpers, moved from `pocs/village-seg-baseline/village_seg.py` (its third use; experiments may not import from a PoC) | to build | the segmenter notebook asserts the satellite and drone splits equal round 3's, as `probe.ipynb` does; `pocs/mobilenet-lraspp-speed/model-lraspp-bn-satellite.pt` scored through the moved code gives the same satellite-val mIoU as `results-lraspp.json` (±0.001) |
| `train_segmenter.ipynb` (this folder): `probe.ipynb`'s recipe with a `LOSS` parameter calling `structure_loss_terms` | to build | `checks.ipynb`: with `current` weights, `structure_loss_terms(...)["loss"]` equals `ce_dice_loss` on a real batch to 1e-6; train/val logging per `training-logging.md` |
| `uavloc.simulation.arms`: the filters, `HeadSpec`, `build_correlator` | exists (`027/arms.py` is already a shim; amended 2026-11-03, before any run) | `033`'s tune notebook with `027.12`'s segmenter and head reproduces `027.12`'s tuning error 6.600 m (±0.01) |
| Head training: `023-learned-correlation-head/train.ipynb` recipe, `ARM = "head"`, seed 42, `SEG_CHECKPOINT` = the arm's satellite-BN segmenter, head saved to this folder's `artifacts/` | adapt (copy into this folder) | the pair bank is rebuilt per segmenter (`bank_key` hashes `seg_checkpoint`; check that a new key appears) |
| `train.ipynb` (this folder): `027`'s run notebook for filter A + head, `HEAD_EXP` = this experiment, both correlator checkpoints = the arm's satellite-BN segmenter | adapt | ends with `log_run`; per arm, T is retuned on the tuning villages over `027`'s grid {0.25, 0.5, 1, 2} with q = 2.0 fixed, as for `027.12`, **before** any test run |
| `sim-flights`, `ua-village-multidate`, `village-crossdate-holdout` | exist | — |
| `uavloc.segmentation.losses`, `uavloc.segmentation.eval`, `uavloc.segmentation.instances` | exist (`021`) | — |

**What only you can provide.**
- [ ] A go-ahead on the bar, including the noise-floor arm `033.4`.
- [ ] Where the segmenters train: the RTX 3090 Ti server (`021/server.md`) or the Mac (MPS,
      ≈ 1 h each). All four on the same machine either way.

## Setup

| | |
|---|---|
| Dataset | segmenter: round-3 data (`pocs/village-seg-baseline`); head: `ua-village-multidate` 46 train / 10 val villages, as `023`; system: `sim-flights` (10 tuning, 47 test villages, 94 test flights) |
| Split | as round 3, `023` and `027`, unchanged |
| Seed | segmenter 0 (as `probe.ipynb`), 1 for `033.4`; head 42; bootstrap 42 |
| Model | LR-ASPP MobileNetV3-L, 3,000 steps, LR 5e-4, per-domain BN; head (32, 32, 16) + 16, 4,000 steps |
| Loss | segmenter: `current` / `building` / `structure` (`021` weights); head: as `023` |
| Hardware | segmenters: one machine for all four; head: MPS; system: CPU only (bug-121) |

Run order: builds and their checks → four segmenters → precondition table → four heads → T tuning
per arm on the tuning villages → four test runs, scored once.

## What would falsify this

- Both loss arms miss bar 1 while meeting their precondition: the segmentation gain does not carry
  over to position in the final system. The loss stays a segmentation result, and the thesis does
  not tie it to localization.
- `033.3` worse than the control with the CI clear of 0: the false-track mechanism from `021`
  survives the head and the filter.
- `|033.1 − 033.4|` larger than any loss effect: one seed per arm cannot resolve this question,
  and more seeds come before any claim.

## Threats

- One seed per loss arm. `033.4` measures retraining noise for the control only.
- The heads trained on 41 of the 47 test villages. The clean-subset guard covers this, on 12
  flights.
- Drone view in the simulator is a satellite image of another date, so the drone BN branch is not
  exercised. It is trained and its drone-val mIoU is reported, but the system result says nothing
  about real camera frames.

## Related

- `021-building-loss-from-scratch`: the loss terms and their segmentation effect; the single-fix
  null and harm this experiment re-tests in the final system.
- `pocs/mobilenet-lraspp-speed`: the onboard segmenter recipe retrained here.
- `023-learned-correlation-head`: head recipe; `026-gated-correlation-head`.
- `027-sequence-filters`: filter A, tuning procedure, `027.12` reference.
