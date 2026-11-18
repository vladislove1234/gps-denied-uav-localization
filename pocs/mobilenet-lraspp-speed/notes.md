# mobilenet-lraspp-speed

**Status:** failed (as registered: accuracy missed; CPU speed met) · **Question:** `Does MobileNetV3-Large + LR-ASPP, trained on the round-3 data, reach round-3 U-Net drone-val mIoU within 0.03 at ≥3× lower latency per drone frame?`

## The question

The drone side is the part that would run onboard. Round 3 (U-Net/ResNet34,
24.4 M params) reaches drone-val mIoU 0.779 with drone BN. This asks whether a
model built for speed keeps that accuracy. Threshold fixed before the run:
drone-val mIoU ≥ 0.749 and drone-frame latency ≤ 1/3 of the U-Net's, on the
same device. Satellite sets are reported but are not part of the question.

## Verdict

**No, as registered — narrowly on accuracy; speed passes on CPU only.**
Drone-val mIoU is **0.733** vs round 3's **0.779** (gap 0.046, threshold 0.03).
A drone frame runs **3.2–4.1× faster on CPU** (the Raspberry-Pi-class proxy) but
**no faster on MPS at batch 1**. The miss is all in `woody` (0.549 → 0.387);
`track`, the class the matching needs, is *better* (0.819 → 0.846). The gap is
inside this set's eval-to-eval swing (0.695–0.788 across the last 10 evals), so
it is not evidence that LR-ASPP is worse on drone frames — only that it did not
prove it is as good.

| @ final step, per-domain BN | U-Net r3 loss | LR-ASPP loss | U-Net r3 mIoU | LR-ASPP mIoU |
|---|---|---|---|---|
| sat-train | 0.630 | 0.591 | 0.698 | 0.707 |
| drone-train | 0.107 | 0.102 | 0.806 | 0.875 |
| birky-strip (val) | 0.966 | 0.997 | 0.471 | 0.460 |
| birky-strip-google (val) | 1.046 | 1.010 | 0.429 | 0.451 |
| bushcha (val, unseen) | 0.792 | 0.885 | 0.588 | 0.516 |
| drone-val | 0.265 | 0.305 | 0.779 | 0.733 |

- **Drone side overfits after ~750 steps.** drone-val loss is lowest at step 750
  (0.234, mIoU 0.788) and then settles near 0.30 while drone-train mIoU keeps
  climbing to 0.875. The last step is reported, not the best, as in round 3.
- **Satellite side is a wash except bushcha.** The Google strip is better
  (woody 0.42 → 0.51), bushcha worse (water 0.48 → 0.21, building 0.46 → 0.40).
  Satellite curves are flat from ~1,750 steps.
- **Speed** (standalone re-run, 3 trials, median of 30 forward passes):

| | U-Net r3 | LR-ASPP | speedup |
|---|---|---|---|
| params / GMAC per drone frame | 24.4 M / 36.9 | 3.2 M / 2.4 | 15× fewer MACs |
| drone frame 416×736, CPU | 529–708 ms | 135–221 ms | **3.2–4.1×** |
| drone frame, MPS | 54–72 ms | 49–73 ms | 0.75–1.2× |
| sat batch 8×512, CPU / MPS | 3.2–3.5 s / 195–213 ms | 1.0–1.6 s / 77–89 ms | 2.2–3.4× / 2.4–2.5× |

  MPS at batch 1 is kernel-launch bound: both models sit at ~50–70 ms whatever
  their MACs. The in-notebook table (`results-lraspp.json`) shows the U-Net at
  170 ms on MPS — an outlier from running straight after training; the
  standalone re-run above is the one to trust.

## What I ran

`probe.ipynb`, run as `runs/probe-lraspp.ipynb`:

- **Data, split, sampler, augmentation, loss, per-domain BN:** identical to
  `pocs/village-seg-baseline/round3.ipynb`. The notebook asserts the satellite
  and drone splits equal round 3's. Helpers are imported from
  `pocs/village-seg-baseline/village_seg.py`.
- **Model:** torchvision `lraspp_mobilenet_v3_large`, ImageNet backbone, new
  6-class head, wrapped to return logits.
- **Deviations from round 3, because the start point differs:** round 3
  fine-tuned a U-Net that had already trained on birky (round 1, 1,200 steps),
  so this model gets **3,000 steps** (≈ round 1 + round 3) and **LR 5e-4**
  (round 3: 2e-4, round 1: 3e-4) for its new head.
- **Logged every 250 steps**, all with per-domain BN: whole-image loss (CE + Dice)
  and mIoU/per-class IoU on two train sets (`sat-train`: every trained-on
  satellite image with the held-out strip ignored; `drone-train`: the 22 train
  frames) and four val sets (birky-strip, birky-strip-google, bushcha,
  drone-val). The running loss of the augmented training batches is logged
  per domain. The round-3 U-Net is scored by the same code once, as the reference.
- **Speed:** median forward pass of 30 after 5 warm-up (in the notebook, then
  re-run standalone ×3 because the in-notebook MPS numbers were unstable), drone frame at 0.10 m/px
  (1 × 3 × 416 × 736) and satellite batch 8 × 512 px, on MPS (Radeon Pro 5300M)
  and CPU (i7-9750H). MACs by `thop`.

Re-run from `src/`:
`PYTORCH_ENABLE_MPS_FALLBACK=1 papermill pocs/mobilenet-lraspp-speed/probe.ipynb pocs/mobilenet-lraspp-speed/runs/probe-lraspp.ipynb -k python3`

## Evidence

- `history-lraspp.csv` — every logged step: set, split, domain, loss, mIoU, per-class IoU
- `results-lraspp.json` — params, splits, reference and final scores, full history, speed table
- `figures/lraspp-curves.png` — loss and mIoU, train vs val, per domain; round-3 U-Net dashed
- `figures/lraspp-speed.png` — latency per device
- `figures/lraspp-drone-val.png`, `figures/lraspp-birky-strip-google.png` — U-Net vs LR-ASPP predictions
- `model-lraspp{,-bn-satellite,-bn-drone}.pt` — weights (gitignored)

## Evidence gaps

- drone-val is effectively two places (block 0, block 4 = 2 frames); a single
  eval moves ±0.04, so a 0.046 gap is not resolvable with one seed.
- CPU is an i7-9750H, not a Raspberry Pi; ARM speedups for hardswish/SE blocks
  may differ. Not verified on target hardware.
- One seed, one LR (5e-4, untuned), 3,000 steps vs the U-Net's 1,200 + 1,500.
- Not run through the position-correction harness: no metres-of-error number.
- `probe.ipynb` imports `pocs/village-seg-baseline/village_seg.py`, which the
  PoC contract forbids (cross-PoC import). Third use of those helpers — they
  have earned a place in `uavloc/` if this line continues.

## Cost

Training 59 min on MPS for 3,000 steps (~0.5 s/step training + ~3.5 min per
eval; per-domain BN recalibration is ~75 s of each eval). Reference U-Net
scoring 1 min. Speed benchmark ~5 min.

## If it graduates
