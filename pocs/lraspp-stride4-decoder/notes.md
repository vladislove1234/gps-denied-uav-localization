# lraspp-stride4-decoder

**Status:** failed · **Question:** `Does one stride-4 decoder step on LR-ASPP cut building merges, at an acceptable CPU cost?`

## The question

A perfect stride-4 prediction merges 14.6 % of val buildings against 25.7 % at stride 8
(`pocs/building-merge-stride`). Does adding one decoder step at stride 4 to LR-ASPP, fine-tuned from the current
checkpoint, lower the building merge rate (and raise Boundary IoU) against a plain fine-tune with the same steps,
without costing more than ~20 % extra CPU time?

## Verdict

**No, not on its own.** After the same 400-step fine-tune, the stride-4 model equals the control: mIoU 0.4688 vs
0.4687, Boundary IoU 0.1226 vs 0.1220, merged buildings 28.6 % vs 28.5 % (mean over sets), stride-8-separable merge
rate 0.212 vs 0.211; neither beats the untouched original (0.4758 mIoU, 27.6 % merged). The decoder did learn a
correction (`correction.ipynb`: median class-relative 1.1-1.3 against a 2.5-3.5 logit margin, flips 1-2 % of
pixels), but it is a near-uniform building bias plus texture, not a signal at the gaps between buildings. So stride
4 gives the *capacity* to separate them (the oracle says so) and CE + Dice does not *ask* for it: gap pixels are a
tiny share of the loss. The decoder only makes sense together with a loss that targets gaps (019). Latency cost is
negligible (135 vs 133 ms, same-session, 30 runs).

## What I ran

`decoder.ipynb`, one notebook, three arms on the same val sets:
- `original`: the current checkpoint, no training (it must reproduce the harness predictions exactly).
- `control`: the same checkpoint fine-tuned 400 steps with CE + Dice (016's recipe), so the fine-tune effect is
  not credited to the decoder.
- `stride4`: the same fine-tune with `Stride4LRASPP`: the stride-8 logits are upsampled x2, concatenated with the
  backbone's layer-3 features (stride 4, 24 ch), passed through 3x3 conv (32) + BN + ReLU and a zero-initialised
  1x1 correction. It starts identical to the base model. Decoder LR 1e-3, pretrained weights 1e-4.

All arms predict through one `probabilities(model, rgb, stride)` function (tiled like `lraspp_outputs`) and are
timed with it on the standard 512 tile. Building statuses use the merge PoC's definitions (copied, not imported).

## Evidence

- `figures/loss-seed42.png`: train and val CE / Dice / loss and val mIoU per 25 steps; stride4 tracks control.
- `figures/merges-seed42.png`, `figures/metrics-seed42.png`: no change vs control.
- `figures/correction-seed42.png`: the learned correction, near-uniform over buildings and fields.
- `results-seed42.json`, `history-seed42.csv`: every logged number. `runs/`: executed notebooks and the log.

One seed, 400 steps, one decoder width (32), decoder LR 1e-3. A longer fine-tune was not tried.

## Cost

~25 min for the full run on MPS (9.5 min training per arm, ~1 s/step, plus val passes and CPU eval).
