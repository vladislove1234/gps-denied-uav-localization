# seg-correction-pi5

**Status:** worked · **Question:** `On a Raspberry Pi 5 CPU, how fast are the U-Net/ResNet18 and U-Net/ResNet34 segmenters next to LR-ASPP, and how long does one full correction of a 10 m position error take with each?`

## The question

`pocs/lraspp-pi5` measured only LR-ASPP inference and the segmentation pipeline.
It did not measure the ResNet U-Nets or the correction itself (matching and the
decision step). This PoC measures both, using the three 021 segmenters
(`current` loss, seed 42, 2,500 steps).

## What I ran

- `export.ipynb` (Mac, `uavloc` env). Exports each 021 checkpoint to ONNX
  (opset 17, batch 1, dynamic H×W) and copies its state dict. Saves seeded
  reference I/O at the lraspp-pi5 bench shapes (drone 1×3×416×736, satellite
  1×3×512×512). Picks one correction case from the 021 records (gated rule,
  10 m injected): the draw that all three models accept with the smallest
  worst-case error, which is `birky-08` at bearing 140.8°. The satellite is
  segmented before flight on the Mac, as it would be onboard. Each model's
  probability map is cut to a 200 m window around the prior. The case is then
  replayed on the Mac CPU as the reference result. Outputs go to `models/` and
  `case/`.
- `bench.ipynb` (Pi). For every runtime (onnxruntime, torch) × architecture ×
  input × thread count (1–4): 5 warm-up passes and 30 timed passes, with the
  temperature, frequency and throttle state logged on every pass, plus a parity
  check against the Mac logits. Same protocol as `lraspp-pi5/bench.ipynb`.
- `correction.ipynb` (Pi). For each model: 5 warm-ups, then 30 corrections of
  the case, each starting from a cold drone cache. The correction runs through
  the real `uavloc` `PositionCorrelator`, with an onnxruntime session as the
  segmenter. Each correction is timed by stage: drone map (read, resize, ONNX,
  softmax), match (satellite patch and weighted NCC) and decide. The notebook
  also records the error before and after, and whether the corrected position
  matches the Mac's.

The Pi imports the `uavloc` package that is rsynced next to the notebooks. It
needs `segmentation-models-pytorch==0.3.4` (the Mac version) and
`opencv-python-headless`.

The correction frame is the 021 test frame: a 120×68 px cross-date crop at
0.385 m/px, not a 416×736 camera frame. Its inference is therefore much smaller
than the bench's drone input.

Mac dry run (i7, 4 threads, not the Pi):

| | inference | drone map | match | decide | total | error 10 m → |
|---|---|---|---|---|---|---|
| unet-resnet18 | 13.7 ms | 17.8 | 29.5 | 1.2 | 45.8 | 0.27 m |
| unet-resnet34 | 17.2 | 21.8 | 41.0 | 0.7 | 63.4 | 0.27 m |
| lraspp-mobilenetv3 | 3.7 | 6.3 | 20.1 | 0.9 | 36.5 | 0.17 m |

## Run on the Pi

```bash
# Mac, from src/pocs/seg-correction-pi5
PI=pi@raspberrypi.local
rsync -av --exclude runs --exclude export.ipynb ./ $PI:~/seg-correction-pi5/
rsync -av --exclude __pycache__ ../../uavloc $PI:~/seg-correction-pi5/

# Pi, in the env used for lraspp-pi5
cd ~/seg-correction-pi5 && mkdir -p runs
python -m pip install "segmentation-models-pytorch==0.3.4" opencv-python-headless
PYTHONPATH=$PWD python -m papermill correction.ipynb runs/correction-run.ipynb -p RUN_ID pi5-correction
PYTHONPATH=$PWD python -m papermill bench.ipynb runs/bench-run.ipynb -p RUN_ID pi5-bench

# Mac
rsync -av $PI:~/seg-correction-pi5/runs/ runs/
```

Logs go to `runs/logs/<RUN_ID>/`. The bench writes `latency.jsonl`,
`parity.json`, `system.json` and `summary.csv`. The correction writes
`timings.csv`, `run.json`, `stages.png` and `heatmaps.png`.

## Verdict

**The ResNet U-Nets run on the Pi 5, but they are 4–5.6× slower than LR-ASPP.
One correction takes 0.15–0.30 s with any of the three models, and 84–90% of
that is matching, not the network.** All three fix the case's 10 m error to
0.17–0.27 m. The Pi's result is the same as the Mac's, to 0.0 m. The cheapest
speed-up is in the matching code: each class takes part as one ~120 ms NCC.

## Results (Pi 5, runs/logs/pi5-bench, pi5-correction)

Inference, median ms (30 passes, FP32, no throttling, max 57 °C):

| | ORT 1 thread | ORT 4 threads | torch best | ORT 4 threads vs LR-ASPP |
|---|---|---|---|---|
| LR-ASPP drone 416×736 | 314 | **166** | 367 (1 thread) | 1× |
| U-Net R18 drone | 1893 | **669** | 943 (2 threads) | 4.0× |
| U-Net R34 drone | 2627 | **933** | 1221 (2 threads) | 5.6× |
| LR-ASPP sat 512² | 274 | **147** | 374 (2 threads) | 1× |
| U-Net R18 sat | 1695 | **602** | 869 (3 threads) | 4.1× |
| U-Net R34 sat | 2318 | **758** | 1106 (3 threads) | 5.2× |

- onnxruntime with 4 threads is the fastest setup for every model. For the
  U-Nets it is 1.3–1.5× faster than the best torch time; for LR-ASPP, 2.2–2.5×.
  torch gains nothing from more threads, as in lraspp-pi5. At 1 thread torch
  beats onnxruntime on the U-Nets (1002 vs 1893 ms).
- ORT scales 2.8× from 1 to 4 threads on the U-Nets, and 1.9× on LR-ASPP.
- LR-ASPP matches lraspp-pi5 to within 6% (166 vs 157 ms drone, 147 vs 142 ms
  satellite). This run used the `ondemand` governor, not `performance`; the CPU
  frequency stayed at 2400 MHz throughout.
- Parity against the Mac is ≤ 1.4e-5 for every runtime and model.

One correction of the case (birky-08, 10 m injected, gated), median of 30 runs
from a cold drone cache, ORT with 4 threads, in ms:

| | ONNX inference | drone map | match | decide | **total** (p90) | error 10 m → |
|---|---|---|---|---|---|---|
| unet-resnet18 | 31.7 | 34.9 | 134.4 | 0.8 | **172** (188) | 0.27 m |
| unet-resnet34 | 44.9 | 49.2 | 250.5 | 0.8 | **301** (312) | 0.27 m |
| lraspp-mobilenetv3 | 7.1 | 11.4 | 136.7 | 0.8 | **153** (165) | 0.17 m |

- Matching is 84–90% of a correction. Its cost grows with the number of
  classes that take part: those with weight > 0 that cover ≥ 1% of the drone
  frame. ResNet18 and LR-ASPP match on `building` only. ResNet34 also finds
  `track` on 1.3% of the frame, adds a second NCC and doubles the match time.
  The cost is about 120 ms per class on a 115×204 satellite patch against a
  67×120 drone map. Each `class_ncc` does 3 direct `conv2d`s with a 67×120
  kernel. Box sums (an integral image) and an FFT correlation should cut this
  sharply. This has not been measured yet.
- So on a crop-sized frame, the gap between models comes from the class count
  more than from the network. Inference alone is 7 / 32 / 45 ms.

## Evidence gaps

- Inference is FP32 only. INT8 and other quantised variants were not tried.
- The correction is timed on one case, on a crop-sized frame.
- The satellite segmentation is treated as pre-flight and is not in the
  correction time. The bench's satellite 512² row gives its cost per tile.
