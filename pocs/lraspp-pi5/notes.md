# lraspp-pi5

**Status:** exploring · **Question:** `How fast does the MobileNetV3-L LR-ASPP segmenter run on a Raspberry Pi 5 CPU, per runtime (onnxruntime vs torch) and thread count?`

## The question

`pocs/mobilenet-lraspp-speed` measured speed on an i7-9750H as a Pi proxy and
listed "not verified on target hardware" as a gap. This measures it on the Pi 5.

## What I ran

- `export.ipynb` (Mac, `uavloc` env): exports `model-lraspp-bn-{drone,satellite}.pt`
  to ONNX (opset 17, fixed shapes: drone frame 1×3×416×736, satellite tile
  1×3×512×512), saves a seeded reference input/output per model, copies the
  `.pt` files. Output in `models/` (gitignored weights).
- `bench.ipynb` (Pi, papermill): per runtime × model × threads, 5 warm-up + 30
  timed forward passes, then parity vs the Mac reference output.

Logs per run in `logs/<run-id>/`:

- `latency.jsonl` — one row per timed pass: runtime, model, threads, input_shape,
  iter, latency_ms, time_utc, temp_c, cpu_freq_mhz, throttled
- `parity.json` — max abs logit diff vs the Mac torch output, per config
- `system.json` — board, OS, package versions, governor, sensors at start/end, params
- `summary.csv` — median / p90 latency, max temp, min CPU freq per config

## Verdict

## Evidence gaps
