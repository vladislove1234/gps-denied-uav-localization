#!/bin/bash
# Registered runs of 017-gray-logit-refiner, in proposal order. Run from src/.
set -e
PM="/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/papermill"
EXP=experiments/017-gray-logit-refiner
run() {  # $1 = output name, rest = papermill -p args
  local out=$1; shift
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PM $EXP/train.ipynb $EXP/artifacts/run-$out.ipynb -k python3 --cwd $EXP --log-level WARNING "$@"
}
run baseline -p MODE baseline -p NOTE "baseline: LR-ASPP bilinear argmax through this notebook"
run gray-logits-seed42 -p SEED 42 -p NOTE "primary: gray + logits, CE + Dice, seed 42"
run gray-logits-seed43 -p SEED 43 -p NOTE "seed swing: primary config, seed 43"
