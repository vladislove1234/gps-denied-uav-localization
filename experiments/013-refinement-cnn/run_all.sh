#!/bin/bash
# Registered runs of 013-refinement-cnn, in proposal order. Run from src/.
set -e
PM="/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/papermill"
EXP=experiments/013-refinement-cnn
run() {  # $1 = output name, rest = papermill -p args
  local out=$1; shift
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PM $EXP/train.ipynb $EXP/artifacts/run-$out.ipynb -k python3 --cwd $EXP --log-level WARNING "$@"
}
run baseline -p MODE baseline -p NOTE "baseline: LR-ASPP bilinear argmax through this notebook"
run edges0-seed42 -p USE_EDGES False -p SEED 42 -p NOTE "primary: RGB + log-probs, seed 42"
run edges1-seed42 -p USE_EDGES True -p SEED 42 -p NOTE "ablation: + Sobel edge magnitude, seed 42"
run edges0-seed43 -p USE_EDGES False -p SEED 43 -p NOTE "seed swing: primary config, seed 43"
run edges1-seed43 -p USE_EDGES True -p SEED 43 -p NOTE "seed swing: edge config, seed 43"
