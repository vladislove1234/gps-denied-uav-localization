#!/bin/zsh
# 015 registered runs, in order: control/head at seed 42, then seed 43. Run from src/.
set -e
PM=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/papermill
EXP=experiments/015-boundary-head
for SEED in 42 43; do
  $PM $EXP/train.ipynb $EXP/artifacts/run-control-seed$SEED.ipynb -k python3 --cwd $EXP \
    -p WITH_HEAD False -p SEED $SEED -p NOTE "control fine-tune, no head, seed $SEED"
  $PM $EXP/train.ipynb $EXP/artifacts/run-head-seed$SEED.ipynb -k python3 --cwd $EXP \
    -p WITH_HEAD True -p SEED $SEED -p COMPARE_CHECKPOINT artifacts/model-control-seed$SEED.pt \
    -p NOTE "multitask fine-tune with boundary head, seed $SEED"
done
