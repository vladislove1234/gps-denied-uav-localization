#!/bin/sh
# Registered runs of 019-building-gap-loss, run from src/: sh experiments/019-building-gap-loss/run_all.sh SEED W0...
# 019.1 (reference, no fine-tune): pass SEED=42 W0=ref.
set -e
EXP=experiments/019-building-gap-loss
PM=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/papermill
SEED=$1; shift
for W0 in "$@"; do
  if [ "$W0" = ref ]; then
    PYTORCH_ENABLE_MPS_FALLBACK=1 $PM $EXP/train.ipynb $EXP/artifacts/run-reference.ipynb -k python3 --cwd $EXP \
      -p STEPS 0 -p RECALIBRATE_BN False -p SEED $SEED -p NOTE "original checkpoint, no fine-tune (reference)"
  else
    PYTORCH_ENABLE_MPS_FALLBACK=1 $PM $EXP/train.ipynb $EXP/artifacts/run-gap$W0-seed$SEED.ipynb -k python3 --cwd $EXP \
      -p GAP_WEIGHT $W0 -p SEED $SEED -p NOTE "gap-weighted CE w0=$W0 seed=$SEED"
  fi
done
