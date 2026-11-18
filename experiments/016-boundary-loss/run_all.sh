#!/bin/sh
# Registered sweep of 016-boundary-loss, run from src/: sh experiments/016-boundary-loss/run_all.sh SEED LAMBDA...
set -e
EXP=experiments/016-boundary-loss
PM=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/papermill
SEED=$1; shift
for LAMBDA in "$@"; do
  PYTORCH_ENABLE_MPS_FALLBACK=1 $PM $EXP/train.ipynb $EXP/artifacts/run-lambda$LAMBDA-seed$SEED.ipynb -k python3 --cwd $EXP \
    -p LAMBDA_BOUNDARY $LAMBDA -p SEED $SEED -p NOTE "boundary CE lambda=$LAMBDA seed=$SEED"
done
