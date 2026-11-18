#!/bin/sh
# Registered runs of 021-building-loss-from-scratch. Run from src/, on the machine that trains all of them:
#   sh experiments/021-building-loss-from-scratch/run_all.sh SEED [ARCH...]
# Trains the current, building and structure loss per architecture from ImageNet init. Default ARCH order is the
# registered one; pass one ARCH per tmux window to run architectures in parallel.
# KERNEL (default uavloc) and PAPERMILL (default papermill) override the Jupyter kernel and the papermill binary.
set -e
EXP=experiments/021-building-loss-from-scratch
PM=${PAPERMILL:-papermill}
KERNEL=${KERNEL:-uavloc}
mkdir -p $EXP/artifacts
SEED=$1; shift
ARCHS=${*:-"unet-resnet18 unet-resnet34 lraspp-mobilenetv3"}
for ARCH in $ARCHS; do
  for LOSS in current building structure; do
    $PM $EXP/train.ipynb $EXP/artifacts/run-$ARCH-$LOSS-seed$SEED.ipynb -k $KERNEL --cwd $EXP --log-output \
      -p ARCHITECTURE $ARCH -p LOSS $LOSS -p SEED $SEED -p NOTE "$ARCH, $LOSS loss, seed $SEED"
  done
done
