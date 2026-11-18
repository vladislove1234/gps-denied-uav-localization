#!/usr/bin/env bash
# 028 from this folder: bash run_all.sh checks | tune | test <T> <q>
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
export OMP_NUM_THREADS=3
case "$1" in
  checks) "$PM" checks.ipynb artifacts/checks-run.ipynb -k python3 ;;
  tune) "$PM" tune.ipynb artifacts/tune-run.ipynb -k python3 ;;
  test)
    tuned="TUNED: {temperature: $2, q_pos_m_per_sqrt_s: $3}"
    "$PM" train.ipynb artifacts/run-normal.ipynb -k python3 -p SCENARIO normal -y "$tuned" &
    "$PM" train.ipynb artifacts/run-cold.ipynb -k python3 -p SCENARIO cold -p WAIT_FOR_RUNS 1 -y "$tuned" &
    wait ;;
esac
