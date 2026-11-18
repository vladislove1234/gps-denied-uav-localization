#!/usr/bin/env bash
# 029 from this folder: bash run_all.sh test  (029.1 prior x0.5, then 029.2 prior x2; 028's tuned T=0.5, q=0.3)
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
export OMP_NUM_THREADS=3
"$PM" train.ipynb artifacts/run-narrow.ipynb -k python3 -p PRIOR_SCALE 0.5 -y "REGISTERED_RUNS: {normal: '029.1', cold: null}" &
"$PM" train.ipynb artifacts/run-wide.ipynb -k python3 -p PRIOR_SCALE 2.0 -y "REGISTERED_RUNS: {normal: '029.2', cold: null}" -p WAIT_FOR_RUNS 1 &
wait
