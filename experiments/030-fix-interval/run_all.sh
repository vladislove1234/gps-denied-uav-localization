#!/usr/bin/env bash
# 030 from this folder: bash run_all.sh  (030.1-030.4 in registered order; 028's tuned T=0.5, q=0.3)
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
export OMP_NUM_THREADS=2
i=0
for spec in "ins-error 10.0" "ins-error 20.0" "replace 10.0" "replace 20.0"; do
  set -- $spec; i=$((i + 1))
  "$PM" train.ipynb "artifacts/run-$1-$2.ipynb" -k python3 -p ARM "$1" -p INTERVAL_S "$2" -p WAIT_FOR_RUNS $((i - 1)) \
    -y "REGISTERED_RUNS: {normal: '030.$i', cold: null}" &
done
wait
