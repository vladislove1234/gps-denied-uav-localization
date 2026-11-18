#!/usr/bin/env bash
# 031 from this folder: tune T (head likelihood, q = 0.3), then 031.1 once 028.1's flights CSV exists.
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
PY="$HOME/miniconda3/envs/uavloc/bin/python"
export OMP_NUM_THREADS=2
"$PM" tune.ipynb artifacts/tune-run.ipynb -k python3
T=$("$PY" -c "import pandas as pd; t = pd.read_csv('artifacts/tuning/shard-0.csv'); print(t.sort_values('mean_end_error_m').iloc[0]['temperature'])")
echo "tuned T=$T"
until [ -f ../028-ins-error-state/artifacts/flights-028.1.csv ]; do sleep 60; done
"$PM" train.ipynb artifacts/run-normal.ipynb -k python3 -p SCENARIO normal -y "TUNED: {temperature: $T, q_pos_m_per_sqrt_s: 0.3}"
