#!/usr/bin/env bash
# 027: checks, tuning (4 shards in parallel), then the test runs. From this folder:
#   bash run_all.sh checks | tune | test <pf_T> <pf_q> <mht_T> <mht_q> <bar_filter>
#   bash run_all.sh tune-head <pf_q> <mht_q> | test-head <pf_T> <pf_q> <mht_T> <mht_q>
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
export OMP_NUM_THREADS=2
case "$1" in
  checks)
    "$PM" checks.ipynb artifacts/checks-run.ipynb -k python3 --cwd ../.. ;;
  tune)
    for shard in 0 1 2 3; do
      "$PM" tune.ipynb "artifacts/tune-shard-$shard.ipynb" -k python3 -p SHARD "$shard" -p N_SHARDS 4 &
    done
    wait ;;
  tune-head)
    for shard in 0 1; do
      "$PM" tune.ipynb "artifacts/tune-head-shard-$shard.ipynb" -k python3 -p SHARD "$shard" -p N_SHARDS 2 \
        -y "HEAD_RUN: '023.3'" -p TUNING_DIR artifacts/tuning-head -y "FIXED_Q: {particles: $2, hypotheses: $3}" &
    done
    wait ;;
  test-head)
    tuned="TUNED: {particles: {temperature: $2, q_pos_m_per_sqrt_s: $3}, hypotheses: {temperature: $4, q_pos_m_per_sqrt_s: $5}}"
    export OMP_NUM_THREADS=8
    "$PM" train.ipynb artifacts/run-head.ipynb -k python3 -p SCENARIO normal -p REGISTERED_KEY head -y "HEAD_RUN: '023.3'" \
      -p TUNING_DIR artifacts/tuning-head -y "ARM_LIST: [replace, particles, hypotheses]" -y "$tuned" \
      -p BAR_FILTER particles -p WAIT_FOR_RUNS 10 ;;
  test)
    tuned="TUNED: {particles: {temperature: $2, q_pos_m_per_sqrt_s: $3}, hypotheses: {temperature: $4, q_pos_m_per_sqrt_s: $5}}"
    export OMP_NUM_THREADS=4
    # normal logs 027.1-027.5; cold waits for those five, then logs 027.6-027.10
    "$PM" train.ipynb artifacts/run-normal.ipynb -k python3 -p SCENARIO normal -y "$tuned" -p BAR_FILTER "$6" &
    "$PM" train.ipynb artifacts/run-cold.ipynb -k python3 -p SCENARIO cold -p WAIT_FOR_RUNS 5 -y "$tuned" -p BAR_FILTER "$6" &
    wait ;;
esac
