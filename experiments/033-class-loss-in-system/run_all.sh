#!/bin/zsh
# 033, per-arm pipeline after the arm's segmenter exists: head (MPS) -> T tuning (CPU shards) -> test run.
#   run_all.sh <arm> <registered_run> [segmenter-checkpoint]
#   arm: control | building | structure | noise      registered_run: 033.1 | 033.2 | 033.3 | 033.4
#   segmenter-checkpoint (relative to src/): default experiments/033-class-loss-in-system/artifacts/segmenter-<arm>/model-bn-satellite.pt
# Env knobs: N_SHARDS (default 2), TUNE_THREADS (default 1), TEST_THREADS (default 2).
#   CPU budget on the 8-core M1 with the four arms run side by side: 4 arms x N_SHARDS 2 x 1 thread = 8 threads while
#   tuning, then 4 arms x 1 process x 2 threads = 8 threads for the test runs. More shards or threads oversubscribe the
#   cores, and each CPU process holds about 1 GB (maps + segmenter + head), so 16 shard processes would also press 16 GB.
# The test run waits until runs.jsonl holds the previous registered runs, so ids follow the proposal whatever order the arms finish in.
# Executed notebooks and logs go to artifacts/runs/ (artifacts/ is gitignored, never committed).
set -eu
ARM=$1
REGISTERED_RUN=$2
SEG_CHECKPOINT=${3:-experiments/033-class-loss-in-system/artifacts/segmenter-$ARM/model-bn-satellite.pt}
N_SHARDS=${N_SHARDS:-2}
TUNE_THREADS=${TUNE_THREADS:-1}
TEST_THREADS=${TEST_THREADS:-2}

cd "$(dirname "$0")"
SRC=$(cd ../.. && pwd)
if [[ ! -f $SRC/$SEG_CHECKPOINT ]]; then
  echo "segmenter checkpoint not found: $SRC/$SEG_CHECKPOINT" >&2
  exit 1
fi
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
OUT=artifacts/runs
mkdir -p "$OUT"
run() {  # run <notebook> <output stem> <papermill args...>
  local notebook=$1 stem=$2; shift 2
  "$PM" "$notebook" "$OUT/$stem.ipynb" --cwd . -k python3 --log-level WARNING "$@" > "$OUT/$stem.log" 2>&1
}

echo "=== head $ARM $(date +%H:%M:%S)"
PYTORCH_ENABLE_MPS_FALLBACK=1 run train_head.ipynb "head-$ARM" -p ARM "$ARM" -p SEG_CHECKPOINT "$SEG_CHECKPOINT"

echo "=== tune $ARM $(date +%H:%M:%S)"
export OMP_NUM_THREADS=$TUNE_THREADS MKL_NUM_THREADS=$TUNE_THREADS
pids=()
for ((shard = 0; shard < N_SHARDS; shard++)); do
  run tune.ipynb "tune-$ARM-shard-$shard" -p ARM "$ARM" -p SEG_CHECKPOINT "$SEG_CHECKPOINT" -p SHARD "$shard" -p N_SHARDS "$N_SHARDS" &
  pids+=($!)
done
for pid in $pids; do wait $pid; done

echo "=== test $ARM $REGISTERED_RUN $(date +%H:%M:%S)"
export OMP_NUM_THREADS=$TEST_THREADS MKL_NUM_THREADS=$TEST_THREADS
run train.ipynb "run-$REGISTERED_RUN" -p ARM "$ARM" -p SEG_CHECKPOINT "$SEG_CHECKPOINT" -y "REGISTERED_RUN: '$REGISTERED_RUN'"   # -p would read 033.1 as the float 33.1
echo "=== done $ARM $(date +%H:%M:%S)"
