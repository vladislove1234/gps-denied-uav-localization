#!/bin/zsh
# 026.1 (must reproduce 025.1) first, then the two gated heads. One at a time, so ms_per_correction is clean.
# Stops at the first failure so run ids stay aligned. Usage: ./run_all.sh [026.1|026.2|026.3] ... (no argument: all)
set -e
cd "$(dirname "$0")"
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
mkdir -p artifacts
if (( $# )); then steps=($@); else steps=(026.1 026.2 026.3); fi   # zsh does not word-split ${@:-a b c}
for step in $steps; do
  case $step in
    026.1) run 026.1 -p NOTE "weighted NCC + accept-reject gate tuned as 025.1: must reproduce 025.1" -p MATCHER ncc ;;
    026.2) run 026.2 -p NOTE "023.3 head (seed 42) + the same gate: the idea" -p MATCHER head -r HEAD_RUN 023.3 ;;
    026.3) run 026.3 -p NOTE "023.5 head (seed 7) + the same gate" -p MATCHER head -r HEAD_RUN 023.5 ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "=== done $(date +%H:%M:%S)"
