#!/bin/zsh
# Registered runs of 023. `run_all.sh gate`: acceptance checks + 023.1 (must reproduce 007.1), then stop and look.
# `run_all.sh trained`: 023.2-023.5. Stops at the first failure so run ids stay aligned.
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
if [[ $1 == gate ]]; then
  echo "=== checks $(date +%H:%M:%S)"
  $PY -m papermill checks.ipynb artifacts/checks-run.ipynb --cwd . -k python3 --log-level WARNING
  run 023.1 -p NOTE "weighted NCC, hand weights: must reproduce 007.1 on the M1" -p ARM ncc
fi
if [[ $1 == trained ]]; then
  run 023.2 -p NOTE "learned class weights (W), seed 42" -p ARM weights -p SEED 42
  run 023.3 -p NOTE "correlation head (H), seed 42: the idea" -p ARM head -p SEED 42
  run 023.4 -p NOTE "learned class weights (W), seed 7" -p ARM weights -p SEED 7
  run 023.5 -p NOTE "correlation head (H), seed 7" -p ARM head -p SEED 7
fi
echo "=== done $(date +%H:%M:%S)"
