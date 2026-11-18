#!/bin/zsh
# Registered runs 018.1-018.4, in proposal order. Stops at the first failure so run ids stay aligned.
set -e
cd "$(dirname "$0")"
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
SEGFIX=experiments/014-boundary-displacement/artifacts
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
run 018.1 -p NOTE "baseline: LR-ASPP, no refiner (= 007.1)"
run 018.2 -p NOTE "primary: SegFix k8 seed42, s=4" -p REFINER segfix -p SEGFIX_CHECKPOINT $SEGFIX/segfix-k8-seed42.pt -p OFFSET_PX 4
run 018.3 -p NOTE "SegFix k8 seed42, s=8" -p REFINER segfix -p SEGFIX_CHECKPOINT $SEGFIX/segfix-k8-seed42.pt -p OFFSET_PX 8
run 018.4 -p NOTE "checkpoint swing: SegFix k8 seed43, s=4" -p REFINER segfix -p SEGFIX_CHECKPOINT $SEGFIX/segfix-k8-seed43.pt -p OFFSET_PX 4
echo "=== done $(date +%H:%M:%S)"
