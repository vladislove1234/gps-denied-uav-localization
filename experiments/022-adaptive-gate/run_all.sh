#!/bin/zsh
# 022.1 (reproduces 007.8) first, then 022.2-022.4. Stops at the first failure so run ids stay aligned.
set -e
cd "$(dirname "$0")"
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
# done separately first (reproduction gate): run 022.1 -p NOTE "static full gate, LOPO-tuned as 007: must reproduce 007.8" -p ARM static -p GATE full
# 022.2 done: run 022.2 -p NOTE "static accept-reject gate, LOPO-tuned: like-for-like baseline" -p ARM static -p GATE accept-reject
run 022.3 -p NOTE "risk gate, shape at tau 0.25 only" -p ARM risk -y "GATE_TAUS: [0.25]"
run 022.4 -p NOTE "risk gate, shape at tau 0.25 0.5 1 2: the idea" -p ARM risk -y "GATE_TAUS: [0.25, 0.5, 1.0, 2.0]"
echo "=== done $(date +%H:%M:%S)"
