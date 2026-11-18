#!/bin/zsh
# Registered runs of 024. `run_all.sh gate`: 024.1 (must reproduce 007.1). `run_all.sh checks`: acceptance checks
# (needs 024.1). `run_all.sh arms`: 024.2, 024.3. Stops at the first failure so run ids stay aligned.
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
  run 024.1 -p NOTE "weighted NCC, hand weights 0/1/1/0.5/0/0.5: control, must reproduce 007.1" -p MATCHING weighted-ncc
fi
if [[ $1 == checks ]]; then
  echo "=== checks $(date +%H:%M:%S)"
  $PY -m papermill checks.ipynb artifacts/checks-run.ipynb --cwd . -k python3 --log-level WARNING
fi
if [[ $1 == arms ]]; then
  run 024.2 -p NOTE "PMI class-compatibility table W counted on 023's 46 train villages" -p MATCHING pmi
  run 024.3 -p NOTE "mutual information of the soft class maps per offset" -p MATCHING mi
fi
echo "=== done $(date +%H:%M:%S)"
