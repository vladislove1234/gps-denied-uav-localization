#!/bin/zsh
# Pre-registered runs 007.1-007.9, then the not-pre-registered 007.10. Stops at the first failure so run ids stay aligned.
set -e
cd "$(dirname "$0")"
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
run 007.1 -p NOTE "baseline: rect scale 1.7, flat prior, no gate (re-run of 006.1)"
run 007.2 -p NOTE "rect scale 3.0 (005.3's window) on current code" -p SEARCH_SCALE 3.0
run 007.3 -p NOTE "round R 20 m, flat prior: window shape alone" -p WINDOW round -p SEARCH_RADIUS_M 20
run 007.4 -p NOTE "round R 20 m + prior sigma 10 m (rule A)" -p WINDOW round -p SEARCH_RADIUS_M 20 -p PRIOR_SIGMA_M 10
run 007.5 -p NOTE "rect scale 1.7 + prior sigma 10 m: prior alone" -p PRIOR_SIGMA_M 10
run 007.6 -p NOTE "round R 30 m + prior sigma 10 m" -p WINDOW round -p SEARCH_RADIUS_M 30 -p PRIOR_SIGMA_M 10
run 007.7 -p NOTE "007.4 + full gate (rule B)" -p WINDOW round -p SEARCH_RADIUS_M 20 -p PRIOR_SIGMA_M 10 -p GATE full
run 007.8 -p NOTE "rect scale 1.7, flat prior + full gate: gate alone" -p GATE full
run 007.9 -p NOTE "007.4 + accept/reject gate: value of across-only" -p WINDOW round -p SEARCH_RADIUS_M 20 -p PRIOR_SIGMA_M 10 -p GATE accept-reject
run 007.10 -p NOTE "NOT PRE-REGISTERED: 007.7 with temperature tuned jointly with the gate" -p WINDOW round -p SEARCH_RADIUS_M 20 -p PRIOR_SIGMA_M 10 -p GATE full -p TEMPERATURE_BY gate
echo "=== done $(date +%H:%M:%S)"
