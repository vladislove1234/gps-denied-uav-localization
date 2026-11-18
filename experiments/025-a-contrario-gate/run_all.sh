#!/bin/zsh
# 025.1 (must reproduce 022.2) first, then the two pre-registered NFA arms. Stops at the first failure so run ids
# stay aligned. Usage: ./run_all.sh [025.1|025.2|025.3|checks] ... (no argument: all, in order)
set -e
cd "$(dirname "$0")"
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
steps=(${@:-025.1 checks 025.2 025.3})
for step in $steps; do
  case $step in
    025.1) run 025.1 -p NOTE "static accept-reject gate, LOPO-tuned: must reproduce 022.2" -p ARM static ;;
    checks) echo "=== checks $(date +%H:%M:%S)"
            $PY -m papermill checks.ipynb artifacts/checks-run.ipynb --cwd . -k python3 --log-level WARNING ;;
    025.2) run 025.2 -p NOTE "NFA < 1, null: 200 windows on the 023 train villages" -p ARM nfa-villages ;;
    025.3) run 025.3 -p NOTE "NFA < 1, null: 200 windows on the own map, >= 60 m from the prior" -p ARM nfa-own-map ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "=== done $(date +%H:%M:%S)"
