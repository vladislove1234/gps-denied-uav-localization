#!/bin/zsh
# The pre-registered runs. log_run numbers runs in the order they are logged, so the logged id equals the
# proposal's label only when all four run in order; REGISTERED_RUN keeps the label in the config. Stops at the first failure.
# Usage: ./run_all.sh [checks|003.1|003.2|003.3|003.4] ... (no argument: all four runs, in order)
set -e
cd "$(dirname "$0")"
mkdir -p artifacts
PY=/Users/vladyslavtaraban/miniconda3/envs/uavloc/bin/python
run() {
  local id=$1; shift
  echo "=== $id $(date +%H:%M:%S)"
  $PY -m papermill train.ipynb artifacts/run-$id.ipynb --cwd . -k python3 --log-level WARNING "$@"
}
steps=(${@:-003.1 003.2 003.3 003.4})
for step in $steps; do
  case $step in
    checks) echo "=== checks $(date +%H:%M:%S)"
            $PY -m papermill checks.ipynb artifacts/checks.ipynb --cwd . -k python3 --log-level WARNING ;;
    003.1) run 003.1 -y "REGISTERED_RUN: '003.1'" -p NOTE "registered 003.1: class maps, weighted NCC, heading search; Apple M1 Pro; 001.1 replays at 11.469 m here vs 11.634 logged (Intel)" \
             -p METHOD segmentation-labels -p MATCHING weighted-ncc -y "HEADING_SEARCH: true" -y "SCALE_SEARCH: false" ;;
    003.2) run 003.2 -y "REGISTERED_RUN: '003.2'" -p POSE_SCHEDULE joint -p FINE_PASS_TOP_K 3 -p NOTE "registered 003.2 (joint schedule, fine pass top 3, 2026-10-20 amendments 1+2): class maps, weighted NCC, heading and scale search; Apple M1 Pro; 001.1 replays at 11.469 m here vs 11.634 logged (Intel)" \
             -p METHOD segmentation-labels -p MATCHING weighted-ncc -y "HEADING_SEARCH: true" -y "SCALE_SEARCH: true" ;;
    003.3) run 003.3 -y "REGISTERED_RUN: '003.3'" -p NOTE "registered 003.3: control: VGG11, cosine, heading search; Apple M1 Pro; 001.1 replays at 11.469 m here vs 11.634 logged (Intel)" \
             -p METHOD vgg11 -p MATCHING cosine -y "HEADING_SEARCH: true" -y "SCALE_SEARCH: false" ;;
    003.4) run 003.4 -y "REGISTERED_RUN: '003.4'" -p POSE_SCHEDULE joint -p FINE_PASS_TOP_K 3 -p NOTE "registered 003.4 (joint schedule, fine pass top 3, 2026-10-20 amendments 1+2): control: VGG11, cosine, heading and scale search; Apple M1 Pro; 001.1 replays at 11.469 m here vs 11.634 logged (Intel)" \
             -p METHOD vgg11 -p MATCHING cosine -y "HEADING_SEARCH: true" -y "SCALE_SEARCH: true" ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "=== done $(date +%H:%M:%S)"
