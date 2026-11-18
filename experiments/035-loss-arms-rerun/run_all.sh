#!/bin/zsh
# 035: audit -> three system re-runs from 033 artifacts (3 lanes x 2 threads, as 033's test runs) -> analysis.
#   run_all.sh                  everything
#   run_all.sh <step> [...]     some of: audit runs analysis
# Each arm's segmenter, head and T are passed explicitly, so the mapping is in the run config.
# Launch detached: nohup python -c "import os; os.setsid(); os.execvp('zsh', ['zsh', 'run_all.sh'])" &  (cerebrum 2026-11-02; zsh, not bash: arrays)
cd "$(dirname "$0")"
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
OUT=artifacts/runs
SRC033=experiments/033-class-loss-in-system/artifacts
mkdir -p "$OUT"
if (( $# )); then steps=($@); else steps=(audit runs analysis); fi

job() {  # job <registered_run> <arm> <source_run> <temperature>
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PM" train.ipynb "$OUT/run-$1.ipynb" --cwd . -k python3 --log-level WARNING \
    -p ARM "$2" -p SEG_CHECKPOINT "$SRC033/segmenter-$2/model-bn-satellite.pt" -p HEAD_PATH "$SRC033/head-$2.pt" \
    -p TEMPERATURE "$4" -y "REGISTERED_RUN: '$1'" -y "SOURCE_RUN: '$3'" > "$OUT/run-$1.log" 2>&1
  echo "$1 exit $? $(date +%H:%M)" >> "$OUT/status"
}

for step in $steps; do
  echo "=== $step $(date +%H:%M:%S)" >> "$OUT/status"
  case $step in
    audit)
      OMP_NUM_THREADS=3 "$PM" audit.ipynb "$OUT/audit.ipynb" --cwd . -k python3 --log-level WARNING > "$OUT/audit.log" 2>&1
      echo "audit exit $?" >> "$OUT/status" ;;
    runs)
      job 035.1 control 033.1 0.5 &
      job 035.2 building 033.2 0.5 &
      job 035.3 structure 033.3 0.5 &
      wait ;;
    analysis)
      "$PM" analysis.ipynb "$OUT/analysis.ipynb" --cwd . -k python3 --log-level WARNING > "$OUT/analysis.log" 2>&1
      echo "analysis exit $?" >> "$OUT/status" ;;
  esac
done
echo "ALL_DONE $(date +%H:%M)" >> "$OUT/status"
