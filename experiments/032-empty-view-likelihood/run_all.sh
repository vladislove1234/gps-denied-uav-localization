#!/usr/bin/env bash
# 032 from this folder: checks (a, b) -> tune β on tuning flights (2 shards) with check c -> 032.1 (head) || 032.2 (NCC).
# Stops after any failed check. Never edit this file while it runs.
set -euo pipefail
PM="$HOME/miniconda3/envs/uavloc/bin/papermill"
PY="$HOME/miniconda3/envs/uavloc/bin/python"
export OMP_NUM_THREADS=3
say() { echo "$(date +%H:%M) $*"; }
"$PM" checks.ipynb artifacts/checks-run.ipynb -k python3 > artifacts/checks.log 2>&1
"$PY" -c "import json, sys; c = json.load(open('artifacts/checks.json')); print(c); sys.exit(0 if all(v['pass'] for v in c.values()) else 1)" \
  || { say "checks a/b failed: stop"; exit 1; }
say "checks a, b pass"
for shard in 0 1; do
  "$PM" tune.ipynb "artifacts/tune-shard-$shard.ipynb" -k python3 -p SHARD "$shard" -p N_SHARDS 2 > "artifacts/tune-$shard.log" 2>&1 &
done
wait
BETA=$("$PY" - <<'PY'
import glob, json, sys
import pandas as pd
t = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("artifacts/tuning/shard-[0-9].csv"))])
off = float(t.loc[t["beta"] == 0.0, "mean_end_error_m"].iloc[0])
target = next(json.loads(l) for l in open("../027-sequence-filters/runs.jsonl") if json.loads(l)["run"] == "027.12")["config"]["tuning"]["tuning_mean_end_error_m"]
json.dump({"c_beta0_reproduces_027_12": {"beta0_m": off, "target_m": target, "abs_diff_m": abs(off - target),
                                          "pass": abs(off - target) < 1e-6}}, open("artifacts/check-c.json", "w"), indent=1)
if abs(off - target) >= 1e-6:
    sys.exit(f"check c failed: beta 0 gives {off}, 027.12 tuning {target}")
print(t.sort_values("mean_end_error_m").iloc[0]["beta"])
PY
) || { say "check c failed: stop"; exit 1; }
say "check c pass; tuned beta=$BETA"
"$PM" train.ipynb artifacts/run-head.ipynb -k python3 -p LIKELIHOOD head -p BETA "$BETA" -y "REGISTERED_RUN: '032.1'" \
  -y "BASELINE_RUN: '027.12'" > artifacts/test-head.log 2>&1 &
"$PM" train.ipynb artifacts/run-ncc.ipynb -k python3 -p LIKELIHOOD ncc -p BETA "$BETA" -y "REGISTERED_RUN: '032.2'" \
  -y "BASELINE_RUN: '027.4'" -p WAIT_FOR_RUNS 1 > artifacts/test-ncc.log 2>&1 &
wait
say "032 done ($(wc -l < runs.jsonl | tr -d ' ') runs)"
