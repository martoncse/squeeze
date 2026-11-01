#!/usr/bin/env bash
# Run Exps 1-4 in order against the deployed Modal app, one log per experiment in results/logs/.
# Stops at the first failure; rerunning resumes from the cache and results/raw/.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p results/logs
for m in exp1_matrix exp2_detectors exp3_paraphrase "exp4_squeeze --icrl"; do
  name="${m%% *}"
  echo "[run_all] $(date +%H:%M:%S) start $m"
  if ! PYTHONIOENCODING=utf-8 "$PY" -m src.$m > "results/logs/$name.log" 2>&1; then
    echo "[run_all] $(date +%H:%M:%S) FAILED $m (see results/logs/$name.log)"; tail -20 "results/logs/$name.log"; exit 1
  fi
  echo "[run_all] $(date +%H:%M:%S) done $m"; grep -h '"H[0-9]_supported"' results/exp*_summary.json 2>/dev/null | sort -u
done
echo "[run_all] all experiments finished"
