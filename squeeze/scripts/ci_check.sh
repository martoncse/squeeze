#!/usr/bin/env bash
# Smoke check, no key and no Modal account needed: compile, unit tests, and the full pipeline on the
# offline mock models (writes to results_mock/ and data_mock/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" -m compileall -q src modal_serve.py
"$PY" -m pytest -q tests
"$PY" -m src.gen_dataset --mock --limit-templates 30
for m in exp1_matrix exp2_detectors exp3_paraphrase "exp4_squeeze --icrl"; do
  "$PY" -m src.$m --mock --limit-templates 30 > /dev/null
  echo "ok: $m"
done
echo "ci_check passed"
