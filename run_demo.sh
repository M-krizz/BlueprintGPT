#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PYTHON="d:/Projects/GenAI/.venv/Scripts/python.exe"

run_demo() {
  local label="$1"
  local module_name="$2"
  echo "== ${label} =="
  "$PYTHON" -m "$module_name"
  echo
}

if [[ "$MODE" == "learned" || "$MODE" == "all" ]]; then
  run_demo "Learned demo" "learned.integration.run_smoke_learned"
fi

if [[ "$MODE" == "algorithmic" || "$MODE" == "all" ]]; then
  run_demo "Algorithmic baseline demo" "demo.run_smoke_algorithmic"
fi
