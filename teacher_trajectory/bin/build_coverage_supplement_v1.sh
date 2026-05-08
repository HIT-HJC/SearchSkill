#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-teacher_trajectory}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN:-/path/to/conda/env/bin/python}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/coverage_supplement}"

cd "$ROOT"
"$PYTHON_BIN" src/build_manifest_coverage_supplement.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"
