#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=""${SEARCHSKILL_ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"/data_preparation"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ROOT_DIR}/logs"

"${PYTHON_BIN}" "${ROOT_DIR}/sample_singlehop_train.py" \
  --root-dir "${ROOT_DIR}" \
  --datasets nq triviaqa \
  --target-size auto \
  --overwrite-existing \
  | tee "${ROOT_DIR}/logs/run_singlehop_sampling.log"
