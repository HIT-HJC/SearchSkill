#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SEARCHSKILL_ROOT:-$(pwd)}/data_preparation}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ROOT_DIR}/logs"

"${PYTHON_BIN}" "${ROOT_DIR}/sample_multihop_train.py" \
  --root-dir "${ROOT_DIR}" \
  --datasets hotpotqa 2wiki musique \
  --target-size auto \
  --max-workers 6 \
  --group-batch-size 12 \
  --group-representatives 4 \
  --max-gpt-groups 0 \
  --candidate-buffer-ratio 1.0 \
  --model gpt-5.4 \
  --model-base-url "${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
  --reasoning-effort xhigh \
  --overwrite-existing \
  | tee "${ROOT_DIR}/logs/run_multihop_sampling.log"
