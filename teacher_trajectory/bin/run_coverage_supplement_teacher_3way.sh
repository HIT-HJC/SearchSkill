#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/teacher_trajectory}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/coverage_supplement}"
SHARD_ROOT="${SHARD_ROOT:-$RUN_ROOT/shards_3way}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/teacher_run_3way}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN:-/path/to/conda/env/bin/python}}"
SKILL_BANK_PATH="${SKILL_BANK_PATH:-"${SEARCHSKILL_ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"/skill_bank/round_4_musique/outputs/final_skill_bank.md}"
OPENAI_ENV_PATH="${OPENAI_ENV_PATH:-"${SEARCHSKILL_ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"/config/.openai_searchskill_env}"

MODEL="${MODEL:-gpt-5.4}"
BASE_URL="${BASE_URL:-https://w.ciykj.cn}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
VERBOSITY="${VERBOSITY:-medium}"
RETRIEVER_HOST="${RETRIEVER_HOST:-gpu031}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
API_RETRY_BACKOFF="${API_RETRY_BACKOFF:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
MAX_STEPS="${MAX_STEPS:-6}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.2}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ -f "$OPENAI_ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$OPENAI_ENV_PATH"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Put it in $OPENAI_ENV_PATH or export it before running." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,gpu031,gpu028}"
export no_proxy="${no_proxy:-$NO_PROXY}"

for shard_id in 0 1 2; do
  manifest="$SHARD_ROOT/manifest_shard_${shard_id}.jsonl"
  shard_out="$OUT_ROOT/shard_${shard_id}"
  log_path="$LOG_ROOT/shard_${shard_id}.log"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing shard manifest: $manifest" >&2
    exit 3
  fi
  mkdir -p "$shard_out"
  nohup "$PYTHON_BIN" "$ROOT/src/run_teacher_rollout.py" \
    --manifest-path "$manifest" \
    --output-dir "$shard_out" \
    --skill-bank-path "$SKILL_BANK_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --verbosity "$VERBOSITY" \
    --temperature 0.2 \
    --max-output-tokens 700 \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --api-max-retries "$API_MAX_RETRIES" \
    --api-retry-backoff "$API_RETRY_BACKOFF" \
    --retriever-host "$RETRIEVER_HOST" \
    --retriever-port "$RETRIEVER_PORT" \
    --retriever-topk 3 \
    --retriever-timeout 45 \
    --max-steps "$MAX_STEPS" \
    --resume \
    --sleep-seconds "$SLEEP_SECONDS" \
    > "$log_path" 2>&1 &
  echo "$!" > "$OUT_ROOT/shard_${shard_id}.pid"
  echo "started shard_${shard_id} pid=$(cat "$OUT_ROOT/shard_${shard_id}.pid") log=$log_path"
done

cat > "$OUT_ROOT/run_config.json" <<JSON
{
  "model": "$MODEL",
  "base_url": "$BASE_URL",
  "reasoning_effort": "$REASONING_EFFORT",
  "retriever_host": "$RETRIEVER_HOST",
  "retriever_port": "$RETRIEVER_PORT",
  "shards": 3,
  "manifest_root": "$SHARD_ROOT",
  "output_root": "$OUT_ROOT"
}
JSON

if [[ "${WAIT_FOR_SHARDS:-0}" == "1" ]]; then
  wait \
    "$(cat "$OUT_ROOT/shard_0.pid")" \
    "$(cat "$OUT_ROOT/shard_1.pid")" \
    "$(cat "$OUT_ROOT/shard_2.pid")"
fi
