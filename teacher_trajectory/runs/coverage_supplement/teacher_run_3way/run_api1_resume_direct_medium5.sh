#!/usr/bin/env bash
set -euo pipefail
ROOT=/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/teacher_trajectory
RUN_ROOT=$ROOT/runs/coverage_supplement
SHARD_ROOT=$RUN_ROOT/api1_remaining_after_api2/shards_3way
OUT_ROOT=$RUN_ROOT/teacher_run_3way
LOG_ROOT=$OUT_ROOT/logs
PYTHON_BIN=${PYTHON_BIN:-/path/to/conda/env/bin/python}
SKILL_BANK_PATH=/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_4_musique/outputs/final_skill_bank.md
OPENAI_ENV_PATH=/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/config/.openai_searchskill_env
source "$OPENAI_ENV_PATH"
export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,gpu031,gpu028}"
export no_proxy="${no_proxy:-$NO_PROXY}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY missing" >&2
  exit 2
fi
cat > "$OUT_ROOT/run_config.json" <<JSON
{
  "model": "gpt-5.4",
  "base_url": "https://w.ciykj.cn",
  "reasoning_effort": "medium",
  "max_steps": 5,
  "retriever_host": "gpu031",
  "retriever_port": "8000",
  "shards": 3,
  "manifest_root": "$SHARD_ROOT",
  "output_root": "$OUT_ROOT",
  "launcher": "direct_srun_wait"
}
JSON
pids=()
for shard_id in 0 1 2; do
  manifest="$SHARD_ROOT/manifest_shard_${shard_id}.jsonl"
  shard_out="$OUT_ROOT/shard_${shard_id}"
  log_path="$LOG_ROOT/shard_${shard_id}.log"
  mkdir -p "$shard_out"
  "$PYTHON_BIN" "$ROOT/src/run_teacher_rollout.py" \
    --manifest-path "$manifest" \
    --output-dir "$shard_out" \
    --skill-bank-path "$SKILL_BANK_PATH" \
    --base-url "https://w.ciykj.cn" \
    --model "gpt-5.4" \
    --reasoning-effort "medium" \
    --verbosity "medium" \
    --temperature 0.2 \
    --max-output-tokens 700 \
    --timeout-seconds 180 \
    --api-max-retries 4 \
    --api-retry-backoff 8 \
    --retriever-host "gpu031" \
    --retriever-port 8000 \
    --retriever-topk 3 \
    --retriever-timeout 45 \
    --max-steps 5 \
    --resume \
    --sleep-seconds 0.2 \
    > "$log_path" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "$OUT_ROOT/shard_${shard_id}.pid"
  echo "started shard_${shard_id} pid=$pid log=$log_path" >&2
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
