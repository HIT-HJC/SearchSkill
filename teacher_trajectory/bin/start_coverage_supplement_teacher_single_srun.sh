#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-teacher_trajectory}"
RUN_ROOT="$ROOT/runs/coverage_supplement"
OUT_ROOT="$RUN_ROOT/teacher_run_single"
OPENAI_ENV_PATH="${OPENAI_ENV_PATH:-${SEARCHSKILL_ROOT:-$(pwd)}/config/.openai_searchskill_env}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN:-/path/to/conda/env/bin/python}}"
SKILL_BANK_PATH="${SKILL_BANK_PATH:-${SEARCHSKILL_ROOT:-$(pwd)}/skill_bank/round_4_musique/outputs/final_skill_bank.md}"

mkdir -p "$OUT_ROOT"
cd "$ROOT"

setsid srun --jobid=1313825 --overlap --ntasks=1 bash -lc "
  cd '$ROOT' &&
  source '$OPENAI_ENV_PATH' &&
  export PYTHONUNBUFFERED=1 &&
  '$PYTHON_BIN' '$ROOT/src/run_teacher_rollout.py' \
    --manifest-path '$RUN_ROOT/manifest.jsonl' \
    --output-dir '$OUT_ROOT' \
    --skill-bank-path '$SKILL_BANK_PATH' \
    --base-url '${OPENAI_BASE_URL:-https://api.openai.com/v1}' \
    --model 'gpt-5.4' \
    --reasoning-effort 'xhigh' \
    --verbosity 'medium' \
    --temperature 0.2 \
    --max-output-tokens 700 \
    --timeout-seconds 240 \
    --api-max-retries 10 \
    --api-retry-backoff 15 \
    --retriever-host '${RETRIEVER_HOST:-127.0.0.1}' \
    --retriever-port 8000 \
    --retriever-topk 3 \
    --retriever-timeout 45 \
    --max-steps 6 \
    --resume \
    --sleep-seconds 0.5
" > "$OUT_ROOT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT_ROOT/master_srun.pid"
echo "started single master pid=$(cat "$OUT_ROOT/master_srun.pid") log=$OUT_ROOT/run.log"
