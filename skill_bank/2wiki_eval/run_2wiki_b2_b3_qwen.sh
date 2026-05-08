#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-eval/2wiki_b2_b3}"
DATA_PATH="${DATA_PATH:-eval/2wiki_b2_b3/data/2wiki_dev_sample200_seed42.jsonl}"
SCRIPT="${SCRIPT:-skill_bank/nq_eval/eval_nq_qwen_skillbank.py}"
MODEL_PATH="${MODEL_PATH:-${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct}"
B2_BANK="${B2_BANK:-skill_bank/round_2_hotpotqa/outputs/round2_skill_bank.md}"
B3_BANK="${B3_BANK:-skill_bank/round_3_2wiki/outputs/round3_skill_bank.md}"
CONDA_SH="${CONDA_SH:-/path/to/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda/envs/searchr1}"
RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-}"
GPU_B2="${GPU_B2:-2}"
GPU_B3="${GPU_B3:-3}"

mkdir -p "$ROOT/b2" "$ROOT/b3"

if [[ -z "$SLURM_JOB_ID_TARGET" ]] || ! command -v srun >/dev/null 2>&1; then
  echo "This comparison launcher requires Slurm. Set SLURM_JOB_ID_TARGET and make sure srun is available." >&2
  exit 2
fi

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_B2' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B2_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b2/2wiki_b2_trace.jsonl' \
    --out-json '$ROOT/b2/2wiki_b2_trace.json' \
    --summary-json '$ROOT/b2/2wiki_b2_summary.json' \
    --log-file '$ROOT/b2/2wiki_b2_run.log' \
    --retriever-host '$RETRIEVER_HOST' \
    --retriever-port '$RETRIEVER_PORT' \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 7 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b2/launcher.out" 2>&1 < /dev/null &
PID_B2=$!

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_B3' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B3_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b3/2wiki_b3_trace.jsonl' \
    --out-json '$ROOT/b3/2wiki_b3_trace.json' \
    --summary-json '$ROOT/b3/2wiki_b3_summary.json' \
    --log-file '$ROOT/b3/2wiki_b3_run.log' \
    --retriever-host '$RETRIEVER_HOST' \
    --retriever-port '$RETRIEVER_PORT' \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 7 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b3/launcher.out" 2>&1 < /dev/null &
PID_B3=$!

echo "B2_PID=$PID_B2"
echo "B3_PID=$PID_B3"
