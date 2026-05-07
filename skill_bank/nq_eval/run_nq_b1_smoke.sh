#!/usr/bin/env bash
set -euo pipefail

ROOT="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/eval/nq_b0_b1"
DATA_PATH="$ROOT/data/nq_dev_sample100_seed42.jsonl"
SCRIPT="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/nq_eval/eval_nq_qwen_skillbank.py"
MODEL_PATH="${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct"
B1_BANK="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_1_singlehop/outputs/round1_skill_bank.md"
CONDA_SH="/online1/public/support/amd/Ananconda3/2023.3/etc/profile.d/conda.sh"
CONDA_ENV="/online1/ycsc_chenkh/hitici_11/.conda/envs/searchr1"

mkdir -p "$ROOT/b1"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python "$SCRIPT" \
  --data-path "$DATA_PATH" \
  --skill-bank-path "$B1_BANK" \
  --model-path "$MODEL_PATH" \
  --out-jsonl "$ROOT/b1/nq_b1_smoke_trace.jsonl" \
  --out-json "$ROOT/b1/nq_b1_smoke_trace.json" \
  --summary-json "$ROOT/b1/nq_b1_smoke_summary.json" \
  --log-file "$ROOT/b1/nq_b1_smoke_run.log" \
  --retriever-host 127.0.0.1 \
  --retriever-port 8000 \
  --temperature 0.2 \
  --top-p 0.95 \
  --max-steps 5 \
  --dtype float32 \
  --print-every 1 \
  --max-samples 5
