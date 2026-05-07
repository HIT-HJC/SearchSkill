#!/usr/bin/env bash
set -euo pipefail

ROOT="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/eval/nq_b0_b1"
DATA_PATH="$ROOT/data/nq_dev_sample100_seed42.jsonl"
SCRIPT="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/nq_eval/eval_nq_qwen_skillbank.py"
MODEL_PATH="${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct"
B0_BANK="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/inputs/seed_skill_bank.md"
B1_BANK="/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_1_singlehop/outputs/round1_skill_bank.md"
CONDA_SH="/online1/public/support/amd/Ananconda3/2023.3/etc/profile.d/conda.sh"
CONDA_ENV="/online1/ycsc_chenkh/hitici_11/.conda/envs/searchr1"

mkdir -p "$ROOT/b0" "$ROOT/b1"

nohup bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES=2 python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B0_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b0/nq_b0_trace.jsonl' \
    --out-json '$ROOT/b0/nq_b0_trace.json' \
    --summary-json '$ROOT/b0/nq_b0_summary.json' \
    --log-file '$ROOT/b0/nq_b0_run.log' \
    --retriever-host 127.0.0.1 \
    --retriever-port 8000 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 5 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b0/launcher.out" 2>&1 < /dev/null &
PID_B0=$!

nohup bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES=3 python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B1_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b1/nq_b1_trace.jsonl' \
    --out-json '$ROOT/b1/nq_b1_trace.json' \
    --summary-json '$ROOT/b1/nq_b1_summary.json' \
    --log-file '$ROOT/b1/nq_b1_run.log' \
    --retriever-host 127.0.0.1 \
    --retriever-port 8000 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 5 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b1/launcher.out" 2>&1 < /dev/null &
PID_B1=$!

echo "B0_PID=$PID_B0"
echo "B1_PID=$PID_B1"
