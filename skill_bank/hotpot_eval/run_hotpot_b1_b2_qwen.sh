#!/usr/bin/env bash
set -euo pipefail

ROOT="eval/hotpot_b1_b2"
DATA_PATH="$ROOT/data/hotpotqa_dev_sample200_seed42.jsonl"
SCRIPT="skill_bank/nq_eval/eval_nq_qwen_skillbank.py"
MODEL_PATH="${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct"
B1_BANK="skill_bank/round_1_singlehop/outputs/round1_skill_bank.md"
B2_BANK="skill_bank/round_2_hotpotqa/outputs/round2_skill_bank.md"
CONDA_SH="/path/to/conda/etc/profile.d/conda.sh"
CONDA_ENV="/path/to/conda/envs/searchr1"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-1306001}"
GPU_B1="${GPU_B1:-2}"
GPU_B2="${GPU_B2:-3}"

mkdir -p "$ROOT/b1" "$ROOT/b2"

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_B1' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B1_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b1/hotpot_b1_trace.jsonl' \
    --out-json '$ROOT/b1/hotpot_b1_trace.json' \
    --summary-json '$ROOT/b1/hotpot_b1_summary.json' \
    --log-file '$ROOT/b1/hotpot_b1_run.log' \
    --retriever-host 127.0.0.1 \
    --retriever-port 8000 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 6 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b1/launcher.out" 2>&1 < /dev/null &
PID_B1=$!

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_B2' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B2_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b2/hotpot_b2_trace.jsonl' \
    --out-json '$ROOT/b2/hotpot_b2_trace.json' \
    --summary-json '$ROOT/b2/hotpot_b2_summary.json' \
    --log-file '$ROOT/b2/hotpot_b2_run.log' \
    --retriever-host 127.0.0.1 \
    --retriever-port 8000 \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 6 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b2/launcher.out" 2>&1 < /dev/null &
PID_B2=$!

echo "B1_PID=$PID_B1"
echo "B2_PID=$PID_B2"
