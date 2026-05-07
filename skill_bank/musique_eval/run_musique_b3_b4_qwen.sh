#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/eval/musique_b3_b4}"
DATA_PATH="${DATA_PATH:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/eval/musique_b3_b4/data/musique_dev_smallval100_seed42.jsonl}"
SCRIPT="${SCRIPT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/nq_eval/eval_nq_qwen_skillbank.py}"
MODEL_PATH="${MODEL_PATH:-${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct}"
B3_BANK="${B3_BANK:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_3_2wiki/outputs/round3_skill_bank.md}"
final_BANK="${final_BANK:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_4_musique/outputs/final_skill_bank.md}"
CONDA_SH="${CONDA_SH:-/online1/public/support/amd/Ananconda3/2023.3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/online1/ycsc_chenkh/hitici_11/.conda/envs/searchr1}"
RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-1306001}"
GPU_B3="${GPU_B3:-2}"
GPU_final="${GPU_final:-3}"

mkdir -p "$ROOT/b3" "$ROOT/b4"

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_B3' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$B3_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b3/musique_b3_trace.jsonl' \
    --out-json '$ROOT/b3/musique_b3_trace.json' \
    --summary-json '$ROOT/b3/musique_b3_summary.json' \
    --log-file '$ROOT/b3/musique_b3_run.log' \
    --retriever-host '$RETRIEVER_HOST' \
    --retriever-port '$RETRIEVER_PORT' \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 8 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b3/launcher.out" 2>&1 < /dev/null &
PID_B3=$!

nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && \
  CUDA_VISIBLE_DEVICES='$GPU_final' python '$SCRIPT' \
    --data-path '$DATA_PATH' \
    --skill-bank-path '$final_BANK' \
    --model-path '$MODEL_PATH' \
    --out-jsonl '$ROOT/b4/musique_b4_trace.jsonl' \
    --out-json '$ROOT/b4/musique_b4_trace.json' \
    --summary-json '$ROOT/b4/musique_b4_summary.json' \
    --log-file '$ROOT/b4/musique_b4_run.log' \
    --retriever-host '$RETRIEVER_HOST' \
    --retriever-port '$RETRIEVER_PORT' \
    --temperature 0.2 \
    --top-p 0.95 \
    --max-steps 8 \
    --dtype float32 \
    --print-every 10
" > "$ROOT/b4/launcher.out" 2>&1 < /dev/null &
PID_final=$!

echo "B3_PID=$PID_B3"
echo "final_PID=$PID_final"
