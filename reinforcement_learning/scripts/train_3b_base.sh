#!/usr/bin/env bash
set -euo pipefail

export ROOT="${ROOT:-/path/to/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
export GPUS="${GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-80}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-hf}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-150}"
export SAVE_FREQ="${SAVE_FREQ:-15}"
export TEST_FREQ="${TEST_FREQ:-15}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
export MODEL_PATH="${MODEL_PATH:-$SEARCHSKILL_ROOT/supervised_finetuning/models/stage2_3b_base_merged}"
export RUN_NAME="${RUN_NAME:-rl_3b_base_from_stage2_4gpu}"
export OUT_DIR="${OUT_DIR:-$SEARCHSKILL_ROOT/reinforcement_learning/runs/rl_3b_base_from_stage2_4gpu}"
export DATA_DIR="${DATA_DIR:-$SEARCHSKILL_ROOT/reinforcement_learning/data/policy_3b_base}"
export LOG_DIR="${LOG_DIR:-$SEARCHSKILL_ROOT/reinforcement_learning/logs}"
export LR="${LR:-1e-6}"
export KL="${KL:-0.001}"
export RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"

cd "$ROOT"
bash reinforcement_learning/scripts/launch_training.sh
