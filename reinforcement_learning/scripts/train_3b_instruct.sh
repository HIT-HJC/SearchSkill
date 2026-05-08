#!/usr/bin/env bash
set -euo pipefail

export ROOT="${ROOT:-/path/to/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
export GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_BATCH_SIZE=8
export VAL_BATCH_SIZE=80
export ROLLOUT_N=8
export ROLLOUT_BACKEND=hf
export PPO_MINI_BATCH_SIZE=32
export PPO_MICRO_BATCH_SIZE=4
export LOG_PROB_MICRO_BATCH_SIZE=32
export REF_LOG_PROB_MICRO_BATCH_SIZE=32
export TOTAL_STEPS=150
export SAVE_FREQ=15
export TEST_FREQ=15
export VAL_BEFORE_TRAIN=false
export MODEL_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage2_3b_instruct_merged
export RUN_NAME=rl_3b_instruct_from_stage2_4gpu
export OUT_DIR="$SEARCHSKILL_ROOT"/reinforcement_learning/runs/rl_3b_instruct_from_stage2_4gpu
export DATA_DIR="$SEARCHSKILL_ROOT"/reinforcement_learning/data/policy_3b_instruct
export LOG_DIR="$SEARCHSKILL_ROOT"/reinforcement_learning/logs
export LR=1e-6
export KL=0.001
export RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
export RETRIEVER_PORT=8000

cd "$ROOT"
bash reinforcement_learning/scripts/launch_training.sh
