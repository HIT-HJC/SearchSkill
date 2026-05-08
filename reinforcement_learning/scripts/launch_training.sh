#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/external/runtime}"
PY="${PY:-${PYTHON_BIN:-python}}"

MODEL_PATH="${MODEL_PATH:-$ROOT/supervised_finetuning/models/stage2_7b_instruct_merged}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/reinforcement_learning/source_data/policy_training_pool/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-$ROOT/reinforcement_learning/source_data/policy_training_pool/dev.jsonl}"
SKILL_BANK_PATH="${SKILL_BANK_PATH:-$ROOT/skill_bank/round_4_musique/outputs/final_skill_bank.md}"
DATA_DIR="${DATA_DIR:-$ROOT/reinforcement_learning/data/policy_data}"
RUN_NAME="${RUN_NAME:-rl_from_stage2_$(date +%m%d_%H%M)}"
OUT_DIR="${OUT_DIR:-$ROOT/reinforcement_learning/runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT/reinforcement_learning/logs}"

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-80}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-76}"
SAVE_FREQ="${SAVE_FREQ:-15}"
TEST_FREQ="${TEST_FREQ:-15}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
LR="${LR:-1e-6}"
KL="${KL:-0.001}"
TP="${TP:-1}"
GPUS="${GPUS:-4}"
NNODES="${NNODES:-1}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-hf}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-32}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$LOG_PROB_MICRO_BATCH_SIZE}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"
MAX_START_LENGTH="${MAX_START_LENGTH:-4096}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-1400}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.98}"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"

"$PY" "$ROOT/reinforcement_learning/scripts/build_policy_dataset.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --dev-jsonl "$DEV_JSONL" \
  --skill-bank-path "$SKILL_BANK_PATH" \
  --output-dir "$DATA_DIR" | tee "$OUT_DIR/data_build.log"

if [[ ! -d "$RUNTIME_ROOT" ]]; then
  echo "RUNTIME_ROOT does not exist: $RUNTIME_ROOT" >&2
  echo "Install your RL runtime there or set RUNTIME_ROOT to its path. See external/README.md." >&2
  exit 2
fi

cd "$RUNTIME_ROOT"
export PYTHONPATH="$RUNTIME_ROOT:${PYTHONPATH:-}"
export SEARCHSKILL_SKILL_BANK_PATH="$SKILL_BANK_PATH"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=XFORMERS
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export PYTORCH_JIT=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SEARCHSKILL_DISABLE_FLASH_CE=1
export SEARCHSKILL_RETRIEVER_CHUNK="${SEARCHSKILL_RETRIEVER_CHUNK:-1}"
export SEARCHSKILL_FORCE_CUDA_VISIBLE="${SEARCHSKILL_FORCE_CUDA_VISIBLE:-1}"
export SEARCHSKILL_ALL_CUDA_VISIBLE="${SEARCHSKILL_ALL_CUDA_VISIBLE:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export SEARCHSKILL_DISABLE_RAY_GPU_OPTIONS="${SEARCHSKILL_DISABLE_RAY_GPU_OPTIONS:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_NUM_GPUS="${RAY_NUM_GPUS:-$GPUS}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlfinal_${RUN_NAME:0:24}}"
mkdir -p "$RAY_TMPDIR"
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:-},localhost,127.0.0.1,::1"

"$PY" -m verl.trainer.main_ppo_searchskill \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" \
  data.max_obs_length="$MAX_OBS_LENGTH" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="$KL" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n_agent="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  trainer.logger=['console'] \
  +trainer.val_only=false \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=SearchSkill \
  trainer.experiment_name="$RUN_NAME" \
  trainer.total_epochs=20 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUT_DIR/checkpoints" \
  max_turns=5 \
  do_search=true \
  retriever.url="http://$RETRIEVER_HOST:$RETRIEVER_PORT/retrieve" \
  retriever.topk=3 2>&1 | tee "$OUT_DIR/train.log"
