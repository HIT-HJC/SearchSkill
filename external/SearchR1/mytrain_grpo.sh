#!/usr/bin/env bash
set -e

# 让 Ray 不够内存时把对象写到磁盘
RAY_SPILL_DIR="${RAY_SPILL_DIR:-/tmp/searchskill_ray_spill}"
export RAY_OBJECT_SPILLING_CONFIG="{"type":"filesystem","params":{"directory_path":"$RAY_SPILL_DIR"}}"
mkdir -p "$RAY_SPILL_DIR"
# 内存使用到 95% 才触发保护，避免过早杀进程
export RAY_memory_usage_threshold=0.95

# 尽量把最大线程/进程数和打开文件数调高（如果系统允许）
ulimit -u unlimited 2>/dev/null || ulimit -u 65535 2>/dev/null || true
ulimit -n 1048576  2>/dev/null || ulimit -n 65535 2>/dev/null || true
# 控制每个 worker 的线程数，避免额外浪费
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ray 噪声/遥测（可选）
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0

# —— SwanLab 追踪（按需启用/修改）——
# 必要：项目与实验名
export SWANLAB_PROJECT="Search-R1"
# 若离线：
# export SWANLAB_MODE=${SWANLAB_MODE:-offline}
# export SWANLAB_OFFLINE_DIR=${SWANLAB_OFFLINE_DIR:-${SEARCHR1_ROOT:-$(pwd)}/swanlab_offline}

DATA_DIR="${DATA_DIR:-${HF_DATA:-/path/to/hf_data}/nq_hotpotqa_train}"
BASE_MODEL="${BASE_MODEL:-${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B}"
SEARCHR1_ROOT="${SEARCHR1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$SEARCHR1_ROOT/verl_checkpoints}"
LOG_ROOT="${LOG_ROOT:-$SEARCHR1_ROOT/rl_logs}"

data_name="nq_hotpotqa_train"
EXPERIMENT_NAME="${data_name}-search-r1-grpo-qwen2.5-7b-em-structureformat-local"
# 可选：自定义运行名（用你的实验名更好看）
export SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-${EXPERIMENT_NAME}}"
CKPT_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_ATTENTION_BACKEND="XFORMERS"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512\
    data.val_batch_size=256\
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=[] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name="Search-R1" \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir="${CKPT_DIR}" \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ${LOG_FILE}
