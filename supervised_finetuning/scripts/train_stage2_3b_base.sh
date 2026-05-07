#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"

cd "$SEARCHSKILL_ROOT"

MODEL_DIR="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage2_3b_base
INIT_ADAPTER="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1_3b_base
TRAIN_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/train.jsonl
EVAL_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/eval.jsonl
BASE_MODEL=${HF_MODELS:-/path/to/hf_models}/Qwen2.5-3B
CACHE_ROOT=${HF_CACHE:-/path/to/hf_cache}

if [[ "${CLEAN_MODEL_DIR:-0}" == "1" ]]; then
  rm -rf "$MODEL_DIR"
fi

test -s "$TRAIN_PATH"
test -s "$EVAL_PATH"
test -s "$BASE_MODEL/config.json"
test -s "$INIT_ADAPTER/adapter_model.safetensors"
mkdir -p "$MODEL_DIR" "$CACHE_ROOT" "$CACHE_ROOT/transformers" "$CACHE_ROOT/datasets" "$CACHE_ROOT/hub" "$CACHE_ROOT/xdg"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$CACHE_ROOT/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,gpu031}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29764}"

exec ${PYTHON_BIN:-/path/to/conda/env/bin/python} -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --master_port "$MASTER_PORT" \
  "$SEARCHSKILL_ROOT"/supervised_finetuning/scripts/train_lora.py \
  --model-path "$BASE_MODEL" \
  --init-adapter-path "$INIT_ADAPTER" \
  --train-path "$TRAIN_PATH" \
  --eval-path "$EVAL_PATH" \
  --output-dir "$MODEL_DIR" \
  --max-length "${MAX_LENGTH:-12288}" \
  --truncate-side left \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1.0}" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --warmup-ratio 0.03 \
  --logging-steps 5 \
  --save-steps 40 \
  --eval-steps 40 \
  --dataloader-num-workers 0 \
  --answer-loss-weight "${ANSWER_LOSS_WEIGHT:-2.0}" \
  --search-loss-weight "${SEARCH_LOSS_WEIGHT:-0.8}"
