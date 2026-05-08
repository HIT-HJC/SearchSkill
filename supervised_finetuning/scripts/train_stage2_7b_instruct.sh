#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"

cd "$SEARCHSKILL_ROOT"

MODEL_DIR="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage2
TRAIN_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/train.jsonl
EVAL_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/eval.jsonl

if [[ "${CLEAN_MODEL_DIR:-0}" == "1" ]]; then
  rm -rf "$MODEL_DIR"
fi

test -s "$TRAIN_PATH"
test -s "$EVAL_PATH"
test -s "$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1/adapter_model.safetensors
mkdir -p "$MODEL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29729}"

exec ${PYTHON_BIN:-python} -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --master_port "$MASTER_PORT" \
  "$SEARCHSKILL_ROOT"/supervised_finetuning/scripts/train_lora.py \
  --model-path ${HF_MODELS:?Set HF_MODELS to your local model root}/Qwen2.5-7B-Instruct \
  --init-adapter-path "$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1 \
  --train-path "$TRAIN_PATH" \
  --eval-path "$EVAL_PATH" \
  --output-dir "$MODEL_DIR" \
  --max-length "${MAX_LENGTH:-12288}" \
  --truncate-side left \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1.0}" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --warmup-ratio 0.03 \
  --logging-steps 5 \
  --save-steps 40 \
  --eval-steps 40 \
  --dataloader-num-workers 0 \
  --answer-loss-weight "${ANSWER_LOSS_WEIGHT:-2.0}" \
  --search-loss-weight "${SEARCH_LOSS_WEIGHT:-0.8}"
