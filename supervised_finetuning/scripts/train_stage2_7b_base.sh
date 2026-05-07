#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"

cd "$SEARCHSKILL_ROOT"

MODEL_DIR="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage2_7b_base
INIT_ADAPTER="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1_7b_base
TRAIN_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/train.jsonl
EVAL_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage2/eval.jsonl

if [[ "${CLEAN_MODEL_DIR:-0}" == "1" ]]; then
  rm -rf "$MODEL_DIR"
fi

test -s "$TRAIN_PATH"
test -s "$EVAL_PATH"
test -s ${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B/config.json
test -s "$INIT_ADAPTER/adapter_model.safetensors"
mkdir -p "$MODEL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29762}"

exec ${PYTHON_BIN:-/path/to/conda/env/bin/python} -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --master_port "$MASTER_PORT" \
  "$SEARCHSKILL_ROOT"/supervised_finetuning/scripts/train_lora.py \
  --model-path ${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B \
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
