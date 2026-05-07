#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
cd "$SEARCHSKILL_ROOT"
MODEL_DIR="$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1
TRAIN_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage1/train.jsonl
EVAL_PATH="$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage1/eval.jsonl
if [[ "${CLEAN_MODEL_DIR:-0}" == "1" ]]; then
  rm -rf "$MODEL_DIR"
fi
test -s "$TRAIN_PATH"
test -s "$EVAL_PATH"
mkdir -p "$MODEL_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29728
exec ${PYTHON_BIN:-/path/to/conda/env/bin/python} -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port 29728 \
  "$SEARCHSKILL_ROOT"/supervised_finetuning/scripts/train_lora.py \
  --model-path ${HF_MODELS:-/path/to/hf_models}/Qwen2.5-7B-Instruct \
  --train-path "$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage1/train.jsonl \
  --eval-path "$SEARCHSKILL_ROOT"/supervised_finetuning/data/stage1/eval.jsonl \
  --output-dir "$SEARCHSKILL_ROOT"/supervised_finetuning/models/stage1 \
  --max-length 8192 \
  --truncate-side left \
  --learning-rate 7e-5 \
  --num-train-epochs 2.0 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.05 \
  --logging-steps 5 \
  --save-steps 40 \
  --eval-steps 40 \
  --dataloader-num-workers 0 \
  --answer-loss-weight 2.5 \
  --search-loss-weight 0.8
