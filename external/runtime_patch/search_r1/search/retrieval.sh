#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-<rl_runtime_root>}"
DATA_NAME="${DATA_NAME:-nq}"
SPLIT="${SPLIT:-test}"
TOPK="${TOPK:-3}"

DATASET_PATH="${DATASET_PATH:-$ROOT/benchmarks/dev/${DATA_NAME}.jsonl}"
INDEX_PATH="${INDEX_PATH:-<retriever_index_dir>}"
CORPUS_PATH="${CORPUS_PATH:-<retriever_corpus_path>}"
MODEL_PATH="${MODEL_PATH:-<retriever_model_path>}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-<hf_home>}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python "$RUNTIME_ROOT/search_r1/search/retrieval.py" \
  --retrieval_method e5 \
  --retrieval_topk "${TOPK}" \
  --index_path "${INDEX_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --data_split "${SPLIT}" \
  --retrieval_model_path "${MODEL_PATH}" \
  --retrieval_pooling_method mean \
  --retrieval_batch_size 512 \
  --retrieval_query_max_length 256 \
  --retrieval_use_fp16
