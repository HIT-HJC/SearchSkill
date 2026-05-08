#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-searchskill}"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_CACHE:?Set HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

python outputs/qwen3_8b_hotpotqa_eval_20260323/qwen_skillbank_retrieval_server_fixed.py \
  --index_path "${E5_INDEX_PATH:?Set E5_INDEX_PATH}" \
  --corpus_path "${E5_CORPUS_PATH:?Set E5_CORPUS_PATH}" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "${HF_MODELS:?Set HF_MODELS}/e5-base-v2" \
  --port 8000 \
  --faiss_gpu
