#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate searchr1

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_datasets_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache

python outputs/qwen3_8b_hotpotqa_eval_20260323/qwen_skillbank_retrieval_server_fixed.py \
  --index_path /path/to/e5_data/e5_Flat.index \
  --corpus_path /path/to/e5_data/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model ${HF_MODELS:-/path/to/hf_models}/e5-base-v2 \
  --port 8000 \
  --faiss_gpu
