#!/usr/bin/env bash
set -euo pipefail

source /online1/public/support/amd/Ananconda3/2023.3/etc/profile.d/conda.sh
conda activate searchr1

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/online1/ycsc_chenkh/hitici_11/.cache/hf_searchskill
export HF_DATASETS_CACHE=/online1/ycsc_chenkh/hitici_11/.cache/hf_datasets
export TRANSFORMERS_CACHE=/online1/ycsc_chenkh/hitici_11/.cache/hf_searchskill

python /online1/ycsc_chenkh/hitici_11/SearchSkill/qwen3_8b_hotpotqa_eval_20260323/qwen_skillbank_retrieval_server_fixed.py \
  --index_path /online1/ycsc_chenkh/hitici_11/e5_data/e5_Flat.index \
  --corpus_path /online1/ycsc_chenkh/hitici_11/e5_data/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model ${HF_MODELS:-/path/to/hf_models}/e5-base-v2 \
  --port 8000 \
  --faiss_gpu
