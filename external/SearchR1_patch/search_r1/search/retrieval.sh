#!/usr/bin/env bash
set -euo pipefail

DATA_NAME=nq_search
SPLIT=test
TOPK=3

# —— 你的共享盘路径 ——
DATASET_PATH=/online1/ycsc_chenk/hitici_11/SearchR1/Search-R1/data/${DATA_NAME}
INDEX_PATH=/online1/ycsc_chenk/hitici_11/e5_data            # 内含 e5_Flat.index
CORPUS_PATH=/online1/ycsc_chenk/hitici_11/e5_data/wiki-18jsonl
MODEL_PATH=/online1/ycsc_chenk/hitici_11/hf_models/e5-base-v2

# 离线 & 缓存
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/online1/ycsc_chenk/hitici_11/hf_home
export CUDA_VISIBLE_DEVICES=0,1

python /path/to/Search-R1/search_r1/search retrieval.py \
  --retrieval_method e5 \
  --retrieval_topk ${TOPK} \
  --index_path ${INDEX_PATH} \
  --corpus_path ${CORPUS_PATH} \
  --dataset_path ${DATASET_PATH} \
  --data_split ${SPLIT} \
  --retrieval_model_path ${MODEL_PATH} \
  --retrieval_pooling_method mean \
  --retrieval_batch_size 512 \
  --retrieval_query_max_length 256 \
  --retrieval_use_fp16

