#!/usr/bin/env bash
set -euo pipefail

# ===== 路径都放到 online1（共享盘） =====
file_path=~/online1/e5_data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl

# 本地 e5 模型目录（登录节点已预下载）
retriever_name=e5
retriever_path=~/online1/hf_models/e5-base-v2

# 可选：把 HF/Transformers 缓存指到共享盘（即使计算节点离线也没关系）
export HF_HOME=~/online1/.cache/huggingface
export TRANSFORMERS_CACHE=~/online1/.cache/huggingface/transformers
mkdir -p "$TRANSFORMERS_CACHE"

# 检查必需文件
for f in "$index_file" "$corpus_file"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    exit 1
  fi
done
if [[ ! -d "$retriever_path" ]]; then
  echo "ERROR: missing local model dir $retriever_path"
  exit 1
fi

# 启动本地检索服务（FAISS 用 GPU）
python search_r1/search/retrieval_server.py \
  --index_path "$index_file" \
  --corpus_path "$corpus_file" \
  --topk 3 \
  --retriever_name "$retriever_name" \
  --retriever_model "$retriever_path" \
  --faiss_gpu


