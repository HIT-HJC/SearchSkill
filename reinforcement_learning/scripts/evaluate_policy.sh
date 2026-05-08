#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/SearchSkill Code}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN:-/path/to/conda/env/bin/python}}"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-1313822}"
SCRIPT="$ROOT/skill_bank/nq_eval/eval_qwen_skillbank_v3.py"
MERGE_SCRIPT="$ROOT/eval/hotpot_sft_full_v1_toolstar200/merge_hotpot_shards.py"
SKILL_BANK_PATH="${SKILL_BANK_PATH:-$ROOT/skill_bank/round_4_musique/outputs/final_skill_bank.md}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_NAME="${RUN_NAME:-policy_eval_$(date +%m%d_%H%M)}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/eval/$RUN_NAME}"

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
PRINT_EVERY="${PRINT_EVERY:-10}"
SHARD_COUNT="${SHARD_COUNT:-4}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
MAX_SAMPLES_MULTI="${MAX_SAMPLES_MULTI:-}"
MAX_SAMPLES_SINGLE="${MAX_SAMPLES_SINGLE:-}"
DATASET_ARG="${1:-all}"

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -f "$ROOT/RL_v4/runs/latest_run_path.txt" ]]; then
    LATEST_RUN="$(cat "$ROOT/RL_v4/runs/latest_run_path.txt")"
    MODEL_PATH="$LATEST_RUN/checkpoint-step0120"
  else
    echo "MODEL_PATH is empty and latest_run_path.txt is missing" >&2
    exit 2
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS[@]}" -ne "$SHARD_COUNT" ]]; then
  echo "GPU_IDS_CSV count (${#GPU_IDS[@]}) must match SHARD_COUNT ($SHARD_COUNT)" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" && ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Missing dense model checkpoint under MODEL_PATH=$MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$EVAL_ROOT"
cd "$ROOT"

preflight_retriever() {
  srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=2 --cpu-bind=none \
    env RETRIEVER_HOST="$RETRIEVER_HOST" RETRIEVER_PORT="$RETRIEVER_PORT" \
    "$PYTHON_BIN" -c 'import os, requests; s=requests.Session(); s.trust_env=False; host=os.environ["RETRIEVER_HOST"]; port=os.environ["RETRIEVER_PORT"]; r=s.post(f"http://{host}:{port}/retrieve", json={"queries":["Barack Obama birthplace"], "topk":1, "return_scores":True}, timeout=30); r.raise_for_status(); print(f"retriever_ok host={host} status={r.status_code}")'
}

data_src_for() {
  case "$1" in
    hotpotqa) echo "$ROOT/eval/hotpot_toolstar_stage2_final_skill_bank_gpu028_4gpu/data/hotpotqa_toolstar_test200.jsonl" ;;
    2wiki) echo "$ROOT/eval/2wiki_toolstar_stage2_final_skill_bank_gpu028_4gpu/data/2wiki_toolstar_test200.jsonl" ;;
    musique) echo "$ROOT/eval/musique_toolstar_stage1_skillctx_two_stage_strict_gpu028_4gpu/data/musique_toolstar_test200.jsonl" ;;
    bamboogle) echo "$ROOT/eval/bamboogle_toolstar_stage1_skillctx_two_stage_strict_gpu028_4gpu/data/bamboogle_toolstar_test125.jsonl" ;;
    nq) echo "${HF_DATA:-/path/to/hf_data}/data/nq/test.jsonl" ;;
    triviaqa) echo "${HF_DATA:-/path/to/hf_data}/data/triviaqa/test.jsonl" ;;
    popqa) echo "${HF_DATA:-/path/to/hf_data}/data/popqa/test.jsonl" ;;
    *) return 1 ;;
  esac
}

max_samples_for() {
  case "$1" in
    hotpotqa|2wiki|musique|bamboogle) echo "$MAX_SAMPLES_MULTI" ;;
    *) echo "$MAX_SAMPLES_SINGLE" ;;
  esac
}

run_one_dataset() {
  local dataset_tag="$1"
  local data_src
  data_src="$(data_src_for "$dataset_tag")"
  if [[ ! -f "$data_src" ]]; then
    echo "Missing data source for $dataset_tag: $data_src" >&2
    exit 2
  fi

  local root="$EVAL_ROOT/$dataset_tag"
  local data_path="$root/data/${dataset_tag}.jsonl"
  local max_samples
  max_samples="$(max_samples_for "$dataset_tag")"
  echo "=== Running $dataset_tag -> $root ==="
  rm -rf "$root/shards" "$root/results" "$root/merged"
  mkdir -p "$root/data" "$root/shards" "$root/results" "$root/merged"
  cp -f "$data_src" "$data_path"

  DATA_PATH="$data_path" SHARD_DIR="$root/shards" SHARD_COUNT="$SHARD_COUNT" MAX_SAMPLES="$max_samples" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

data_path = Path(os.environ["DATA_PATH"])
shard_dir = Path(os.environ["SHARD_DIR"])
shard_count = int(os.environ["SHARD_COUNT"])
max_samples_text = os.environ.get("MAX_SAMPLES", "").strip()

rows = []
with data_path.open("r", encoding="utf-8") as handle:
    for global_idx, line in enumerate(handle):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row["_source_global_idx"] = global_idx
        rows.append(row)
if max_samples_text:
    rows = rows[: int(max_samples_text)]

base = len(rows) // shard_count
remainder = len(rows) % shard_count
start = 0
for shard_id in range(shard_count):
    shard_size = base + (1 if shard_id < remainder else 0)
    shard_rows = rows[start : start + shard_size]
    start += shard_size
    out_path = shard_dir / f"shard_{shard_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for row in shard_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(shard_rows)} rows to {out_path}", flush=True)
PY

  local pids=()
  for shard_id in $(seq 0 $((SHARD_COUNT - 1))); do
    local shard_path="$root/shards/shard_${shard_id}.jsonl"
    local shard_root="$root/results/shard_${shard_id}"
    local gpu_id="${GPU_IDS[$shard_id]}"
    mkdir -p "$shard_root"
    nohup srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
      env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
      "$PYTHON_BIN" "$SCRIPT" \
        --data-path "$shard_path" \
        --dataset-tag "$dataset_tag" \
        --skill-bank-path "$SKILL_BANK_PATH" \
        --model-path "$MODEL_PATH" \
        --out-jsonl "$shard_root/trace.jsonl" \
        --out-json "$shard_root/trace.json" \
        --summary-json "$shard_root/summary.json" \
        --log-file "$shard_root/run.log" \
        --retriever-host "$RETRIEVER_HOST" \
        --retriever-port "$RETRIEVER_PORT" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --disable-thinking \
        --dtype "$DTYPE" \
        --print-every "$PRINT_EVERY" \
        --strict-em-only \
        --skill-context-mode ids \
        --trust-remote-code \
      > "$shard_root/launcher.out" 2>&1 < /dev/null &
    pids+=("$!")
    echo "launched ${dataset_tag} shard_${shard_id} on CUDA_VISIBLE_DEVICES=${gpu_id}"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one shard failed for $dataset_tag" >&2
    exit 3
  fi

  "$PYTHON_BIN" "$MERGE_SCRIPT" \
    --input-dir "$root/results" \
    --output-jsonl "$root/merged/${dataset_tag}_trace.jsonl" \
    --output-json "$root/merged/${dataset_tag}_trace.json" \
    --summary-json "$root/merged/${dataset_tag}_summary.json" \
    --source-data-path "$data_path"

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
summary = json.loads(Path("$root/merged/${dataset_tag}_summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "dataset": "$dataset_tag",
    "n_examples": summary.get("n_examples"),
    "n_correct": summary.get("n_correct"),
    "em": summary.get("em"),
    "elapsed_wall_seconds": summary.get("elapsed_wall_seconds"),
    "model_path": "$MODEL_PATH",
}, ensure_ascii=False, indent=2), flush=True)
PY
}

case "$DATASET_ARG" in
  all) DATASETS=(hotpotqa 2wiki musique bamboogle nq triviaqa popqa) ;;
  multihop) DATASETS=(hotpotqa 2wiki musique bamboogle) ;;
  singlehop) DATASETS=(nq triviaqa popqa) ;;
  *) DATASETS=("$DATASET_ARG") ;;
esac

preflight_retriever
for dataset in "${DATASETS[@]}"; do
  run_one_dataset "$dataset"
done

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
root = Path("$EVAL_ROOT")
rows = []
for path in sorted(root.glob("*/merged/*_summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "dataset": path.parent.parent.name,
        "n_examples": payload.get("n_examples"),
        "n_correct": payload.get("n_correct"),
        "em": payload.get("em"),
        "summary_path": str(path),
    })
out = root / "all_summary.json"
out.write_text(json.dumps({"model_path": "$MODEL_PATH", "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"all_summary": str(out), "results": rows}, ensure_ascii=False, indent=2), flush=True)
PY
