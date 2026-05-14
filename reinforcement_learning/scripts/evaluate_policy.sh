#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
export SEARCHSKILL_ROOT="${SEARCHSKILL_ROOT:-$ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_JOB_ID_TARGET="${SLURM_JOB_ID_TARGET:-}"
SCRIPT="$ROOT/skill_bank/nq_eval/eval_nq_qwen_skillbank.py"
SKILL_BANK_PATH="${SKILL_BANK_PATH:-$ROOT/skill_bank/round_4_musique/outputs/final_skill_bank.md}"
MODEL_PATH="${MODEL_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_NAME="${RUN_NAME:-policy_eval_$(date +%m%d_%H%M)}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/eval/$RUN_NAME}"

RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
PRINT_EVERY="${PRINT_EVERY:-10}"
SHARD_COUNT="${SHARD_COUNT:-1}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0}"
MAX_SAMPLES_MULTI="${MAX_SAMPLES_MULTI:-}"
MAX_SAMPLES_SINGLE="${MAX_SAMPLES_SINGLE:-}"
DATASET_ARG="${1:-all}"
BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-dev}"

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -f "$ROOT/reinforcement_learning/runs/latest_run_path.txt" ]]; then
    MODEL_PATH="$(cat "$ROOT/reinforcement_learning/runs/latest_run_path.txt")"
  else
    echo "MODEL_PATH is empty and reinforcement_learning/runs/latest_run_path.txt is missing" >&2
    exit 2
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS[@]}" -ne "$SHARD_COUNT" ]]; then
  echo "GPU_IDS_CSV count (${#GPU_IDS[@]}) must match SHARD_COUNT ($SHARD_COUNT)" >&2
  exit 1
fi

if [[ -d "$MODEL_PATH" && ! -f "$MODEL_PATH/model.safetensors.index.json" && ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Missing dense model checkpoint under MODEL_PATH=$MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$EVAL_ROOT"
cd "$ROOT"

preflight_retriever() {
  if [[ -n "$SLURM_JOB_ID_TARGET" ]] && command -v srun >/dev/null 2>&1; then
    srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=2 --cpu-bind=none \
      env RETRIEVER_HOST="$RETRIEVER_HOST" RETRIEVER_PORT="$RETRIEVER_PORT" \
      "$PYTHON_BIN" -c 'import os, requests; s=requests.Session(); s.trust_env=False; host=os.environ["RETRIEVER_HOST"]; port=os.environ["RETRIEVER_PORT"]; r=s.post(f"http://{host}:{port}/retrieve", json={"queries":["Barack Obama birthplace"], "topk":1, "return_scores":True}, timeout=30); r.raise_for_status(); print(f"retriever_ok host={host} status={r.status_code}")'
  else
    env RETRIEVER_HOST="$RETRIEVER_HOST" RETRIEVER_PORT="$RETRIEVER_PORT" \
      "$PYTHON_BIN" -c 'import os, requests; s=requests.Session(); s.trust_env=False; host=os.environ["RETRIEVER_HOST"]; port=os.environ["RETRIEVER_PORT"]; r=s.post(f"http://{host}:{port}/retrieve", json={"queries":["Barack Obama birthplace"], "topk":1, "return_scores":True}, timeout=30); r.raise_for_status(); print(f"retriever_ok host={host} status={r.status_code}")'
  fi
}

data_src_for() {
  case "$1" in
    hotpotqa) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/hotpotqa.jsonl" ;;
    2wiki) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/2wiki.jsonl" ;;
    musique) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/musique.jsonl" ;;
    bamboogle) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/bamboogle.jsonl" ;;
    nq) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/nq.jsonl" ;;
    triviaqa) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/triviaqa.jsonl" ;;
    popqa) echo "$ROOT/benchmarks/$BENCHMARK_SPLIT/popqa.jsonl" ;;
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
    local launch_prefix=()
    if [[ -n "$SLURM_JOB_ID_TARGET" ]] && command -v srun >/dev/null 2>&1; then
      launch_prefix=(srun --jobid "$SLURM_JOB_ID_TARGET" --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=none)
    fi
    local adapter_args=()
    if [[ -n "$ADAPTER_PATH" ]]; then
      adapter_args=(--adapter-path "$ADAPTER_PATH")
    fi
    nohup "${launch_prefix[@]}" env CUDA_VISIBLE_DEVICES="$gpu_id" TOKENIZERS_PARALLELISM=false \
        "$PYTHON_BIN" "$SCRIPT" \
        --data-path "$shard_path" \
        --dataset-tag "$dataset_tag" \
        --skill-bank-path "$SKILL_BANK_PATH" \
        --model-path "$MODEL_PATH" \
        "${adapter_args[@]}" \
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

  RESULT_ROOT="$root/results" DATA_PATH="$data_path" DATASET_TAG="$dataset_tag" MERGED_ROOT="$root/merged" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
data_path = Path(os.environ["DATA_PATH"])
dataset_tag = os.environ["DATASET_TAG"]
merged_root = Path(os.environ["MERGED_ROOT"])
merged_root.mkdir(parents=True, exist_ok=True)

records = []
summaries = []
for shard_dir in sorted(result_root.glob("shard_*")):
    trace_jsonl = shard_dir / "trace.jsonl"
    if trace_jsonl.exists():
        with trace_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    summary_path = shard_dir / "summary.json"
    if summary_path.exists():
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

out_jsonl = merged_root / f"{dataset_tag}_trace.jsonl"
with out_jsonl.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

(merged_root / f"{dataset_tag}_trace.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
n_examples = sum(int(item.get("n_examples", 0)) for item in summaries) or len(records)
n_correct = sum(int(item.get("n_correct", 0)) for item in summaries)
summary = {
    "dataset": dataset_tag,
    "data_path": str(data_path),
    "n_examples": n_examples,
    "n_correct": n_correct,
    "em": n_correct / max(1, n_examples),
    "shard_summaries": summaries,
}
(merged_root / f"{dataset_tag}_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

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
