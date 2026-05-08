# Reproducing SearchSkill

This document is the stable entry point for reproducing the released SearchSkill package after cloning the repository. It separates the reliable artifact-reuse path from the heavier full-regeneration path.

## 1. Clone And Environment

```bash
git clone https://github.com/HIT-HJC/SearchSkill.git "SearchSkill Code"
cd "SearchSkill Code"
export ROOT="$(pwd)"
export SEARCHSKILL_ROOT="$ROOT"
```

Create a Linux CUDA environment for training and evaluation. The released scripts were written for Python 3.9/3.10 on Linux with CUDA GPUs, PyTorch CUDA builds, HuggingFace Transformers/Datasets, pandas, pyarrow, Ray, vLLM, and the vendored VERL/Search-R1 runtime. Windows is fine for inspecting files, but SFT/RL launch scripts expect a Linux shell and CUDA runtime.

Minimum setup for data checks:

```bash
python -m pip install -r data_preparation/requirements.txt
```

Runtime setup for SFT/RL:

```bash
python -m pip install -r external/SearchR1/requirements.txt
python -m pip install -e external/SearchR1
export PYTHON_BIN="$(command -v python)"
```

## 2. External Assets

The repository does not include these assets:

- Base models: Qwen2.5 3B/7B base and instruct checkpoints, or equivalent paths supplied through `MODEL_PATH`.
- Trained SFT/RL checkpoints: generated under ignored `models/` or `runs/` directories.
- Retriever index and corpus: usually E5/BM25 index plus Wikipedia corpus, served through an HTTP `/retrieve` endpoint.
- Original full datasets: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, and MuSiQue mirrors.
- API credentials: only needed when regenerating SkillBank expansions or teacher trajectories.

Recommended model identifiers are `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B`, and `Qwen/Qwen2.5-7B-Instruct`. If you use local mirrors, keep the same directory names under `HF_MODELS`.

Set the common environment:

```bash
export HF_MODELS="/path/to/hf_models"
export HF_DATA="/path/to/hf_data"
export HF_CACHE="/path/to/hf_cache"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Set `OPENAI_API_KEY` only for stages that call a teacher or skill-expansion model.

Common variables:

| Variable | Required for | Default/meaning |
| --- | --- | --- |
| `SEARCHSKILL_ROOT`, `ROOT` | all shell wrappers | repository root |
| `PYTHON_BIN`, `PY` | scripts | current `python` or an explicit environment binary |
| `HF_MODELS` | SFT/RL/eval | base model mirror root |
| `HF_DATA`, `HF_CACHE` | data prep | dataset JSONL and HuggingFace cache roots |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | SkillBank and teacher regeneration | API key plus responses-compatible base URL, with or without `/v1` |
| `RETRIEVER_HOST`, `RETRIEVER_PORT` | teacher/RL/eval | HTTP retriever endpoint, default `127.0.0.1:8000` |
| `MODEL_PATH` | SFT merge/RL/eval | merged checkpoint or dense checkpoint |
| `DATA_DIR`, `RUN_NAME`, `OUT_DIR`, `LOG_DIR` | RL | override wrapper defaults |
| `CUDA_VISIBLE_DEVICES`, `GPUS`, `RAY_NUM_CPUS`, `RAY_NUM_GPUS`, `RAY_TMPDIR` | RL | hardware and Ray placement controls |
| `SHARD_COUNT`, `GPU_IDS_CSV`, `EVAL_ROOT`, `SLURM_JOB_ID_TARGET` | evaluation | shard count, GPU list, output root, optional Slurm job |

## 3. Artifact-Reuse Path

This is the recommended path for a first clone. It avoids expensive API calls and starts from included artifacts.

1. Verify included artifacts:

```bash
python - <<'PY'
from pathlib import Path
required = [
    "data_preparation/samples",
    "skill_bank/round_4_musique/outputs/final_skill_bank.md",
    "teacher_trajectory/runs/canonical_teacher_set",
    "supervised_finetuning/data/stage2/train.jsonl",
    "reinforcement_learning/data/policy_7b_instruct/train.parquet",
    "external/SearchR1/verl/trainer/main_ppo_searchskill.py",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"missing artifacts: {missing}")
print("artifact reuse path is ready")
PY
```

2. Train SFT models if you need checkpoints:

```bash
bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh
python supervised_finetuning/scripts/merge_lora.py \
  --base-model-path "$HF_MODELS/Qwen2.5-7B-Instruct" \
  --adapter-path supervised_finetuning/models/stage2 \
  --output-dir supervised_finetuning/models/stage2_7b_instruct_merged \
  --device cuda \
  --overwrite
```

3. Launch RL from a merged SFT checkpoint:

```bash
export MODEL_PATH="$SEARCHSKILL_ROOT/supervised_finetuning/models/stage2_7b_instruct_merged"
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

The backbone wrappers rebuild or refresh their matching parquet directory before training, for example `data/policy_7b_instruct`. To train from a custom prebuilt parquet directory, call `scripts/launch_training.sh` directly with `DATA_DIR`.

4. Evaluate a checkpoint:

```bash
export MODEL_PATH="/path/to/checkpoint_or_merged_model"
bash reinforcement_learning/scripts/evaluate_policy.sh
```

## 4. Full-Regeneration Path

Use this path only when you want to rebuild every intermediate artifact.

1. Rebuild sampled data:

```bash
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh
```

2. Rebuild SkillBank rounds:

```bash
python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py --base-url "$OPENAI_BASE_URL"
```

3. Rebuild teacher trajectories:

Follow the runnable minimal recipe in `teacher_trajectory/REPRODUCE.md`. The checked-in `bin/*.sh` scripts show the intended launch pattern, but cluster-specific Slurm options should be adjusted for your machines.

4. Rebuild SFT data:

```bash
python supervised_finetuning/scripts/build_stage1_dataset.py \
  --input-path teacher_trajectory/runs/canonical_teacher_set/all/trajectories.filtered.jsonl \
  --output-dir supervised_finetuning/data/stage1

python supervised_finetuning/scripts/build_stage2_dataset.py \
  --input-train supervised_finetuning/data/stage1/train.jsonl \
  --input-eval supervised_finetuning/data/stage1/eval.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir supervised_finetuning/data/stage2
```

5. Rebuild RL data:

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

6. Train and evaluate as in the artifact-reuse path.

## 5. Stability Checklist

Before publishing new changes or asking another user to reproduce:

```bash
git status --short
git grep -n -I -E '(/your/private/path|your-internal-user|your-internal-host|[A-Z]:\\Users\\|sk-[A-Za-z0-9_-]{20,}|ghp_|hf_[A-Za-z0-9]{20,})' -- .
python - <<'PY'
import json
from pathlib import Path
for path in Path(".").rglob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
for path in Path(".").rglob("*.jsonl"):
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, 1):
            if line.strip():
                json.loads(line)
print("json/jsonl validation passed")
PY
```

Replace the placeholder scanner terms with private path, username, hostname, and job-label patterns from your own environment. The grep command should return no private local paths or secrets before release.
