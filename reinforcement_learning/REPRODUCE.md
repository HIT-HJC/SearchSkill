# Reproducing Reinforcement Learning

For the full project flow, start with `../REPRODUCE.md`. This file only covers RL.

## Prerequisites

- A merged SFT checkpoint, passed as `MODEL_PATH`.
- A live retriever endpoint at `RETRIEVER_HOST:RETRIEVER_PORT`.
- CUDA GPUs. The default wrappers are configured for multi-GPU runs.
- The vendored runtime installed with `python -m pip install -e external/SearchR1`.

## Check Environment

```bash
export SEARCHSKILL_ROOT="/path/to/SearchSkill Code"
export ROOT="$SEARCHSKILL_ROOT"
export PYTHON_BIN="$(command -v python)"
export MODEL_PATH="/path/to/merged_sft_checkpoint"
export RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"

bash reinforcement_learning/scripts/check_gpu.sh
```

## Build Or Reuse RL Data

The repository includes parquet data for the released runs. To rebuild:

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

Backbone wrappers use these default data directories:

| Wrapper | Default `DATA_DIR` |
| --- | --- |
| `train_3b_base.sh` | `reinforcement_learning/data/policy_3b_base` |
| `train_3b_instruct.sh` | `reinforcement_learning/data/policy_3b_instruct` |
| `train_7b_base.sh` | `reinforcement_learning/data/policy_7b_base` |
| `train_7b_instruct.sh` | `reinforcement_learning/data/policy_7b_instruct` |

## Train

Use a wrapper for the target backbone:

```bash
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

Wrappers provide defaults but respect existing environment values for `MODEL_PATH`, `DATA_DIR`, `RUN_NAME`, `OUT_DIR`, `GPUS`, `CUDA_VISIBLE_DEVICES`, and batch-size controls.

Or customize:

```bash
MODEL_PATH="/path/to/model" \
DATA_DIR="reinforcement_learning/data/policy_data" \
RUN_NAME="my_searchskill_rl_run" \
bash reinforcement_learning/scripts/launch_training.sh
```

Outputs go under ignored `reinforcement_learning/runs/` and `reinforcement_learning/logs/`.

## Evaluate

```bash
MODEL_PATH="/path/to/checkpoint_or_merged_model" \
bash reinforcement_learning/scripts/evaluate_policy.sh
```

The evaluator expects benchmark data and a working retriever. The default dataset paths point to checked-in files under `benchmarks/`.

Examples:

```bash
MODEL_PATH="/path/to/checkpoint_or_merged_model" bash reinforcement_learning/scripts/evaluate_policy.sh singlehop
MODEL_PATH="/path/to/checkpoint_or_merged_model" bash reinforcement_learning/scripts/evaluate_policy.sh hotpotqa
MODEL_PATH="/path/to/checkpoint_or_merged_model" SHARD_COUNT=2 GPU_IDS_CSV=0,1 bash reinforcement_learning/scripts/evaluate_policy.sh all
```

If `MODEL_PATH` is omitted, the evaluator reads `reinforcement_learning/runs/latest_run_path.txt` when present.
