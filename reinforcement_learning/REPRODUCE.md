# Step 5: RL And Test

For the full flow, start with `../REPRODUCE.md`.

## Prepare

```bash
export ROOT="$(pwd)"
export SEARCHSKILL_ROOT="$ROOT"
export PYTHON_BIN="$(command -v python)"
export RUNTIME_ROOT="<rl_runtime_root>"
export MODEL_PATH="<merged_sft_checkpoint>"
export RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
```

Copy `external/runtime_patch/` into `RUNTIME_ROOT` before training.

## Build Data

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

## Train

```bash
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

## Test

```bash
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh nq
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=full bash reinforcement_learning/scripts/evaluate_policy.sh all
```
