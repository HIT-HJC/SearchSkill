# Reinforcement Learning

This stage optimizes the SearchSkill policy after SFT. The repository includes RL source JSONL and parquet data, but not trained checkpoints or run logs.

## Inputs

- `MODEL_PATH`: merged SFT checkpoint or another dense checkpoint.
- `RUNTIME_ROOT`: path to your RL runtime with `external/runtime_patch/` applied.
- `RETRIEVER_HOST` / `RETRIEVER_PORT`: retrieval server endpoint.
- CUDA GPUs for training.

## Data

- `source_data/policy_training_pool/`: train/dev JSONL source data.
- `data/policy_*`: parquet files used by the released training wrappers.

Rebuild parquet data:

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

## Train

```bash
export MODEL_PATH="<merged_sft_checkpoint>"
export RUNTIME_ROOT="<rl_runtime_root>"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

All wrapper defaults can be overridden with environment variables such as `DATA_DIR`, `RUN_NAME`, `OUT_DIR`, `GPUS`, and `CUDA_VISIBLE_DEVICES`.

## Test

```bash
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh nq
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=full bash reinforcement_learning/scripts/evaluate_policy.sh all
```
