# Reinforcement Learning

This stage runs final SearchSkill policy optimization using the vendored runtime in `external/SearchR1/`. The repository includes RL source JSONL and parquet data, but not trained RL checkpoints or run logs.

## Included Data

- `source_data/policy_training_pool/`: train/dev JSONL source data.
- `data/policy_3b_base/`: parquet data for the 3B base policy.
- `data/policy_3b_instruct/`: parquet data for the 3B instruct policy.
- `data/policy_7b_base/`: parquet data for the 7B base policy.
- `data/policy_7b_instruct/`: parquet data for the 7B instruct policy.
- `analysis/`: released reward and policy-comparison summaries.

## Scripts

- `scripts/build_policy_dataset.py`: builds VERL-compatible parquet files.
- `scripts/launch_training.sh`: canonical configurable RL launch script.
- `scripts/train_3b_base.sh`, `scripts/train_3b_instruct.sh`, `scripts/train_7b_base.sh`, `scripts/train_7b_instruct.sh`: backbone-specific wrappers.
- `scripts/evaluate_policy.sh`: evaluation helper.
- `scripts/check_reward_alignment.py`: reward sanity check.
- `scripts/check_gpu.sh`: simple CUDA visibility check.

## Required Runtime

`external/SearchR1/` already contains the SearchSkill trainer and reward implementation. Install it in your Python environment:

```bash
python -m pip install -r external/SearchR1/requirements.txt
python -m pip install -e external/SearchR1
```

RL requires a merged SFT checkpoint through `MODEL_PATH`, a live retriever endpoint, and multi-GPU CUDA resources.

## Reuse Path

Use checked-in parquet data:

```bash
test -s reinforcement_learning/data/policy_7b_instruct/train.parquet
```

The backbone wrappers use their matching checked-in parquet directory, then rebuild or refresh it before training. Launch with:

```bash
export MODEL_PATH="$SEARCHSKILL_ROOT/supervised_finetuning/models/stage2_7b_instruct_merged"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

To train from an already prepared custom parquet directory without using a backbone wrapper, call the configurable launcher directly:

```bash
MODEL_PATH="/path/to/model" \
DATA_DIR="reinforcement_learning/data/policy_data" \
RUN_NAME="my_searchskill_rl_run" \
bash reinforcement_learning/scripts/launch_training.sh
```

## Regeneration Path

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

Then launch `scripts/launch_training.sh` or a backbone-specific wrapper. Wrapper defaults are overrideable with environment variables such as `MODEL_PATH`, `DATA_DIR`, `RUN_NAME`, `OUT_DIR`, `GPUS`, and `CUDA_VISIBLE_DEVICES`.
