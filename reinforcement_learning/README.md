# Reinforcement Learning

This module contains the final SearchSkill policy optimization setup.

## Data

- `source_data/policy_training_pool/`: train/dev JSONL source data for policy optimization.
- `data/policy_3b_base/`: parquet data for the 3B base policy run.
- `data/policy_3b_instruct/`: parquet data for the 3B instruct policy run.
- `data/policy_7b_base/`: parquet data for the 7B base policy run.
- `data/policy_7b_instruct/`: parquet data for the 7B instruct policy run.

## Main Scripts

- `scripts/build_policy_dataset.py`: builds VERL-compatible parquet files.
- `scripts/launch_training.sh`: canonical RL launch script.
- `scripts/train_3b_base.sh`, `scripts/train_3b_instruct.sh`, `scripts/train_7b_base.sh`, `scripts/train_7b_instruct.sh`: model-specific wrappers.
- `scripts/evaluate_policy.sh`: evaluation helper.
- `scripts/check_reward_alignment.py`: reward sanity check.

## External Code

`external/SearchR1/` is included and already contains the required SearchSkill trainer and reward code. The launch script defaults to this checkout through `SEARCHR1_ROOT`.

## Replace Before Running

Set `ROOT`, `SEARCHSKILL_ROOT`, `PYTHON_BIN`, `HF_MODELS`, `MODEL_PATH` if overriding defaults, `RETRIEVER_HOST`, and `RETRIEVER_PORT`. Policy runs require a live retriever endpoint.
