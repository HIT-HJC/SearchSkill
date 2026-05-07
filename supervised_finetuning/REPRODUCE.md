# Supervised Fine-Tuning

This module contains the two-stage SFT setup used before reinforcement learning.

## Data

- `data/stage1/`: stage-one skill-context training and evaluation messages.
- `data/stage2/`: stage-two two-phase SkillBank training and evaluation messages.

## Main Scripts

- `scripts/build_stage1_dataset.py`: builds stage-one data from canonical teacher trajectories.
- `scripts/build_stage2_dataset.py`: repacks stage-one data into the two-phase SkillBank protocol.
- `scripts/train_lora.py`: LoRA SFT trainer.
- `scripts/merge_lora.py`: merges LoRA adapters into deployable checkpoints.
- `scripts/train_stage1_*.sh`: stage-one training wrappers for the four target backbones.
- `scripts/train_stage2_*.sh`: stage-two training wrappers for the four target backbones.

## Expected Model Outputs

The scripts write to `supervised_finetuning/models/`, which is ignored by git. Downstream RL scripts expect merged stage-two checkpoints such as:

- `supervised_finetuning/models/stage2_3b_base_merged`
- `supervised_finetuning/models/stage2_3b_instruct_merged`
- `supervised_finetuning/models/stage2_7b_base_merged`
- `supervised_finetuning/models/stage2_7b_instruct_merged`

## Replace Before Running

Set `ROOT`, `SEARCHSKILL_ROOT`, `PYTHON_BIN`, `HF_MODELS`, and `HF_CACHE`. The scripts default to Qwen 2.5 3B/7B base and instruct model names under `HF_MODELS`.
