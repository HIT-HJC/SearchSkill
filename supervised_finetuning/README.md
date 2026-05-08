# Supervised Fine-Tuning

This stage trains the two-stage SearchSkill policy before reinforcement learning. The repository includes the SFT datasets, but not the trained LoRA adapters or merged model checkpoints.

## Included Data

- `data/stage1/`: stage-one skill-context training and evaluation messages.
- `data/stage2/`: stage-two two-phase final-SkillBank training and evaluation messages.
- `plots/`: released SFT loss plots.

## Scripts

- `scripts/build_stage1_dataset.py`: builds stage-one data from canonical teacher trajectories.
- `scripts/build_stage2_dataset.py`: repacks stage-one data into the two-phase final-SkillBank protocol.
- `scripts/train_lora.py`: LoRA SFT trainer.
- `scripts/merge_lora.py`: merges LoRA adapters into deployable checkpoints.
- `scripts/train_stage1_*.sh`: stage-one wrappers for Qwen2.5 3B/7B base and instruct backbones.
- `scripts/train_stage2_*.sh`: stage-two wrappers for the same backbones.

## Reuse Path

Use the included stage-two data directly:

```bash
test -s supervised_finetuning/data/stage2/train.jsonl
test -s supervised_finetuning/data/stage2/eval.jsonl
```

To continue to RL, you must train or provide merged SFT checkpoints. Expected output directories are ignored by git:

- `supervised_finetuning/models/stage2_3b_base_merged`
- `supervised_finetuning/models/stage2_3b_instruct_merged`
- `supervised_finetuning/models/stage2_7b_base_merged`
- `supervised_finetuning/models/stage2_7b_instruct_merged`

## Regeneration Path

Rebuild SFT data:

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

Train and merge one backbone, for example:

```bash
bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh
python supervised_finetuning/scripts/merge_lora.py --help
```

Set `HF_MODELS`, `HF_CACHE`, `PYTHON_BIN`, and GPU-related launch settings before training.
