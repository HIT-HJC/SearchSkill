# SearchSkill Code

This package contains the cleaned release version of the SearchSkill experimental pipeline. It is organized around the method stages rather than internal trial numbers, so a new user can follow the project from data preparation and SkillBank construction to supervised fine-tuning, reinforcement learning, evaluation, and analysis.

## Layout

- `data_preparation/`: dataset profiling, sampling, group annotation, and sampled training pools.
- `skill_bank/`: seed-to-final SkillBank construction and evaluation helpers.
- `teacher_trajectory/`: teacher rollout, trajectory filtering, and SFT packing inputs.
- `supervised_finetuning/`: two-stage supervised fine-tuning data and launch scripts.
- `reinforcement_learning/`: final policy optimization data, launch scripts, and analysis outputs.
- `benchmarks/`: benchmark subset manifests and sampling utilities.
- `analysis_scripts/`: shared analysis scripts used after evaluation.
- `external/SearchR1/`: vendored Search-R1/VERL checkout with SearchSkill changes already applied.
- `external/SearchR1_patch/`: audit copy of the SearchSkill-specific Search-R1 files.

## Reproduction Flow

1. Prepare and inspect sampled data under `data_preparation/`.
2. Build or reuse the final SkillBank from `skill_bank/round_4_musique/outputs/final_skill_bank.md`.
3. Build teacher trajectories with `teacher_trajectory/`, or reuse the included canonical and coverage trajectories.
4. Train stage-one SFT models with `supervised_finetuning/scripts/train_stage1_*.sh`.
5. Train stage-two SFT models with `supervised_finetuning/scripts/train_stage2_*.sh`.
6. Merge the stage-two LoRA checkpoints with `supervised_finetuning/scripts/merge_lora.py`.
7. Run policy optimization with `reinforcement_learning/scripts/train_*.sh`.
8. Evaluate and analyze with `reinforcement_learning/scripts/evaluate_policy.sh` and `reinforcement_learning/analysis/`.

## Environment Variables To Set

```bash
export ROOT="/path/to/SearchSkill Code"
export SEARCHSKILL_ROOT="$ROOT"
export PYTHON_BIN="/path/to/conda/env/bin/python"
export HF_MODELS="/path/to/hf_models"
export HF_DATA="/path/to/hf_data"
export HF_CACHE="/path/to/hf_cache"
export OPENAI_API_KEY="your_key_if_teacher_generation_or_skill_expansion_is_used"
export RETRIEVER_HOST="your_retriever_host"
export RETRIEVER_PORT="8000"
```

Because the package directory contains a space, always quote `$ROOT` and derived paths in shell commands.

## Assets Not Included

Base model weights, trained checkpoints, live retrieval indexes, private API keys, logs, Ray state, and cache directories are not included. The code expects users to provide model paths through `HF_MODELS`, `MODEL_PATH`, or the model-specific launch scripts.
