# Reproducing Supervised Fine-Tuning

For the full project flow, start with `../REPRODUCE.md`. This file only covers SFT.

## Stable Path

The SFT datasets are checked in:

```bash
python - <<'PY'
from pathlib import Path
for p in [
    "supervised_finetuning/data/stage1/train.jsonl",
    "supervised_finetuning/data/stage2/train.jsonl",
]:
    if not Path(p).exists():
        raise SystemExit(f"missing {p}")
print("SFT data is present")
PY
```

## Train

Set common paths:

```bash
export SEARCHSKILL_ROOT="/path/to/SearchSkill Code"
export ROOT="$SEARCHSKILL_ROOT"
export PYTHON_BIN="$(command -v python)"
export HF_MODELS="/path/to/hf_models"
export HF_CACHE="/path/to/hf_cache"
```

Run the backbone-specific wrappers you need:

```bash
bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh
```

For other backbones, use the matching `train_stage*_3b_base.sh`, `train_stage*_3b_instruct.sh`, or `train_stage*_7b_base.sh` scripts.

## Merge

Use `scripts/merge_lora.py` to merge adapters into a checkpoint usable by RL:

```bash
python supervised_finetuning/scripts/merge_lora.py --help
```

By default, merging uses CPU. For a 7B checkpoint, prefer `--device cuda` when GPU memory allows. The merged checkpoint path should be passed to RL through `MODEL_PATH`.
