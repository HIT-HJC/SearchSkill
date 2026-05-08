# SearchSkill Code

This repository contains the cleaned release package for the SearchSkill experimental pipeline. It is organized by method stage, so a new user can inspect the released artifacts, reuse the included data products, or rerun the pipeline from data preparation through SkillBank construction, teacher trajectories, supervised fine-tuning, reinforcement learning, evaluation, and analysis.

The repository includes code, sampled data, SkillBank artifacts, teacher trajectory artifacts, SFT/RL training data, benchmark subsets, and the vendored runtime changes needed by the RL stage. It does not include model weights, trained checkpoints, retrieval indexes, private API keys, logs, cache directories, or Ray state.

## What Can Be Reproduced

There are two supported reproduction modes:

1. Artifact reuse: start from the included sampled data, SkillBank, teacher trajectories, SFT data, and RL data. This is the most stable path after cloning the repository.
2. Full regeneration: rerun data sampling, SkillBank expansion, teacher rollout, SFT data packing, SFT training, RL data building, RL training, and evaluation. This requires external datasets, base model weights, a live retriever, API credentials for teacher or SkillBank generation, and enough GPU resources.

## Repository Layout

- `data_preparation/`: dataset profiling, sampling, group annotation, and sampled training pools.
- `skill_bank/`: seed-to-final SkillBank construction, per-round artifacts, and SkillBank evaluation helpers.
- `teacher_trajectory/`: teacher rollout manifests, trajectory records, canonical trajectory sets, and SFT packing inputs.
- `supervised_finetuning/`: two-stage SFT data, LoRA training scripts, and merge helpers.
- `reinforcement_learning/`: policy optimization data, launch scripts, reward checks, and analysis outputs.
- `benchmarks/`: fixed benchmark subsets, sample manifests, and benchmark resampling utilities.
- `analysis_scripts/`: shared post-evaluation analysis utilities.
- `external/SearchR1/`: vendored runtime checkout with SearchSkill trainer and reward code already applied.
- `external/SearchR1_patch/`: audit copy of the SearchSkill-specific runtime files.

## Required External Assets

Set these paths before running training or evaluation:

```bash
export ROOT="/path/to/SearchSkill Code"
export SEARCHSKILL_ROOT="$ROOT"
export PYTHON_BIN="/path/to/conda/env/bin/python"
export HF_MODELS="/path/to/hf_models"
export HF_DATA="/path/to/hf_data"
export HF_CACHE="/path/to/hf_cache"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
```

Only set `OPENAI_API_KEY` and `OPENAI_BASE_URL` when regenerating SkillBank expansions or teacher trajectories:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Expected model directories under `HF_MODELS` are the Qwen2.5 3B/7B base and instruct checkpoints used by the launch scripts. The API scripts accept `OPENAI_BASE_URL` with or without a trailing `/v1`. The retriever should expose a Search-R1-compatible `/retrieve` endpoint over HTTP.

## Quick Smoke Checks

After cloning, run these checks before launching expensive jobs:

```bash
cd "$SEARCHSKILL_ROOT"
git status --short

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
required = [
    "skill_bank/round_4_musique/outputs/final_skill_bank.md",
    "supervised_finetuning/data/stage2/train.jsonl",
    "reinforcement_learning/data/policy_7b_instruct/train.parquet",
    "external/SearchR1/verl/utils/reward_score/searchskill.py",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"missing required artifacts: {missing}")
print("SearchSkill artifact check passed")
PY
```

Optional retriever check:

```bash
"$PYTHON_BIN" - <<'PY'
import os, requests
host = os.environ.get("RETRIEVER_HOST", "127.0.0.1")
port = os.environ.get("RETRIEVER_PORT", "8000")
r = requests.post(
    f"http://{host}:{port}/retrieve",
    json={"queries": ["Barack Obama birthplace"], "topk": 1, "return_scores": True},
    timeout=30,
)
r.raise_for_status()
print("retriever check passed")
PY
```

## End-To-End Reproduction Order

1. Read [REPRODUCE.md](REPRODUCE.md) for the complete command flow and resource assumptions.
2. Inspect or regenerate sampled data under `data_preparation/`.
3. Reuse or regenerate the final SkillBank at `skill_bank/round_4_musique/outputs/final_skill_bank.md`.
4. Reuse or regenerate canonical teacher trajectories under `teacher_trajectory/runs/canonical_teacher_set/`.
5. Reuse or rebuild SFT datasets under `supervised_finetuning/data/stage1/` and `supervised_finetuning/data/stage2/`.
6. Train and merge LoRA adapters under `supervised_finetuning/models/`.
7. Reuse or rebuild RL parquet data under `reinforcement_learning/data/`.
8. Launch RL with `reinforcement_learning/scripts/train_*.sh`.
9. Evaluate with `reinforcement_learning/scripts/evaluate_policy.sh` and analyze outputs with `analysis_scripts/`.

## Public Release Notes

- All known local absolute paths and private endpoint defaults have been replaced with relative paths or placeholders.
- `.gitignore` excludes checkpoints, model binaries, caches, logs, Ray state, `wandb/`, and environment files.
- Large JSONL data files are included for reproducibility. Some exceed GitHub's 50 MB recommendation but are below the hard 100 MB file limit; move them to Git LFS or an external artifact release if your publication policy requires that.
- The vendored runtime is included so users do not need to reconstruct local runtime patches before running the SearchSkill RL trainer.
