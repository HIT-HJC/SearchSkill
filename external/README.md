# External Runtime

This folder contains vendored runtime code needed by SearchSkill.

## `SearchR1/`

`SearchR1/` is a cleaned vendored Search-R1/VERL checkout with SearchSkill changes already applied. It is included so users can launch the SearchSkill RL trainer without manually reconstructing runtime patches.

Important entry points:

```bash
python -m verl.trainer.main_ppo_searchskill
```

Important SearchSkill files:

- `verl/trainer/main_ppo_searchskill.py`
- `verl/utils/reward_score/searchskill.py`
- `search_r1/llm_agent/searchskill_generation.py`
- `search_r1/search/retrieval_server.py`
- `search_r1/search/retrieval.py`

Install in editable mode from the repository root:

```bash
python -m pip install -r external/SearchR1/requirements.txt
python -m pip install -e external/SearchR1
```

## `SearchR1_patch/`

`SearchR1_patch/` mirrors the SearchSkill-specific files and is kept for audit. Use it to compare the release modifications against the vendored checkout, or to port the same changes to a different Search-R1 checkout.

## What Is Not Included

Git metadata, checkpoints, logs, Ray state, local HuggingFace caches, vLLM build outputs, and ad-hoc evaluation output folders are excluded.
