# Vendored Search-R1/VERL Runtime

This checkout is included so SearchSkill experiments can be launched directly from this repository.

## SearchSkill Entry Points

RL trainer:

```bash
python -m verl.trainer.main_ppo_searchskill
```

Reward implementation:

```bash
verl/utils/reward_score/searchskill.py
```

Rollout and retrieval integration:

```bash
search_r1/llm_agent/searchskill_generation.py
search_r1/search/retrieval.py
search_r1/search/retrieval_server.py
```

## Install

From the repository root:

```bash
python -m pip install -r external/SearchR1/requirements.txt
python -m pip install -e external/SearchR1
```

## Release Cleanup

Build artifacts, logs, caches, Ray state, local evaluation outputs, and private environment files were removed. Model weights and retrieval indexes must be provided separately.
