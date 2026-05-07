# Vendored Search-R1/VERL

This checkout is included so SearchSkill experiments can be launched directly from this package. It contains the SearchSkill trainer entry point:

```bash
python -m verl.trainer.main_ppo_searchskill
```

and the reward implementation:

```bash
verl/utils/reward_score/searchskill.py
```

Build artifacts, logs, caches, and local evaluation outputs were removed.
