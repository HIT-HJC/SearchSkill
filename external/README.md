# External Runtime Patch

This repository does not vendor a full RL runtime. The directory `runtime_patch/` contains only the SearchSkill-specific files that need to be copied or overlaid into a VERL-style RL runtime with the package layout used by the training scripts.

Expected workflow:

```bash
cp -r external/runtime_patch/* "<rl_runtime>/"
export RUNTIME_ROOT="<rl_runtime>"
```

Then run:

```bash
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

The runtime must provide the trainer, Ray workers, rollout code, and reward-score package expected by the files in `runtime_patch/`.
After applying the patch, imports such as `verl.trainer.main_ppo_searchskill` should resolve inside your runtime environment.
