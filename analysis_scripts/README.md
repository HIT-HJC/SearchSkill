# Analysis Scripts

This folder contains small post-evaluation utilities for diagnosing SearchSkill behavior. They are not required for training, but they are useful for comparing SFT and RL policies.

## Scripts

- `measure_rl_action_metrics.py`: computes action-level and retrieval-behavior metrics from trace JSONL files.
- `analyze_rl_action_diff.py`: compares selected SFT and RL traces and highlights changed behavior.
- `diagnose_rl_training_direction.py`: inspects whether RL updates move behavior in the intended direction.

## Inputs

These scripts expect evaluation trace files produced by `reinforcement_learning/scripts/evaluate_policy.sh` or equivalent evaluation runs. Edit the relative paths at the top of each script to point at your local `eval/` or `outputs/` directory.

## Example

```bash
python analysis_scripts/measure_rl_action_metrics.py
```

If a script cannot find its input trace, update the path constants instead of moving generated outputs into the repository.
