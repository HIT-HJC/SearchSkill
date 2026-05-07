# Teacher Trajectory Construction

This module builds teacher trajectories, filters them, and prepares the canonical examples used by supervised fine-tuning.

## Main Scripts

- `src/build_manifest.py`: builds rollout manifests from sampled data.
- `src/run_teacher_rollout.py`: calls the teacher model and records search trajectories.
- `src/merge_rollout_outputs.py`: merges shard outputs.
- `src/build_canonical_teacher_set.py`: builds the canonical teacher set.
- `src/pack_sft.py`: packs trajectories into SFT-ready messages.
- `bin/*.sh`: launch helpers for API and Slurm runs.

## Included Runs

- `runs/canonical_teacher_set/`: selected canonical trajectories.
- `runs/coverage_supplement/`: supplemental coverage trajectories.
- `runs/multi_hop_teacher/`: multi-hop teacher rollouts.
- `runs/single_hop_teacher/`: single-hop teacher rollouts.

Failed trajectory dumps and transient logs were removed from the release package.

## Replace Before Running

Set `OPENAI_API_KEY`, teacher model/API options, `PYTHON_BIN`, and any Slurm node or partition settings in `bin/*.sh`.
