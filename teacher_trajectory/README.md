# Teacher Trajectory Construction

This stage builds teacher search trajectories, filters them, and prepares canonical examples for supervised fine-tuning. The released repository already includes trajectory artifacts, so users can build SFT data without rerunning expensive teacher rollout.

## Included Artifacts

- `data/canonical_trajectories.jsonl`: selected canonical trajectories used by SFT.

Failed dumps, raw logs, API keys, and transient run directories are intentionally excluded.

## Main Scripts

- `src/build_manifest.py`: builds rollout manifests from sampled examples.
- `src/run_teacher_rollout.py`: calls a teacher model and records search trajectories.
- `src/merge_rollout_outputs.py`: merges shard outputs.
- `src/build_canonical_teacher_set.py`: builds the canonical trajectory set.
- `src/pack_sft.py`: packs trajectories into SFT-ready messages.

## Reuse Path

Use the checked-in canonical trajectories directly:

```bash
test -s teacher_trajectory/data/canonical_trajectories.jsonl
```

Then build SFT data with `supervised_finetuning/scripts/build_stage1_dataset.py`, or use the already included SFT data under `supervised_finetuning/data/`.

## Regeneration Requirements

To rerun teacher rollout you need:

- `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.
- A live retriever endpoint at `RETRIEVER_HOST:RETRIEVER_PORT`.
- The final SkillBank at `skill_bank/round_4_musique/outputs/final_skill_bank.md`.
- Sampled data from `data_preparation/`.
- Enough parallelism or cluster resources for API calls.

Example inspection commands:

```bash
python teacher_trajectory/src/build_manifest.py --help
python teacher_trajectory/src/run_teacher_rollout.py --help
python teacher_trajectory/src/build_canonical_teacher_set.py --help
python teacher_trajectory/src/pack_sft.py --help
```

## Outputs For Next Stage

The SFT stage consumes `teacher_trajectory/data/canonical_trajectories.jsonl`.
