# SearchSkill

This repository is a compact release of the SearchSkill pipeline. It keeps the files needed to understand and rerun the main flow:

1. prepare data,
2. build the skill bank,
3. build teacher trajectories,
4. train SFT adapters,
5. run RL,
6. evaluate on the included dev or full test files.

Model weights, trained checkpoints, retrieval indexes, API keys, caches, logs, and internal experiment outputs are not included.

## Layout

- `data_preparation/`: sampling and profiling scripts plus released sampled data.
- `skill_bank/`: seed bank, four construction rounds, final bank, and one clean evaluation script.
- `teacher_trajectory/`: trajectory-generation code and the released canonical trajectory file.
- `supervised_finetuning/`: SFT data builders and training/merge scripts.
- `reinforcement_learning/`: RL data builder and launch/evaluation scripts.
- `benchmarks/`: `dev/` and `full/` test JSONL files.
- `external/runtime_patch/`: the SearchSkill-specific runtime files to copy into your RL runtime.

## Quick Evaluation

Set paths:

```bash
export SEARCHSKILL_ROOT="$(pwd)"
export ROOT="$SEARCHSKILL_ROOT"
export PYTHON_BIN="$(command -v python)"
export MODEL_PATH="<model_or_checkpoint>"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
```

Run a small dev test:

```bash
cd "$SEARCHSKILL_ROOT"
MODEL_PATH="$MODEL_PATH" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh nq
```

Run all included dev tests:

```bash
MODEL_PATH="$MODEL_PATH" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh all
```

Use `BENCHMARK_SPLIT=full` for the full test files.

## Rebuild Pipeline

For rebuilding intermediate files, follow [REPRODUCE.md](REPRODUCE.md). The shortest path is:

```bash
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh

python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py --base-url "$OPENAI_BASE_URL"

python teacher_trajectory/src/build_manifest.py --output-dir teacher_trajectory/work/manifest
python teacher_trajectory/src/run_teacher_rollout.py --help

bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

## Public Release Notes

- Local paths and machine-specific defaults are replaced by placeholders or environment variables.
- Benchmarks are intentionally simple: `benchmarks/dev/*.jsonl` and `benchmarks/full/*.jsonl`.
- The repository keeps patch files for the RL runtime instead of vendoring a full external codebase.
