# Reproducing Teacher Trajectories

For the full project flow, start with `../REPRODUCE.md`. This file only covers teacher trajectory generation.

## Stable Path

Use the released canonical trajectory set:

```bash
python - <<'PY'
from pathlib import Path
path = Path("teacher_trajectory/runs/canonical_teacher_set/all/trajectories.filtered.jsonl")
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit("missing canonical teacher trajectories")
print("canonical teacher trajectories are present")
PY
```

## Full Regeneration

Set:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export RETRIEVER_HOST="${RETRIEVER_HOST:-127.0.0.1}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
export PYTHON_BIN="$(command -v python)"
```

Build manifests and run rollouts with the scripts in `src/`. Because manifest options are experiment-specific, inspect arguments first:

```bash
python teacher_trajectory/src/build_manifest.py --help
python teacher_trajectory/src/run_teacher_rollout.py --help
```

After rollout, merge and select canonical trajectories:

```bash
python teacher_trajectory/src/merge_rollout_outputs.py --help
python teacher_trajectory/src/build_canonical_teacher_set.py --help
python teacher_trajectory/src/pack_sft.py --help
```

Use `bin/*.sh` as launch examples, not universal scripts. Replace any Slurm, API, retriever, or path settings for your environment.
