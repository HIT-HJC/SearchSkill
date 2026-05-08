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

Minimal runnable recipe:

```bash
python teacher_trajectory/src/build_manifest.py \
  --output-dir teacher_trajectory/runs/example/manifest \
  --train-datasets hotpotqa \
  --hotpot-count 20 \
  --nq-count 0 \
  --triviaqa-count 0 \
  --2wiki-count 0 \
  --musique-count 0 \
  --failure-count 0

python teacher_trajectory/src/run_teacher_rollout.py \
  --manifest-path teacher_trajectory/runs/example/manifest/manifest.jsonl \
  --output-dir teacher_trajectory/runs/example/rollout \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --base-url "$OPENAI_BASE_URL" \
  --retriever-host "$RETRIEVER_HOST" \
  --retriever-port "$RETRIEVER_PORT" \
  --max-examples 20 \
  --resume
```

After rollout, merge and select canonical trajectories:

```bash
python teacher_trajectory/src/merge_rollout_outputs.py \
  --input-dirs teacher_trajectory/runs/example/rollout \
  --output-dir teacher_trajectory/runs/example/merged

python teacher_trajectory/src/build_canonical_teacher_set.py \
  --input-spec example=teacher_trajectory/runs/example/merged \
  --output-dir teacher_trajectory/runs/example/canonical

python teacher_trajectory/src/pack_sft.py \
  --filtered-path teacher_trajectory/runs/example/canonical/trajectories.filtered.jsonl \
  --output-path teacher_trajectory/runs/example/sft/train.jsonl
```

Use `bin/*.sh` as launch examples, not universal scripts. The released canonical set was assembled from the same `src/` pipeline, then filtered into `runs/canonical_teacher_set/all/trajectories.filtered.jsonl`. Replace any Slurm, API, retriever, or path settings for your environment.
