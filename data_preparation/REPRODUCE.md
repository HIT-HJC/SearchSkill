# Reproducing Data Preparation

For the full project flow, start with `../REPRODUCE.md`. This file only covers the data-preparation stage.

## Stable Path

The released repository already includes sampled data, profiles, group annotations, and reports. To verify them:

```bash
python - <<'PY'
from pathlib import Path
required = [
    "data_preparation/samples/nq/train_sample_light.jsonl",
    "data_preparation/samples/hotpotqa/train_sample_light.jsonl",
    "data_preparation/samples/2wiki/train_sample_light.jsonl",
    "data_preparation/samples/musique/train_sample_light.jsonl",
    "data_preparation/reports/sampling_summary.json",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(missing)
print("data-preparation artifacts are present")
PY
```

## Full Regeneration

```bash
export SEARCHSKILL_ROOT="/path/to/SearchSkill Code"
export HF_DATA="/path/to/hf_data"
export HF_CACHE="/path/to/hf_cache"
export PYTHON_BIN="$(command -v python)"

python -m pip install -r data_preparation/requirements.txt
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh
```

`sample_singlehop_train.py` reads JSONL mirrors from `HF_DATA` or `--hf-data-root`. `sample_multihop_train.py` reads evaluation JSONL mirrors from `HF_DATA` or `--hf-data-root`, and FlashRAG `.arrow` cache files from `HF_CACHE` or `--hf-cache-root`. A typical layout is:

```text
$HF_DATA/data/nq/train.jsonl
$HF_DATA/data/nq/test.jsonl
$HF_DATA/data/hotpotqa/test.jsonl
$HF_CACHE/datasets/RUC-NLPIR___flash_rag_datasets/hotpotqa/.../*.arrow
```

Review the generated reports before proceeding:

```bash
ls data_preparation/reports
```

If the regenerated samples differ from the release, document the dataset mirror, seed, and script arguments used.
