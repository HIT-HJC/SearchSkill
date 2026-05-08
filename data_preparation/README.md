# Data Preparation

This stage builds and documents the sampled training pools used by SearchSkill. The repository already includes the sampled artifacts needed by later stages, so a new user can inspect or reuse them without rerunning sampling.

## Included Artifacts

- `samples/`: sampled full and light training pools for single-hop and multi-hop datasets.
- `profiles/`: per-example metadata used for balancing and diagnostics.
- `group_annotations/`: group labels used by the sampling and SkillBank stages.
- `reports/`: sampling summaries and profile reports.
- `requirements.txt`: lightweight dependencies for data preparation scripts.

## Scripts

- `sample_singlehop_train.py`: samples NQ, TriviaQA, and related single-hop data.
- `sample_multihop_train.py`: samples HotpotQA, 2Wiki, and MuSiQue data.
- `run_singlehop_sampling.sh`: shell wrapper for single-hop sampling.
- `run_multihop_sampling.sh`: shell wrapper for multi-hop sampling.

## Inputs You Must Provide To Regenerate

Set these variables before rerunning:

```bash
export SEARCHSKILL_ROOT="/path/to/SearchSkill Code"
export HF_DATA="/path/to/hf_data"
export HF_CACHE="/path/to/hf_cache"
export PYTHON_BIN="/path/to/python"
```

The scripts expect local mirrors of the original datasets. Paths inside the scripts are placeholders by design; adjust them for your dataset layout.

## Reuse Path

Use the checked-in artifacts directly:

```bash
ls data_preparation/samples
ls data_preparation/reports
```

Later stages consume the included samples and do not require rerunning this stage.

## Regeneration Path

```bash
python -m pip install -r data_preparation/requirements.txt
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh
```

If you enable teacher-assisted grouping in `sample_multihop_train.py`, also set:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Outputs For Next Stage

The SkillBank stage reads sampled pools and profile reports from this directory, especially `samples/*/train_sample_light.jsonl` and `samples/*/train_sample_full.jsonl`.
