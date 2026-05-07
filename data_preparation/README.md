# Data Preparation

This module builds the sampled single-hop and multi-hop training pools used by SearchSkill.

## Contents

- `sample_singlehop_train.py`: samples single-hop data such as NQ, TriviaQA, and PopQA.
- `sample_multihop_train.py`: samples multi-hop data such as HotpotQA, 2Wiki, and MuSiQue.
- `run_singlehop_sampling.sh` and `run_multihop_sampling.sh`: shell entry points.
- `group_annotations/`: group labels used for balancing and diagnostics.
- `samples/`, `profiles/`, `reports/`: included artifacts for reproducibility.

## Replace Before Running

Set `HF_DATA`, `HF_CACHE`, and any dataset-specific paths used by the scripts. If teacher-assisted annotation is regenerated, set `OPENAI_API_KEY` and the API base/model options used by the Python scripts.
