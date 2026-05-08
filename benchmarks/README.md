# Benchmarks

This folder contains fixed benchmark subsets and utilities used to evaluate SearchSkill policies.

## Contents

- `singlehop/`: NQ, TriviaQA, and PopQA benchmark subsets plus sample indices.
- `multihop_toolstar/`: multi-hop evaluation files for HotpotQA, 2Wiki, MuSiQue, and Bamboogle.
- `sample_1000_manifest.json`: provenance for the 1,000-example single-hop subsets.
- `resample_*.py`: utilities for rebuilding benchmark subsets from local dataset mirrors.
- `analyze_*.py` and `compare_*.py`: distribution and comparison scripts for evaluation outputs.

## Reuse Path

The checked-in benchmark files are ready to use with the evaluation scripts. They do not require regeneration.

## Regeneration Path

Provide a local dataset mirror and run the relevant resampling script, for example:

```bash
python benchmarks/resample_singlehop_balanced_harder.py \
  --data-root "$HF_DATA/data" \
  --output-root benchmarks/singlehop
```

Do not commit temporary backups or ad-hoc output folders. Keep new benchmark manifests if you intentionally change the benchmark set.

## Notes

Some analysis scripts contain relative paths to expected evaluation outputs under `outputs/` or `eval/`. Update those paths to match your run directory before using them.
