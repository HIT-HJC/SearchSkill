# Benchmarks

This directory only contains JSONL test files used for quick public evaluation.

```text
benchmarks/
  dev/
    nq.jsonl
    triviaqa.jsonl
    popqa.jsonl
    hotpotqa.jsonl
    2wiki.jsonl
    musique.jsonl
    bamboogle.jsonl
  full/
    nq.jsonl
    triviaqa.jsonl
    popqa.jsonl
    hotpotqa.jsonl
    2wiki.jsonl
    musique.jsonl
    bamboogle.jsonl
```

Run dev tests:

```bash
MODEL_PATH="<model_or_checkpoint>" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh nq
```

Before running eval, install `requirements-eval.txt`, provide a local checkpoint or Hugging Face model id in `MODEL_PATH`, and start the retriever server described in the root README. The launcher defaults to one GPU; override `SHARD_COUNT` and `GPU_IDS_CSV` for multi-GPU evaluation.

Run full tests:

```bash
MODEL_PATH="<model_or_checkpoint>" BENCHMARK_SPLIT=full bash reinforcement_learning/scripts/evaluate_policy.sh all
```
