## HotpotQA B1 vs B2 Evaluation

This directory compares `round1_skill_bank` and `round2_skill_bank` on a fixed HotpotQA dev subset with `Qwen2.5-7B-Instruct`.

## Inputs

- `sample_hotpot_dev_subset.py`: samples a fixed subset from a local HotpotQA dev/test file.
- `run_hotpot_b1_b2_qwen.sh`: launches both evaluations with one GPU per SkillBank.
- `skill_bank/nq_eval/eval_nq_qwen_skillbank.py`: shared evaluator reused across datasets.

## Required Environment

```bash
export MODEL_PATH="$HF_MODELS/Qwen2.5-7B-Instruct"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
```

Adjust `CONDA_SH` and `CONDA_ENV` in the shell script if you use conda activation through Slurm.

## Run

```bash
python skill_bank/hotpot_eval/sample_hotpot_dev_subset.py
bash skill_bank/hotpot_eval/run_hotpot_b1_b2_qwen.sh
```
