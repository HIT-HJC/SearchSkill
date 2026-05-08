## MuSiQue B3 vs Final Evaluation

This directory compares `round3_skill_bank` and `final_skill_bank` on a fixed MuSiQue dev subset with `Qwen2.5-7B-Instruct`.

## Inputs

- `sample_musique_smallval_subset.py`: samples a fixed subset from a local MuSiQue dev/test file.
- `run_musique_b3_b4_qwen.sh`: launches both evaluations with one GPU per SkillBank.
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
python skill_bank/musique_eval/sample_musique_smallval_subset.py
bash skill_bank/musique_eval/run_musique_b3_b4_qwen.sh
```
