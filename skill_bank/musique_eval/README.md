## MuSiQue B3 vs final Eval

This directory holds a lightweight evaluation pipeline for comparing
`round3_skill_bank` and `final_skill_bank` on a fixed small MuSiQue dev subset with
`Qwen2.5-7B-Instruct`.

- `sample_musique_smallval_subset.py` samples a fixed subset from the synced
  MuSiQue dev file (`test.jsonl` in the local mirror).
- `run_musique_b3_b4_qwen.sh` launches two parallel evaluations inside an
  existing Slurm allocation, one GPU per skill bank.

The actual evaluator reuses
`skill_bank/nq_eval/eval_nq_qwen_skillbank.py`
because the evaluation logic is dataset-agnostic.
