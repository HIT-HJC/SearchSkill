## HotpotQA B1 vs B2 Eval

This directory holds a lightweight evaluation pipeline for comparing
`round1_skill_bank` and `round2_skill_bank` on a fixed HotpotQA dev subset with
`Qwen2.5-7B-Instruct`.

- `sample_hotpot_dev_subset.py` samples a fixed subset from the synced
  HotpotQA dev file (`test.jsonl` in the local mirror).
- `run_hotpot_b1_b2_qwen.sh` launches two parallel evaluations inside an
  existing Slurm allocation, one GPU per skill bank.

The actual evaluator reuses
`/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/nq_eval/eval_nq_qwen_skillbank.py`
because the evaluation logic is dataset-agnostic.
