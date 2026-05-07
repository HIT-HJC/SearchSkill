## 2Wiki B2 vs B3 Eval

This directory holds a lightweight evaluation pipeline for comparing
`round2_skill_bank` and `round3_skill_bank` on a fixed 2Wiki dev subset with
`Qwen2.5-7B-Instruct`.

- `sample_2wiki_dev_subset.py` samples a fixed subset from the synced
  2Wiki dev file (`test.jsonl` in the local mirror).
- `run_2wiki_b2_b3_qwen.sh` launches two parallel evaluations inside an
  existing Slurm allocation, one GPU per skill bank.

The actual evaluator reuses
`/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/nq_eval/eval_nq_qwen_skillbank.py`
because the evaluation logic is dataset-agnostic.
