# SkillBank

This stage builds the final retrieval skill bank used by teacher trajectories, SFT data construction, and RL training.

## Use Released Bank

```bash
skill_bank/round_4_musique/outputs/final_skill_bank.md
```

No API key is required if you reuse this file.

## Rebuild

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py --base-url "$OPENAI_BASE_URL"
```

## Clean Test Entry

`nq_eval/eval_nq_qwen_skillbank.py` is the kept evaluation entry. It does not add rule-based skill recommendations by default.
