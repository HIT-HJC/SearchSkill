# Reproducing SkillBank Construction

For the full project flow, start with `../REPRODUCE.md`. This file only covers SkillBank construction.

## Stable Path

The final SkillBank is already included:

```bash
test -s skill_bank/round_4_musique/outputs/final_skill_bank.md
```

Downstream stages should use this file unless you intentionally rerun SkillBank evolution.

## Full Regeneration

Set API credentials:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
```

Run each round in order:

```bash
python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py

python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py

python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py

python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py
```

After rerunning, compare:

```bash
git diff -- skill_bank/round_4_musique/outputs/final_skill_bank.md
```

Only commit a changed SkillBank after checking that the new skills remain stable, non-duplicative, and compatible with the two-stage SFT protocol.
