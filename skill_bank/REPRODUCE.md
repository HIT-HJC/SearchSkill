# Skill Bank

This module evolves a seed SkillBank into the final SkillBank used by the two-stage policy and reinforcement-learning runs.

## Flow

1. `inputs/seed_skill_bank.md`: seed skills.
2. `round_1_singlehop/`: single-hop skill expansion.
3. `round_2_hotpotqa/`: bridge and comparison skill expansion.
4. `round_3_2wiki/`: multi-hop relation and disambiguation expansion.
5. `round_4_musique/`: compositional expansion and final SkillBank export.

The final artifact is:

```bash
skill_bank/round_4_musique/outputs/final_skill_bank.md
```

## Replace Before Running

Skill expansion scripts require `OPENAI_API_KEY` if closed-model expansion is used. Evaluation scripts require a local model path via `MODEL_PATH`, plus retriever host and port if retrieval is enabled.
