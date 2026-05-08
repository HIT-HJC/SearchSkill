# Skill Bank

This stage evolves a seed SkillBank into the final SkillBank used by the two-stage SFT policy and the RL policy. The released repository already includes all round artifacts and the final SkillBank, so downstream users normally reuse the checked-in result.

## Main Artifact

```bash
skill_bank/round_4_musique/outputs/final_skill_bank.md
```

This is the SkillBank used by:

- `supervised_finetuning/scripts/build_stage2_dataset.py`
- `reinforcement_learning/scripts/build_policy_dataset.py`
- `external/SearchR1/verl/trainer/main_ppo_searchskill.py`

## Evolution Rounds

1. `inputs/seed_skill_bank.md`: seed skills.
2. `round_1_singlehop/`: single-hop expansion from NQ and TriviaQA examples.
3. `round_2_hotpotqa/`: bridge, comparison, and verification expansion.
4. `round_3_2wiki/`: relation-chain and disambiguation expansion.
5. `round_4_musique/`: long-hop decomposition and final export.

Each round contains:

- `build_packets.py`: creates representative packets from sampled examples.
- `run_b*_expand.py`: calls a teacher model to propose additions or refinements.
- `config.json`: round configuration and input pointers.
- `artifacts/`: packet summaries and model-generation summaries.
- `outputs/`: accepted SkillBank output for that round.

## Reuse Path

Use the final checked-in SkillBank directly:

```bash
less skill_bank/round_4_musique/outputs/final_skill_bank.md
```

No API key is required for reuse.

## Regeneration Path

Regeneration requires a model API compatible with the OpenAI API shape:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py
```

Review generated outputs before replacing the checked-in final SkillBank.

## Evaluation Helpers

- `nq_eval/`: single-hop and prompt-ablation evaluation utilities.
- `hotpot_eval/`, `2wiki_eval/`, `musique_eval/`: round-to-round comparison helpers.

These helpers require a local model path, a live retriever endpoint, and dataset-specific dev/test files.
