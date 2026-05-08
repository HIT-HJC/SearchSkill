# Round B1: Single-Hop Evolution

This round evolves `seed_skill_bank` into `round1_skill_bank` using the sampled
single-hop training pools from:

- `nq`
- `triviaqa`

Round goal:

- keep the existing multihop bank stable
- let the teacher model add new single-hop skills aggressively when the pattern is stable and reusable
- still avoid noisy near-duplicate skills or broad rewrites with weak evidence

Inputs:

- `skill_bank/inputs/seed_skill_bank.md`
- `data_preparation/samples/nq/train_sample_light.jsonl`
- `data_preparation/samples/triviaqa/train_sample_light.jsonl`

Expected artifacts:

- `artifacts/`: grouped packets, intermediate GPT outputs, edit proposals
- `outputs/round1_skill_bank.md`: evolved bank after review
- `outputs/trajectory_seed_plan.json`: optional plan for later trajectory generation
- `logs/`: request logs, token accounting, and run summaries

Planned workflow:

1. Read `train_sample_light.jsonl` from `nq` and `triviaqa`
2. Group or batch representative question packets
3. Ask `GPT-5.4` to expand seed with new single-hop skills, plus any necessary refinements
4. Merge the accepted additions and refinements into `round1_skill_bank`
5. Use `round1_skill_bank` to generate high-quality single-hop teacher trajectories later

Reproduction commands:

```bash
python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py
```

`run_b1_expand.py` requires `OPENAI_API_KEY` and an OpenAI-compatible `OPENAI_BASE_URL`. For normal downstream reproduction, reuse `outputs/round1_skill_bank.md` instead of rerunning this round.
