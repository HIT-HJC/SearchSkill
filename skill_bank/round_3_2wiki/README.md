# Round B3: 2Wiki Evolution

This round evolves `round2_skill_bank` into `round3_skill_bank` using the sampled
2Wiki training pool.

Round goal:

- preserve the B1 single-hop and B2 hotpot-style multihop gains
- add or refine reusable compositional skills for relation chains,
  bridge-comparison hybrids, and inference-style retrieval
- sharpen the boundaries between shorter bridge skills and longer
  compositional skills so routing does not over-trigger the new policies

Inputs:

- `skill_bank/round_2_hotpotqa/outputs/round2_skill_bank.md`
- `data_preparation/samples/2wiki/train_sample_light.jsonl`
- `data_preparation/samples/2wiki/train_sample_full.jsonl`

Expected artifacts:

- `artifacts/`: grouped packets, packet summaries, intermediate GPT outputs
- `outputs/round3_skill_bank.md`: evolved bank after review
- `logs/`: request payloads, raw model responses, token accounting

Planned workflow:

1. Read `train_sample_light.jsonl` and `train_sample_full.jsonl` from `2wiki`
2. Group by signature and build representative packets
3. Summarize the compositional evidence into chain / comparison /
   inference / disambiguation buckets
4. Ask `GPT-5.4` to expand B2 with high-quality compositional skills and
   targeted refinements
5. Use `round3_skill_bank` as the base bank for the later MuSiQue round

Reproduction commands:

```bash
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py
```

`run_b3_expand.py` requires `OPENAI_API_KEY` and an OpenAI-compatible `OPENAI_BASE_URL`. For normal downstream reproduction, reuse `outputs/round3_skill_bank.md` instead of rerunning this round.
