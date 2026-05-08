# Round B2: HotpotQA Evolution

This round evolves `round1_skill_bank` into `round2_skill_bank` using the sampled
HotpotQA training pool.

Round goal:

- preserve the B1 single-hop gains
- add or refine reusable multihop skills for bridge, comparison, and
  multi-step verification
- sharpen the boundaries between single-hop and multihop skills so the
  student model does not over-trigger the new skills

Inputs:

- `skill_bank/round_1_singlehop/outputs/round1_skill_bank.md`
- `data_preparation/samples/hotpotqa/train_sample_light.jsonl`
- `data_preparation/samples/hotpotqa/train_sample_full.jsonl`

Expected artifacts:

- `artifacts/`: grouped packets, packet summaries, intermediate GPT outputs
- `outputs/round2_skill_bank.md`: evolved bank after review
- `logs/`: request payloads, raw model responses, token accounting

Planned workflow:

1. Read `train_sample_light.jsonl` and `train_sample_full.jsonl` from `hotpotqa`
2. Group by signature and build representative packets
3. Summarize the multihop evidence into bridge / comparison / verification buckets
4. Ask `GPT-5.4` to expand B1 with high-quality multihop skills and targeted refinements
5. Use `round2_skill_bank` for later trajectory generation and cross-dataset evolution

Reproduction commands:

```bash
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py
```

`run_b2_expand.py` requires `OPENAI_API_KEY` and an OpenAI-compatible `OPENAI_BASE_URL`. For normal downstream reproduction, reuse `outputs/round2_skill_bank.md` instead of rerunning this round.
