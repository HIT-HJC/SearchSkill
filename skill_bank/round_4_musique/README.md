# Round final: MuSiQue Evolution

This round evolves `round3_skill_bank` into `final_skill_bank` using the sampled
MuSiQue training pool.

Round goal:

- preserve the B1 to B3 gains
- add or refine reusable decomposition skills for 3-hop and 4-hop
  retrieval, checkpointing, and mid-chain recovery
- sharpen the boundaries between medium-length multihop skills and the
  longer MuSiQue-specific decomposition skills

Inputs:

- `skill_bank/round_3_2wiki/outputs/round3_skill_bank.md`
- `data_preparation/samples/musique/train_sample_light.jsonl`
- `data_preparation/samples/musique/train_sample_full.jsonl`

Expected artifacts:

- `artifacts/`: grouped packets, packet summaries, intermediate GPT outputs
- `outputs/final_skill_bank.md`: evolved bank after review
- `logs/`: request payloads, raw model responses, token accounting

Planned workflow:

1. Read `train_sample_light.jsonl` and `train_sample_full.jsonl` from `musique`
2. Group by signature and build representative packets
3. Summarize the long-hop evidence into decomposition / checkpointing /
   endpoint extraction / disambiguation buckets
4. Ask `GPT-5.4` to expand B3 with high-quality long-hop skills and
   targeted refinements
5. Use `final_skill_bank` for later trajectory generation and SFT

Reproduction commands:

```bash
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py
```

`run_b4_expand.py` requires `OPENAI_API_KEY` and an OpenAI-compatible `OPENAI_BASE_URL`. The final downstream artifact is `outputs/final_skill_bank.md`.
