# SearchSkill Patch Audit

This folder mirrors the SearchSkill-specific runtime files included in `external/SearchR1/`. It is not a separate package; it is an audit and portability aid.

## Key Files

- `verl/trainer/main_ppo_searchskill.py`: SearchSkill RL trainer entry point.
- `verl/utils/reward_score/searchskill.py`: reward and trajectory scoring logic.
- `search_r1/llm_agent/searchskill_generation.py`: policy rollout logic for SearchSkill tool use.
- `search_r1/search/retrieval.py`, `retrieval_server.py`, and `retrieval.sh`: retrieval-side compatibility files.
- Selected VERL/vLLM compatibility patches needed by the released runtime.

## How To Use

For normal reproduction, install and run `external/SearchR1/`. Do not run code from this patch folder directly.

To audit the patch:

```bash
diff -ru external/SearchR1_patch external/SearchR1
```

The diff will include path layout differences because this folder only contains the SearchSkill-specific subset.
