# Retrieval Skill Bank v2 Refined

`bridge-entity-search`
Use when the question identifies the answer through a hidden bridge entity. First retrieve the bridge entity, then search the requested attribute of that entity.

`parallel-attribute-compare`
Use for comparisons and yes/no questions about two entities. Retrieve evidence for both sides before deciding.

`temporal-range-extract`
Use for time, year, count, height, and range questions. Preserve the exact temporal or numeric form supported by evidence.

`conflict-check`
Use when retrieved passages contain multiple plausible entities, organizations, or numbers. Run one targeted verification search anchored on the current candidate and asked relation.

`verbatim-evidence-span`
Use before answering. Copy the final answer as a short span from evidence instead of paraphrasing away titles, units, or ranges.

`answer-grounding-check`
Use after a draft answer appears. If the answer is not explicitly supported by the currently retrieved evidence, do one extra targeted search instead of finalizing an unsupported guess.
