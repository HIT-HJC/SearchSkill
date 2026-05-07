# External Dependencies

`SearchR1/` is a cleaned vendored Search-R1/VERL checkout with the SearchSkill policy trainer and reward implementation already applied. Users do not need to clone Search-R1 separately.

`SearchR1_patch/` is kept as an audit copy of the files added or modified for SearchSkill. It can be compared against the vendored checkout or applied to another Search-R1 checkout if needed.

Excluded from the vendored checkout: git metadata, run outputs, logs, Ray temporary state, local HuggingFace cache/home, vLLM build artifacts, and ad-hoc evaluation output folders.
