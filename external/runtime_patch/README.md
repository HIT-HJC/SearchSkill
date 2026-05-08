# Runtime Patch Files

These files are the SearchSkill-specific runtime changes used by RL training.

Copy this directory over your RL runtime root:

```bash
cp -r external/runtime_patch/* "<rl_runtime>/"
export RUNTIME_ROOT="<rl_runtime>"
```

Do not run files from `external/runtime_patch/` directly; they are patch files meant to live inside the runtime package.
