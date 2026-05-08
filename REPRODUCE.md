# Reproducing SearchSkill

This file gives the minimal public workflow. All commands assume:

```bash
git clone https://github.com/HIT-HJC/SearchSkill.git
cd SearchSkill
export ROOT="$(pwd)"
export SEARCHSKILL_ROOT="$ROOT"
export PYTHON_BIN="$(command -v python)"
```

## 1. Environment

Install normal Python/CUDA packages for your machine. Data processing needs:

```bash
python -m pip install -r data_preparation/requirements.txt
```

SFT/RL additionally needs PyTorch, Transformers, Datasets, pandas, pyarrow, Ray, vLLM, PEFT, and your RL runtime. Copy or overlay the files in `external/runtime_patch/` into that runtime, then set:

```bash
export RUNTIME_ROOT="<rl_runtime>"
export HF_MODELS="<hf_model_root>"
export HF_DATA="<hf_data_root>"
export HF_CACHE="<hf_cache_root>"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
```

Set `OPENAI_API_KEY` only for SkillBank or teacher-trajectory regeneration:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
```

## 2. Data

Use the released samples directly, or rebuild them:

```bash
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh
```

The scripts read dataset mirrors from `HF_DATA` and cache files from `HF_CACHE`.

## 3. SkillBank

The final bank is:

```bash
skill_bank/round_4_musique/outputs/final_skill_bank.md
```

To rebuild:

```bash
python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py --base-url "$OPENAI_BASE_URL"
python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py --base-url "$OPENAI_BASE_URL"
```

## 4. Teacher Trajectories

The released canonical file is:

```bash
teacher_trajectory/data/canonical_trajectories.jsonl
```

To regenerate a small run:

```bash
python teacher_trajectory/src/build_manifest.py \
  --output-dir teacher_trajectory/work/manifest \
  --train-datasets hotpotqa \
  --hotpot-count 20 \
  --nq-count 0 --triviaqa-count 0 --2wiki-count 0 --musique-count 0 --failure-count 0

python teacher_trajectory/src/run_teacher_rollout.py \
  --manifest-path teacher_trajectory/work/manifest/manifest.jsonl \
  --output-dir teacher_trajectory/work/rollout \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --base-url "$OPENAI_BASE_URL" \
  --retriever-host "$RETRIEVER_HOST" \
  --retriever-port "$RETRIEVER_PORT" \
  --max-examples 20 \
  --resume
```

## 5. SFT

Build or reuse the checked-in SFT data:

```bash
python supervised_finetuning/scripts/build_stage1_dataset.py \
  --input-path teacher_trajectory/data/canonical_trajectories.jsonl \
  --output-dir supervised_finetuning/data/stage1

python supervised_finetuning/scripts/build_stage2_dataset.py \
  --input-train supervised_finetuning/data/stage1/train.jsonl \
  --input-eval supervised_finetuning/data/stage1/eval.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir supervised_finetuning/data/stage2
```

Train and merge:

```bash
bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh

python supervised_finetuning/scripts/merge_lora.py \
  --base-model-path "$HF_MODELS/Qwen2.5-7B-Instruct" \
  --adapter-path supervised_finetuning/models/stage2 \
  --output-dir supervised_finetuning/models/stage2_7b_instruct_merged \
  --device cuda \
  --overwrite
```

## 6. RL

Build RL data:

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_data
```

Train:

```bash
export MODEL_PATH="$SEARCHSKILL_ROOT/supervised_finetuning/models/stage2_7b_instruct_merged"
bash reinforcement_learning/scripts/train_7b_instruct.sh
```

## 7. Test

Use the included benchmark splits:

```bash
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh nq
MODEL_PATH="<checkpoint_or_model>" BENCHMARK_SPLIT=full bash reinforcement_learning/scripts/evaluate_policy.sh all
```

`BENCHMARK_SPLIT` must be `dev` or `full`.
