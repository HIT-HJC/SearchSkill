<div align="center">

<h1>SearchSkill: Teaching LLMs to Use Search Tools with Evolving Skill Banks</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2605.09038-b31b1b.svg)](https://arxiv.org/abs/2605.09038)
[![Models](https://img.shields.io/badge/Hugging%20Face-Models-ffcc4d.svg)](https://huggingface.co/HJCHJC)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#1-install)

[SearchSkill-SFT-7B-Instruct](https://huggingface.co/HJCHJC/SearchSkill-SFT-7B-Instruct) |
[SearchSkill-SFT-7B-Base](https://huggingface.co/HJCHJC/SearchSkill-SFT-7B-Base) |
[SearchSkill-RL-7B-Instruct-GRPO](https://huggingface.co/HJCHJC/SearchSkill-RL-7B-Instruct-GRPO) |
[SearchSkill-RL-7B-Base-GRPO](https://huggingface.co/HJCHJC/SearchSkill-RL-7B-Base-GRPO)

</div>

SearchSkill teaches language models to use search tools through an evolving SkillBank. This repository keeps the code and data needed to inspect the pipeline, rebuild training data, train SFT/RL policies, and evaluate on the included dev/full benchmark splits.

## Repository Layout

- `data_preparation/`: sampled training data, profiling reports, and sampling scripts.
- `skill_bank/`: seed bank, four evolution rounds, final SkillBank, and the policy eval script.
- `teacher_trajectory/`: teacher rollout code and released canonical trajectories.
- `supervised_finetuning/`: SFT data builders, LoRA training wrappers, and merge script.
- `reinforcement_learning/`: RL data builder, training wrappers, and benchmark evaluation launcher.
- `benchmarks/`: public `dev/` and `full/` test JSONL files.
- `external/runtime_patch/`: SearchSkill-specific files to overlay into a compatible VERL-style RL runtime.

## 1. Install

Create and activate a conda environment. Install the PyTorch build that matches your CUDA version, then install the project dependencies.

```bash
git clone https://github.com/HIT-HJC/SearchSkill.git
cd SearchSkill

conda create -n searchskill python=3.10 -y
conda activate searchskill

python -m pip install --upgrade pip
# Example for CUDA 12.1. Replace this line with the correct PyTorch command for your machine.
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Set common paths:

```bash
export SEARCHSKILL_ROOT="$(pwd)"
export ROOT="$SEARCHSKILL_ROOT"
export PYTHON_BIN="$(command -v python)"
export HF_MODELS="<directory_containing_Qwen2.5_models_and_e5-base-v2>"
export HF_DATA="<dataset_mirror_root>"
export HF_CACHE="<hf_cache_root>"
```

For evaluation-only use, install `requirements-eval.txt` instead of the full dependency file in the same environment after installing PyTorch:

```bash
python -m pip install -r requirements-eval.txt
```

## 2. Start The Retriever

SearchSkill uses an HTTP retriever endpoint during teacher rollout, RL training, and evaluation. Start it before any command that calls search.

```bash
export E5_INDEX_PATH="<retriever_index_dir>"
export E5_CORPUS_PATH="<retriever_corpus_jsonl>"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
export FAISS_GPU=0  # set to 1 for a GPU FAISS index

bash data_preparation/samples/trajectory_pruning/start_local_retriever.sh
```

The retriever corpus should contain either a `contents` field formatted as `title\ntext`, or separate `title` and `text` fields. The retriever wrapper loads `$HF_MODELS/e5-base-v2`; install a GPU-enabled FAISS build separately if you set `FAISS_GPU=1`. The evaluation launcher checks `/retrieve` before running.

Key retriever files:

- `data_preparation/samples/trajectory_pruning/start_local_retriever.sh`
- `external/runtime_patch/search_r1/search/retrieval_server.py`
- `external/runtime_patch/search_r1/search/retrieval.py`

## 3. Prepare Data

The repository already includes released samples and benchmark splits. To rebuild the sampled training data:

```bash
bash data_preparation/run_singlehop_sampling.sh
bash data_preparation/run_multihop_sampling.sh
```

Key code:

- `data_preparation/sample_singlehop_train.py`
- `data_preparation/sample_multihop_train.py`
- `data_preparation/samples/*/train_sample_*.jsonl`
- `benchmarks/dev/*.jsonl`
- `benchmarks/full/*.jsonl`

## 4. Build Or Reuse The SkillBank

The final released bank is:

```bash
skill_bank/round_4_musique/outputs/final_skill_bank.md
```

To rebuild the four public evolution rounds, set an OpenAI-compatible API endpoint and run:

```bash
export OPENAI_API_KEY="<your_key>"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

python skill_bank/round_1_singlehop/build_packets.py
python skill_bank/round_1_singlehop/run_b1_expand.py --base-url "$OPENAI_BASE_URL"

python skill_bank/round_2_hotpotqa/build_packets.py
python skill_bank/round_2_hotpotqa/run_b2_expand.py --base-url "$OPENAI_BASE_URL"

python skill_bank/round_3_2wiki/build_packets.py
python skill_bank/round_3_2wiki/run_b3_expand.py --base-url "$OPENAI_BASE_URL"

python skill_bank/round_4_musique/build_packets.py
python skill_bank/round_4_musique/run_b4_expand.py --base-url "$OPENAI_BASE_URL"
```

Key code:

- `skill_bank/inputs/seed_skill_bank.md`
- `skill_bank/round_*/build_packets.py`
- `skill_bank/round_*/run_b*_expand.py`
- `skill_bank/round_4_musique/outputs/final_skill_bank.md`

## 5. Build Teacher Trajectories

The released canonical trajectory file is:

```bash
teacher_trajectory/data/canonical_trajectories.jsonl
```

To run a small teacher rollout, keep the retriever server running and execute:

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

Key code:

- `teacher_trajectory/src/build_manifest.py`
- `teacher_trajectory/src/run_teacher_rollout.py`
- `teacher_trajectory/src/merge_rollout_outputs.py`
- `teacher_trajectory/src/build_canonical_teacher_set.py`
- `teacher_trajectory/src/pack_sft.py`

## 6. Train SFT

The checked-in SFT data can be used directly. To rebuild it from the released canonical trajectories:

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

Train the two-stage 7B instruct policy:

```bash
export HF_MODELS="<directory_containing_Qwen2.5_models>"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

bash supervised_finetuning/scripts/train_stage1_7b_instruct.sh
bash supervised_finetuning/scripts/train_stage2_7b_instruct.sh
```

Merge the stage-two LoRA adapter into a dense checkpoint:

```bash
python supervised_finetuning/scripts/merge_lora.py \
  --base-model-path "$HF_MODELS/Qwen2.5-7B-Instruct" \
  --adapter-path supervised_finetuning/models/stage2 \
  --output-dir supervised_finetuning/models/stage2_7b_instruct_merged \
  --device cuda \
  --overwrite
```

Other released wrappers are available for 3B/7B base and instruct backbones:

- `supervised_finetuning/scripts/train_stage1_*.sh`
- `supervised_finetuning/scripts/train_stage2_*.sh`
- `supervised_finetuning/scripts/train_lora.py`
- `supervised_finetuning/scripts/merge_lora.py`

## 7. Run RL

RL training requires a compatible VERL-style runtime. Overlay the runtime patch first:

```bash
cp -r external/runtime_patch/* "<rl_runtime>/"
export RUNTIME_ROOT="<rl_runtime>"
```

Build or reuse the released RL parquet data:

```bash
python reinforcement_learning/scripts/build_policy_dataset.py \
  --train-jsonl reinforcement_learning/source_data/policy_training_pool/train.jsonl \
  --dev-jsonl reinforcement_learning/source_data/policy_training_pool/dev.jsonl \
  --skill-bank-path skill_bank/round_4_musique/outputs/final_skill_bank.md \
  --output-dir reinforcement_learning/data/policy_7b_instruct
```

Start the retriever server, set the SFT checkpoint, and launch GRPO training:

```bash
export MODEL_PATH="$SEARCHSKILL_ROOT/supervised_finetuning/models/stage2_7b_instruct_merged"
export DATA_DIR="$SEARCHSKILL_ROOT/reinforcement_learning/data/policy_7b_instruct"
export RETRIEVER_HOST="127.0.0.1"
export RETRIEVER_PORT="8000"
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash reinforcement_learning/scripts/train_7b_instruct.sh
```

Key code:

- `reinforcement_learning/scripts/build_policy_dataset.py`
- `reinforcement_learning/scripts/launch_training.sh`
- `reinforcement_learning/scripts/train_*.sh`
- `external/runtime_patch/verl/trainer/main_ppo_searchskill.py`
- `external/runtime_patch/verl/utils/reward_score/searchskill.py`

## 8. Evaluate

You can evaluate either a local checkpoint or one of the released Hugging Face SearchSkill models. Keep the retriever server running, then run a dev split:

```bash
export MODEL_PATH="HJCHJC/SearchSkill-SFT-7B-Instruct"
export BENCHMARK_SPLIT=dev
export SHARD_COUNT=1
export GPU_IDS_CSV=0

bash reinforcement_learning/scripts/evaluate_policy.sh nq
```

Run all dev benchmarks:

```bash
MODEL_PATH="$MODEL_PATH" BENCHMARK_SPLIT=dev bash reinforcement_learning/scripts/evaluate_policy.sh all
```

Run full benchmarks:

```bash
MODEL_PATH="$MODEL_PATH" BENCHMARK_SPLIT=full bash reinforcement_learning/scripts/evaluate_policy.sh all
```

The evaluation output is written under `eval/<run_name>/`. The launcher supports `all`, `singlehop`, `multihop`, or a single dataset name: `nq`, `triviaqa`, `popqa`, `hotpotqa`, `2wiki`, `musique`, or `bamboogle`.

Key code:

- `reinforcement_learning/scripts/evaluate_policy.sh`
- `skill_bank/nq_eval/eval_nq_qwen_skillbank.py`
- `skill_bank/nq_eval/eval_common.py`
- `benchmarks/dev/*.jsonl`
- `benchmarks/full/*.jsonl`

## Notes

- Model weights are hosted at [https://huggingface.co/HJCHJC](https://huggingface.co/HJCHJC).
- Retrieval indexes and corpus files are external resources and are not included in this repository.
- API keys, caches, logs, local paths, and raw experiment outputs are intentionally excluded.
- For a more verbose step-by-step reproduction guide, see [REPRODUCE.md](REPRODUCE.md).
