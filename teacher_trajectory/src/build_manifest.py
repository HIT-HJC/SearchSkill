from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common import (
    balanced_take,
    build_metadata_summary,
    build_task_family,
    classify_failure_error,
    dump_json,
    dump_jsonl,
    load_jsonl,
    load_skill_ids,
    repo_root,
    suggest_primary_skills,
    suggest_support_skills,
)


TRAIN_DATASETS = ["nq", "triviaqa", "hotpotqa", "2wiki", "musique"]
FAILURE_DATASETS = ["nq", "hotpotqa", "2wiki", "musique"]


def parse_dataset_list(value: str, allowed: Iterable[str]) -> List[str]:
    allowed_list = list(allowed)
    allowed_set = set(allowed_list)
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        return []
    for item in raw_items:
        if item not in allowed_set:
            raise argparse.ArgumentTypeError(f"Unknown dataset '{item}'. Allowed: {', '.join(allowed_list)}")
    return raw_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest for GPT-5.4 trajectory synthesis.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skill-bank-path",
        type=Path,
        default=repo_root() / "skill_bank" / "round_4_musique" / "outputs" / "final_skill_bank.md",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nq-count", type=int, default=32)
    parser.add_argument("--triviaqa-count", type=int, default=32)
    parser.add_argument("--hotpot-count", type=int, default=48)
    parser.add_argument("--2wiki-count", dest="two_wiki_count", type=int, default=48)
    parser.add_argument("--musique-count", type=int, default=48)
    parser.add_argument("--failure-count", type=int, default=8, help="Per-dataset failure replay count.")
    parser.add_argument("--use-full-multihop", action="store_true")
    parser.add_argument(
        "--train-datasets",
        type=lambda value: parse_dataset_list(value, TRAIN_DATASETS),
        default=",".join(TRAIN_DATASETS),
        help="Comma-separated subset of training datasets to include.",
    )
    parser.add_argument(
        "--failure-datasets",
        type=lambda value: parse_dataset_list(value, FAILURE_DATASETS),
        default=",".join(FAILURE_DATASETS),
        help="Comma-separated subset of failure replay datasets to include.",
    )
    args = parser.parse_args()
    if isinstance(args.train_datasets, str):
        args.train_datasets = parse_dataset_list(args.train_datasets, TRAIN_DATASETS)
    if isinstance(args.failure_datasets, str):
        args.failure_datasets = parse_dataset_list(args.failure_datasets, FAILURE_DATASETS)
    return args


def sample_pair_paths(dataset: str, use_full_multihop: bool) -> Tuple[Path, Path]:
    root = repo_root() / "data_preparation" / "samples"
    if dataset in {"hotpotqa", "2wiki", "musique"} and not use_full_multihop:
        base = root / "trajectory_pruning" / dataset
    else:
        base = root / dataset
    return base / "train_sample_light.jsonl", base / "train_sample_full.jsonl"


def load_training_examples(dataset: str, count: int, seed: int, use_full_multihop: bool, legal_skill_ids: List[str]) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    light_path, full_path = sample_pair_paths(dataset, use_full_multihop)
    light_by_id = {row["id"]: row for row in load_jsonl(light_path)}
    records: List[Dict[str, Any]] = []
    for full_row in load_jsonl(full_path):
        merged: Dict[str, Any] = {}
        merged.update(light_by_id.get(full_row["id"], {}))
        merged.update(full_row)
        merged["dataset"] = dataset
        merged["sample_origin"] = "train_pruned" if "trajectory_pruning" in str(full_path) else "train_full"
        merged["task_family"] = build_task_family(dataset)
        merged["gold_answers"] = full_row.get("golden_answers", [])
        merged["metadata_summary"] = build_metadata_summary(merged)
        merged["candidate_primary_skills"] = suggest_primary_skills(merged, legal_skill_ids)
        merged["suggested_support_skills"] = suggest_support_skills(legal_skill_ids)
        records.append(merged)
    return balanced_take(records, count, seed)


def load_failure_examples(dataset: str, count: int, seed: int, legal_skill_ids: List[str]) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    trace_map = {
        "nq": repo_root() / "eval" / "nq_b0_b1" / "b1" / "nq_b1_trace.jsonl",
        "hotpotqa": repo_root() / "eval" / "hotpot_b1_b2" / "b2" / "hotpot_b2_trace.jsonl",
        "2wiki": repo_root() / "eval" / "2wiki_b2_b3_routerfix_v1" / "b3" / "2wiki_b3_trace.jsonl",
        "musique": repo_root() / "eval" / "musique_b3_b4_routerfix_v1" / "b4" / "musique_b4_trace.jsonl",
    }
    path = trace_map[dataset]
    records: List[Dict[str, Any]] = []
    for row in load_jsonl(path):
        if row.get("em") == 1:
            continue
        selected_skills: List[str] = []
        for step in row.get("steps", []):
            for skill in step.get("selected_skills", []) or []:
                if skill and skill not in selected_skills:
                    selected_skills.append(skill)
        record = {
            "id": f"{dataset}_failure_{row.get('id')}",
            "source_example_id": row.get("id"),
            "dataset": dataset,
            "sample_origin": "failure_replay",
            "task_family": build_task_family(dataset),
            "question": row.get("question"),
            "gold_answers": row.get("gold", []),
            "metadata": {},
            "failure_info": {
                "previous_prediction": row.get("prediction"),
                "wrong_skills": selected_skills,
                "error_tag": classify_failure_error(str(row.get("question", "")), str(row.get("prediction", "")), selected_skills),
                "trace_path": str(path),
            },
        }
        record["metadata_summary"] = build_metadata_summary(record)
        record["candidate_primary_skills"] = suggest_primary_skills(record, legal_skill_ids)
        record["suggested_support_skills"] = suggest_support_skills(legal_skill_ids)
        records.append(record)
    return balanced_take(records, count, seed + 17)


def main() -> None:
    args = parse_args()
    legal_skill_ids = load_skill_ids(args.skill_bank_path)
    manifest: List[Dict[str, Any]] = []

    counts = [
        ("nq", args.nq_count),
        ("triviaqa", args.triviaqa_count),
        ("hotpotqa", args.hotpot_count),
        ("2wiki", args.two_wiki_count),
        ("musique", args.musique_count),
    ]
    for dataset, count in counts:
        if dataset not in args.train_datasets:
            continue
        manifest.extend(load_training_examples(dataset, count, args.seed, args.use_full_multihop, legal_skill_ids))

    for dataset in FAILURE_DATASETS:
        if dataset not in args.failure_datasets:
            continue
        manifest.extend(load_failure_examples(dataset, args.failure_count, args.seed, legal_skill_ids))

    output_dir = args.output_dir
    dump_jsonl(output_dir / "manifest.jsonl", manifest)
    summary = {
        "skill_bank_path": str(args.skill_bank_path),
        "total_examples": len(manifest),
        "train_datasets": args.train_datasets,
        "failure_datasets": args.failure_datasets,
        "dataset_counts": {},
        "sample_origin_counts": {},
    }
    for row in manifest:
        summary["dataset_counts"][row["dataset"]] = summary["dataset_counts"].get(row["dataset"], 0) + 1
        summary["sample_origin_counts"][row["sample_origin"]] = summary["sample_origin_counts"].get(row["sample_origin"], 0) + 1
    dump_json(output_dir / "manifest_summary.json", summary)


if __name__ == "__main__":
    main()
