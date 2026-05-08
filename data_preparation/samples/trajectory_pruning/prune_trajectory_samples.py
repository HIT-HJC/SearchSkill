#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = Path("outputs/data_preparation/samples")
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hotpotqa": {
        "cap_per_signature": 4,
        "protect_signature_freq_leq": 3,
    },
    "2wiki": {
        "cap_per_signature": 10,
        "protect_signature_freq_leq": 3,
    },
    "musique": {
        "cap_per_signature": 4,
        "protect_signature_freq_leq": 3,
    },
}


@dataclass
class SampleRow:
    dataset: str
    index: int
    sample_id: str
    question: str
    light: Dict[str, Any]
    full: Dict[str, Any]
    signature: str
    labels: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune sampled multihop train subsets for trajectory construction.")
    parser.add_argument("--root-dir", type=str, default=str(ROOT_DIR))
    parser.add_argument("--out-dir-name", type=str, default="trajectory_pruning")
    parser.add_argument("--datasets", type=str, default="hotpotqa,2wiki,musique")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_dataset_names(raw: str) -> List[str]:
    names = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {name}")
        names.append(name)
    if not names:
        raise ValueError("At least one dataset is required.")
    return names


def build_labels(light: Dict[str, Any]) -> List[str]:
    labels = [
        f"native:{light.get('native_type', 'unknown')}",
        f"hop:{light.get('hop_count', 'unknown')}",
        f"entity:{light.get('entity_bin', 'unknown')}",
        f"wh:{light.get('wh_word', 'unknown')}",
    ]
    for flag in light.get("flags", []):
        labels.append(f"flag:{flag}")
    return labels


def load_samples(root_dir: Path, dataset: str) -> List[SampleRow]:
    dataset_dir = root_dir / dataset
    light_rows = load_jsonl(dataset_dir / "train_sample_light.jsonl")
    full_rows = load_jsonl(dataset_dir / "train_sample_full.jsonl")
    if len(light_rows) != len(full_rows):
        raise ValueError(f"{dataset}: light/full row count mismatch: {len(light_rows)} vs {len(full_rows)}")

    full_by_id: Dict[str, Dict[str, Any]] = {}
    for full in full_rows:
        full_id = str(full.get("id", ""))
        if not full_id:
            raise ValueError(f"{dataset}: encountered full row without id")
        if full_id in full_by_id:
            raise ValueError(f"{dataset}: duplicate id in full rows: {full_id}")
        full_by_id[full_id] = full

    samples: List[SampleRow] = []
    for index, light in enumerate(light_rows):
        sample_id = str(light.get("id", index))
        full = full_by_id.get(sample_id)
        if full is None:
            raise ValueError(f"{dataset}: id not found in full rows: {sample_id}")
        question = str(full.get("question", light.get("question", ""))).strip()
        light_question = str(light.get("question", question)).strip()
        if light_question != question:
            raise ValueError(f"{dataset}: question mismatch for id {sample_id}")
        signature = str(light.get("signature", ""))
        samples.append(
            SampleRow(
                dataset=dataset,
                index=index,
                sample_id=sample_id,
                question=question,
                light=light,
                full=full,
                signature=signature,
                labels=build_labels(light),
            )
        )
    return samples


def compute_rare_label_threshold(total: int) -> int:
    return max(24, math.ceil(total * 0.003))


def score_sample(row: SampleRow, label_counts: Dict[str, int]) -> Tuple[float, int, int, str]:
    rarity_score = sum(1.0 / label_counts[label] for label in row.labels if label_counts[label] > 0)
    flag_bonus = len(row.light.get("flags", []))
    question_len = len(row.question.split())
    return (-rarity_score, -flag_bonus, -question_len, row.sample_id)


def prune_dataset(samples: Sequence[SampleRow], config: Dict[str, Any]) -> Tuple[List[SampleRow], Dict[str, Any]]:
    total = len(samples)
    signature_counts = Counter(row.signature for row in samples)
    label_counts = Counter(label for row in samples for label in row.labels)
    rare_label_threshold = compute_rare_label_threshold(total)

    protected_indices = set()
    protection_reasons: Dict[int, List[str]] = defaultdict(list)

    for row in samples:
        if signature_counts[row.signature] <= int(config["protect_signature_freq_leq"]):
            protected_indices.add(row.index)
            protection_reasons[row.index].append("low_freq_signature")
        rare_hits = [label for label in row.labels if label_counts[label] <= rare_label_threshold]
        if rare_hits:
            protected_indices.add(row.index)
            protection_reasons[row.index].append("rare_label")

    by_signature: Dict[str, List[SampleRow]] = defaultdict(list)
    for row in samples:
        by_signature[row.signature].append(row)

    selected_indices = set(protected_indices)
    cap_per_signature = int(config["cap_per_signature"])
    signature_selection_stats: Dict[str, Dict[str, int]] = {}

    for signature, rows in by_signature.items():
        protected_rows = [row for row in rows if row.index in protected_indices]
        unprotected_rows = [row for row in rows if row.index not in protected_indices]
        slots = max(cap_per_signature, len(protected_rows))
        if unprotected_rows and len(protected_rows) < slots:
            ranked = sorted(unprotected_rows, key=lambda row: score_sample(row, label_counts))
            for row in ranked[: slots - len(protected_rows)]:
                selected_indices.add(row.index)
        signature_selection_stats[signature] = {
            "original": len(rows),
            "protected": len(protected_rows),
            "kept": sum(1 for row in rows if row.index in selected_indices),
        }

    selected = [row for row in samples if row.index in selected_indices]

    kept_label_counts = Counter(label for row in selected for label in row.labels)
    kept_signature_counts = Counter(row.signature for row in selected)

    report = {
        "original_size": total,
        "kept_size": len(selected),
        "kept_ratio": len(selected) / total if total else 0.0,
        "unique_signatures_original": len(signature_counts),
        "unique_signatures_kept": len(kept_signature_counts),
        "protect_signature_freq_leq": int(config["protect_signature_freq_leq"]),
        "cap_per_signature": cap_per_signature,
        "rare_label_threshold": rare_label_threshold,
        "protected_examples": len(protected_indices),
        "protected_by_reason": {
            "low_freq_signature": sum("low_freq_signature" in reasons for reasons in protection_reasons.values()),
            "rare_label": sum("rare_label" in reasons for reasons in protection_reasons.values()),
        },
        "native_type_original": dict(Counter(row.light.get("native_type", "unknown") for row in samples)),
        "native_type_kept": dict(Counter(row.light.get("native_type", "unknown") for row in selected)),
        "hop_original": dict(Counter(str(row.light.get("hop_count", "unknown")) for row in samples)),
        "hop_kept": dict(Counter(str(row.light.get("hop_count", "unknown")) for row in selected)),
        "entity_original": dict(Counter(row.light.get("entity_bin", "unknown") for row in samples)),
        "entity_kept": dict(Counter(row.light.get("entity_bin", "unknown") for row in selected)),
        "top_flags_original": dict(Counter(flag for row in samples for flag in row.light.get("flags", [])).most_common(20)),
        "top_flags_kept": dict(Counter(flag for row in selected for flag in row.light.get("flags", [])).most_common(20)),
        "rare_labels_preserved": {
            label: {
                "original": count,
                "kept": kept_label_counts.get(label, 0),
            }
            for label, count in sorted(label_counts.items())
            if count <= rare_label_threshold
        },
        "high_freq_signature_examples": [
            {
                "signature": signature,
                **stats,
            }
            for signature, stats in sorted(
                signature_selection_stats.items(),
                key=lambda item: (-item[1]["original"], item[0]),
            )[:20]
        ],
    }
    return selected, report


def build_output_records(selected: Sequence[SampleRow]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    light_records = [row.light for row in selected]
    full_records = [row.full for row in selected]
    return light_records, full_records


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_root = root_dir / args.out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "root_dir": str(root_dir),
        "out_root": str(out_root),
        "datasets": {},
    }

    for dataset in parse_dataset_names(args.datasets):
        samples = load_samples(root_dir, dataset)
        selected, report = prune_dataset(samples, DATASET_CONFIGS[dataset])
        light_records, full_records = build_output_records(selected)

        dataset_dir = out_root / dataset
        dump_jsonl(dataset_dir / "train_sample_light.jsonl", light_records)
        dump_jsonl(dataset_dir / "train_sample_full.jsonl", full_records)
        dump_json(dataset_dir / "pruning_report.json", report)

        summary["datasets"][dataset] = {
            "original_size": report["original_size"],
            "kept_size": report["kept_size"],
            "kept_ratio": report["kept_ratio"],
            "cap_per_signature": report["cap_per_signature"],
            "rare_label_threshold": report["rare_label_threshold"],
        }

    dump_json(out_root / "pruning_summary.json", summary)


if __name__ == "__main__":
    main()
