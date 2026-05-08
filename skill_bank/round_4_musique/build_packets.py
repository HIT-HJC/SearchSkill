#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CONFIG = Path(
    "skill_bank/round_4_musique/config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped MuSiQue packets for final SkillBank evolution."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=3,
        help="Maximum representative examples to keep in each grouped packet.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_group_id(dataset: str, signature: str) -> str:
    digest = hashlib.sha1(f"{dataset}::{signature}".encode("utf-8")).hexdigest()[:12]
    return f"{dataset}_sig_{digest}"


def select_examples(
    rows: List[Dict[str, Any]],
    full_by_id: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=lambda item: item.get("id", ""))
    selected: List[Dict[str, Any]] = []
    seen_questions = set()
    for row in rows:
        question = row.get("question", "").strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        full_row = full_by_id.get(row.get("id", ""), {})
        metadata = full_row.get("metadata", {})
        selected.append(
            {
                "id": row.get("id"),
                "question": question,
                "primary_answer": row.get("primary_answer"),
                "golden_answers": full_row.get("golden_answers", []),
                "flags": row.get("flags", []),
                "native_type": row.get("native_type"),
                "hop_count": row.get("hop_count"),
                "wh_word": row.get("wh_word"),
                "answer_form_hint": row.get("answer_form_hint"),
                "entity_bin": row.get("entity_bin"),
                "token_bin": row.get("token_bin"),
                "metadata_type": metadata.get("type"),
                "metadata_level": metadata.get("level"),
                "supporting_titles": metadata.get("supporting_facts", {}).get("title", [])[:4],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def build_packets(
    config: Dict[str, Any], examples_per_group: int
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    samples_root = Path(config["samples_root"])
    datasets = list(config["datasets"])
    packets: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "round_name": config["round_name"],
        "bank_name": config["bank_name"],
        "datasets": {},
        "total_groups": 0,
        "total_examples": 0,
        "examples_per_group": examples_per_group,
    }

    for dataset in datasets:
        light_path = samples_root / dataset / "train_sample_light.jsonl"
        full_path = samples_root / dataset / "train_sample_full.jsonl"

        full_by_id = {row["id"]: row for row in load_jsonl(full_path)}
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in load_jsonl(light_path):
            groups[row["signature"]].append(row)

        dataset_group_count = 0
        dataset_example_count = 0
        for signature, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            first = rows[0]
            examples = select_examples(rows, full_by_id, limit=examples_per_group)
            packet = {
                "group_id": stable_group_id(dataset, signature),
                "dataset": dataset,
                "signature": signature,
                "group_size": len(rows),
                "profile": {
                    "native_type": first.get("native_type"),
                    "hop_count": first.get("hop_count"),
                    "wh_word": first.get("wh_word"),
                    "answer_form_hint": first.get("answer_form_hint"),
                    "entity_bin": first.get("entity_bin"),
                    "token_bin": first.get("token_bin"),
                    "flags": first.get("flags", []),
                    "native_summary": first.get("native_summary", {}),
                },
                "representative_examples": examples,
            }
            packets.append(packet)
            dataset_group_count += 1
            dataset_example_count += len(examples)

        summary["datasets"][dataset] = {
            "group_count": dataset_group_count,
            "representative_examples": dataset_example_count,
            "light_path": str(light_path),
            "full_path": str(full_path),
        }
        summary["total_groups"] += dataset_group_count
        summary["total_examples"] += dataset_example_count

    return packets, summary


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    round_dir = args.config.parent
    packets, summary = build_packets(config, examples_per_group=args.examples_per_group)

    packets_path = round_dir / "artifacts" / "skill_discovery_packets.jsonl"
    summary_path = round_dir / "artifacts" / "skill_discovery_packets_summary.json"

    packet_rows = dump_jsonl(packets_path, packets)
    dump_json(summary_path, summary)

    print(f"Wrote {packet_rows} grouped packets to {packets_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
