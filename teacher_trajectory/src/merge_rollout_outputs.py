from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import dump_json, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple rollout output directories into one deduplicated run.")
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def choose_better(
    current: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    current_priority: int,
    candidate_priority: int,
) -> bool:
    current_passed = bool(current.get("validation", {}).get("passed"))
    candidate_passed = bool(candidate.get("validation", {}).get("passed"))
    if candidate_passed != current_passed:
        return candidate_passed

    current_runtime_error = bool(current.get("validation", {}).get("runtime_error"))
    candidate_runtime_error = bool(candidate.get("validation", {}).get("runtime_error"))
    if candidate_runtime_error != current_runtime_error:
        return not candidate_runtime_error

    if candidate_priority != current_priority:
        return candidate_priority < current_priority

    current_steps = len(current.get("steps", []))
    candidate_steps = len(candidate.get("steps", []))
    if candidate_steps != current_steps:
        return candidate_steps < current_steps

    current_outputs = len(current.get("raw_outputs", []))
    candidate_outputs = len(candidate.get("raw_outputs", []))
    return candidate_outputs > current_outputs


def build_summary(template: Dict[str, Any], rows: List[Dict[str, Any]], input_dirs: List[Path]) -> Dict[str, Any]:
    summary = {
        "manifest_path": template.get("manifest_path", ""),
        "skill_bank_path": template.get("skill_bank_path", ""),
        "model": template.get("model", ""),
        "base_url": template.get("base_url", ""),
        "total_requested": template.get("total_requested", len(rows)),
        "processed": 0,
        "passed": 0,
        "failed": 0,
        "runtime_errors": 0,
        "datasets": {},
        "merged_from": [str(path) for path in input_dirs],
    }
    for row in rows:
        dataset = row.get("dataset", "unknown")
        passed = bool(row.get("validation", {}).get("passed"))
        summary["processed"] += 1
        summary["datasets"].setdefault(dataset, {"processed": 0, "passed": 0})
        summary["datasets"][dataset]["processed"] += 1
        if passed:
            summary["passed"] += 1
            summary["datasets"][dataset]["passed"] += 1
        else:
            summary["failed"] += 1
    return summary


def main() -> None:
    args = parse_args()
    template_summary: Dict[str, Any] = {}
    chosen_rows: Dict[str, Tuple[Dict[str, Any], int, int]] = {}
    arrival_index = 0

    for priority, input_dir in enumerate(args.input_dirs):
        summary_path = input_dir / "run_summary.json"
        if not template_summary and summary_path.exists():
            template_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        raw_path = input_dir / "trajectories.raw.jsonl"
        if not raw_path.exists():
            continue

        for row in load_jsonl(raw_path):
            row_id = row["id"]
            if row_id not in chosen_rows:
                chosen_rows[row_id] = (row, priority, arrival_index)
                arrival_index += 1
                continue

            current_row, current_priority, current_arrival = chosen_rows[row_id]
            if choose_better(
                current=current_row,
                candidate=row,
                current_priority=current_priority,
                candidate_priority=priority,
            ):
                chosen_rows[row_id] = (row, priority, current_arrival)

    ordered_rows = [
        row
        for row, _, _ in sorted(
            chosen_rows.values(),
            key=lambda item: (item[1], item[2], item[0].get("dataset", ""), item[0]["id"]),
        )
    ]

    filtered_rows = [row for row in ordered_rows if row.get("validation", {}).get("passed")]
    failed_rows = [row for row in ordered_rows if not row.get("validation", {}).get("passed")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "trajectories.raw.jsonl", ordered_rows)
    dump_jsonl(args.output_dir / "trajectories.filtered.jsonl", filtered_rows)
    dump_jsonl(args.output_dir / "trajectories.failed.jsonl", failed_rows)
    dump_json(
        args.output_dir / "run_summary.json",
        build_summary(template_summary, ordered_rows, args.input_dirs),
    )


if __name__ == "__main__":
    main()
