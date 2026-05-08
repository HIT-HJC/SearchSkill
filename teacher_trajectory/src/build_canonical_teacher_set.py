from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import dump_json, dump_jsonl, load_jsonl


DEFAULT_INPUT_SPECS = [
    "stage2_hotpotqa=teacher_trajectory/runs/multi_hop_teacher/hotpotqa/teacher_run",
    "stage2_2wiki=teacher_trajectory/runs/multi_hop_teacher/2wiki/teacher_run",
    "stage2_musique=teacher_trajectory/runs/multi_hop_teacher/musique/teacher_run",
    "stage2_nq=teacher_trajectory/runs/single_hop_teacher/nq/teacher_run",
    "stage2_triviaqa=teacher_trajectory/runs/single_hop_teacher/triviaqa/teacher_run",
    "pilot_final=teacher_trajectory/runs/pilot_public_v1/teacher_run_final",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical teacher trajectory set from pilot plus stage2 rollout outputs."
    )
    parser.add_argument(
        "--input-spec",
        action="append",
        dest="input_specs",
        default=[],
        help="Input source as label=<teacher_run_dir>. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_input_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --input-spec {spec!r}; expected label=/abs/path")
    label, path_text = spec.split("=", 1)
    label = label.strip()
    path = Path(path_text.strip())
    if not label:
        raise ValueError(f"Invalid --input-spec {spec!r}; missing label")
    return label, path


def record_key(row: Dict[str, Any]) -> str:
    return f"{row.get('dataset', '')}:{row.get('id', '')}"


def clone_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row)


def validation(row: Dict[str, Any]) -> Dict[str, Any]:
    data = row.get("validation") or {}
    return data if isinstance(data, dict) else {}


def count_retrieved_steps(row: Dict[str, Any]) -> int:
    count = 0
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).lower()
        if action in {"search", "verify"} and str(step.get("retrieved", "")).strip():
            count += 1
    return count


def build_quality_tuple(row: Dict[str, Any], source_priority: int) -> Tuple[int, ...]:
    v = validation(row)
    steps = row.get("steps") or []
    final_nonempty = 1 if str(row.get("final_answer", "")).strip() else 0
    no_unknown_skills = 1 if not v.get("unknown_skills") else 0
    no_support_only_primary = 1 if not v.get("support_only_primary_steps") else 0
    route_ok = 1 if v.get("route_matches_candidates") is True else 0
    return (
        1 if v.get("passed") is True else 0,
        1 if not v.get("runtime_error") else 0,
        final_nonempty,
        1 if steps else 0,
        no_unknown_skills,
        no_support_only_primary,
        route_ok,
        -source_priority,
        count_retrieved_steps(row),
        len(steps),
        len(row.get("raw_outputs") or []),
    )


def choose_better_row(
    current: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    current_priority: int,
    candidate_priority: int,
) -> bool:
    return build_quality_tuple(candidate, candidate_priority) > build_quality_tuple(current, current_priority)


def load_latest_rows(input_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_path = input_dir / "trajectories.raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw trajectory file: {raw_path}")

    latest_rows: Dict[str, Dict[str, Any]] = {}
    raw_rows = 0
    for row in load_jsonl(raw_path):
        raw_rows += 1
        latest_rows[record_key(row)] = clone_row(row)

    deduped_rows = sorted(latest_rows.values(), key=lambda row: (row.get("dataset", ""), row.get("id", "")))
    duplicate_rows = raw_rows - len(deduped_rows)
    passed = sum(1 for row in deduped_rows if validation(row).get("passed") is True)
    runtime_error = sum(1 for row in deduped_rows if validation(row).get("runtime_error") is True)
    source_stats = {
        "input_dir": str(input_dir),
        "raw_rows": raw_rows,
        "latest_unique_rows": len(deduped_rows),
        "duplicate_rows": duplicate_rows,
        "passed_latest_rows": passed,
        "failed_latest_rows": len(deduped_rows) - passed,
        "runtime_error_latest_rows": runtime_error,
    }
    return deduped_rows, source_stats


def summarize_rows(rows: Sequence[Dict[str, Any]], *, source_labels: Iterable[str] | None = None) -> Dict[str, Any]:
    summary = {
        "total_requested": len(rows),
        "processed": len(rows),
        "passed": 0,
        "failed": 0,
        "runtime_errors": 0,
        "datasets": {},
    }
    if source_labels is not None:
        summary["input_sources"] = list(source_labels)
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        v = validation(row)
        passed = v.get("passed") is True
        runtime_error = v.get("runtime_error") is True
        summary["datasets"].setdefault(dataset, {"processed": 0, "passed": 0})
        summary["datasets"][dataset]["processed"] += 1
        if passed:
            summary["passed"] += 1
            summary["datasets"][dataset]["passed"] += 1
        else:
            summary["failed"] += 1
        if runtime_error:
            summary["runtime_errors"] += 1
    return summary


def main() -> None:
    args = parse_args()
    input_specs = args.input_specs or DEFAULT_INPUT_SPECS
    parsed_specs = [parse_input_spec(spec) for spec in input_specs]

    chosen_rows: Dict[str, Tuple[Dict[str, Any], int, str]] = {}
    source_stats: Dict[str, Dict[str, Any]] = {}
    overlap_decisions: List[Dict[str, Any]] = []

    for source_priority, (label, input_dir) in enumerate(parsed_specs):
        deduped_rows, stats = load_latest_rows(input_dir)
        source_stats[label] = stats
        for row in deduped_rows:
            key = record_key(row)
            row = clone_row(row)
            row["canonical_source_label"] = label
            row["canonical_source_dir"] = str(input_dir)
            if key not in chosen_rows:
                chosen_rows[key] = (row, source_priority, label)
                continue

            current_row, current_priority, current_label = chosen_rows[key]
            if choose_better_row(
                current=current_row,
                candidate=row,
                current_priority=current_priority,
                candidate_priority=source_priority,
            ):
                overlap_decisions.append(
                    {
                        "record_key": key,
                        "kept_source": label,
                        "replaced_source": current_label,
                        "kept_passed": validation(row).get("passed") is True,
                        "replaced_passed": validation(current_row).get("passed") is True,
                    }
                )
                chosen_rows[key] = (row, source_priority, label)
            else:
                overlap_decisions.append(
                    {
                        "record_key": key,
                        "kept_source": current_label,
                        "replaced_source": label,
                        "kept_passed": validation(current_row).get("passed") is True,
                        "replaced_passed": validation(row).get("passed") is True,
                    }
                )

    canonical_rows = [
        row
        for row, _, _ in sorted(
            chosen_rows.values(),
            key=lambda item: (item[0].get("dataset", ""), item[0].get("id", "")),
        )
    ]
    filtered_rows = [row for row in canonical_rows if validation(row).get("passed") is True]
    failed_rows = [row for row in canonical_rows if validation(row).get("passed") is not True]

    all_dir = args.output_dir / "all"
    by_dataset_root = args.output_dir / "by_dataset"
    all_dir.mkdir(parents=True, exist_ok=True)
    by_dataset_root.mkdir(parents=True, exist_ok=True)

    dump_jsonl(all_dir / "trajectories.raw.jsonl", canonical_rows)
    dump_jsonl(all_dir / "trajectories.filtered.jsonl", filtered_rows)
    dump_jsonl(all_dir / "trajectories.failed.jsonl", failed_rows)
    dump_json(
        all_dir / "run_summary.json",
        {
            "canonical": True,
            "output_scope": "all",
            **summarize_rows(canonical_rows, source_labels=[label for label, _ in parsed_specs]),
        },
    )

    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        rows_by_dataset[str(row.get("dataset", "unknown"))].append(row)

    for dataset, rows in sorted(rows_by_dataset.items()):
        dataset_dir = by_dataset_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_filtered = [row for row in rows if validation(row).get("passed") is True]
        dataset_failed = [row for row in rows if validation(row).get("passed") is not True]
        dump_jsonl(dataset_dir / "trajectories.raw.jsonl", rows)
        dump_jsonl(dataset_dir / "trajectories.filtered.jsonl", dataset_filtered)
        dump_jsonl(dataset_dir / "trajectories.failed.jsonl", dataset_failed)
        dump_json(
            dataset_dir / "run_summary.json",
            {
                "canonical": True,
                "output_scope": dataset,
                **summarize_rows(rows),
            },
        )

    overlap_counter = Counter(
        (item["kept_source"], item["replaced_source"])
        for item in overlap_decisions
    )
    source_counts = Counter(str(row.get("canonical_source_label", "unknown")) for row in canonical_rows)
    dataset_source_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in canonical_rows:
        dataset_source_counts[str(row.get("dataset", "unknown"))][str(row.get("canonical_source_label", "unknown"))] += 1

    dump_json(
        args.output_dir / "canonical_summary.json",
        {
            "canonical": True,
            "output_dir": str(args.output_dir),
            "input_sources": {label: str(path) for label, path in parsed_specs},
            "source_stats": source_stats,
            "all_summary": summarize_rows(canonical_rows, source_labels=[label for label, _ in parsed_specs]),
            "chosen_source_counts": dict(source_counts),
            "chosen_source_counts_by_dataset": {
                dataset: dict(counter)
                for dataset, counter in sorted(dataset_source_counts.items())
            },
            "overlap_record_count": len(overlap_decisions),
            "overlap_decision_counts": {
                f"{kept}<--{replaced}": count
                for (kept, replaced), count in sorted(overlap_counter.items())
            },
        },
    )


if __name__ == "__main__":
    main()
