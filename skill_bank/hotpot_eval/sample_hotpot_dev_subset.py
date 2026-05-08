#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT_PATH = Path("/path/to/hf_data/data/hotpotqa/test.jsonl")
DEFAULT_OUTPUT_PATH = Path(
    "eval/hotpot_b1_b2/data/hotpotqa_dev_sample200_seed42.jsonl"
)
DEFAULT_SUMMARY_PATH = Path(
    "eval/hotpot_b1_b2/data/hotpotqa_dev_sample200_seed42_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a fixed HotpotQA dev subset for B1/B2 evaluation.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if args.sample_size > len(rows):
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size {len(rows)}")

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, args.sample_size)

    dump_jsonl(args.output_path, sampled)
    dump_json(
        args.summary_path,
        {
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "sample_ids": [row.get("id") for row in sampled],
        },
    )
    print(f"Wrote {len(sampled)} rows to {args.output_path}")
    print(f"Wrote summary to {args.summary_path}")


if __name__ == "__main__":
    main()
