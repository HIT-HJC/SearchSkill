from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import dump_json, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack filtered teacher trajectories into SFT message format.")
    parser.add_argument("--filtered-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    intro = (
        f"Question: {row['question']}\n"
        f"Dataset: {row['dataset']}\n"
        f"Candidate primary skills: {', '.join(row.get('candidate_primary_skills', []))}\n"
        "Return the next structured action as JSON."
    )
    messages.append({"role": "user", "content": intro})
    for step in row.get("steps", []):
        assistant_text = (
            "{\n"
            f'  "primary_skill": "{step.get("primary_skill", "")}",\n'
            f'  "support_skills": {step.get("support_skills", [])},\n'
            f'  "action": "{step.get("action", "")}",\n'
            f'  "query": "{step.get("query", "")}",\n'
            f'  "draft_answer": "{step.get("draft_answer", "")}"\n'
            "}"
        )
        messages.append({"role": "assistant", "content": assistant_text})
        retrieved = step.get("retrieved", "")
        if retrieved:
            messages.append(
                {
                    "role": "user",
                    "content": "Retriever results:\n" + retrieved[:8000] + "\nReturn the next structured action as JSON.",
                }
            )
    return messages


def main() -> None:
    args = parse_args()
    packed: List[Dict[str, Any]] = []
    for row in load_jsonl(args.filtered_path):
        packed.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "question": row["question"],
                "gold_answers": row.get("gold_answers", []),
                "messages": build_messages(row),
                "final_answer": row.get("final_answer", ""),
                "first_primary_skill": row.get("steps", [{}])[0].get("primary_skill", "") if row.get("steps") else "",
            }
        )
    dump_jsonl(args.output_path, packed)
    dump_json(
        args.output_path.with_suffix(".summary.json"),
        {"total_examples": len(packed), "output_path": str(args.output_path)},
    )


if __name__ == "__main__":
    main()
