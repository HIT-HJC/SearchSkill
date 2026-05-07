#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_SKILL_BANK = Path(
    "/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_4_musique/outputs/final_skill_bank.md"
)

SYSTEM_PROMPT = (
    "You are a SearchSkill policy model using the final SkillBank in two phases. "
    "In the skill-selection phase, output only <select_skill>skill-id</select_skill> "
    "or <select_skill>skill-id|skill-id</select_skill>, then stop. "
    "After the selected skill cards are provided, output exactly "
    "<skill>the-same-skill-ids</skill> followed by exactly one action tag, either "
    "<search>query</search> or <answer>span</answer>, and stop immediately after "
    "the closing action tag. Do not output explanations, markdown, natural-language "
    "tool descriptions, or <information> by yourself. Answer as soon as the evidence "
    "is sufficient."
)

SELECTION_INSTRUCTION = (
    "Selection phase: choose 1 to 3 skill ids from the the final SkillBank index that should "
    "govern the next action. If the evidence is sufficient, choose a closure skill "
    "such as verbatim-evidence-span and/or answer-grounding-check. Output only "
    "<select_skill>skill-id</select_skill> or <select_skill>skill-id|skill-id</select_skill>. "
    "Do not search or answer in this turn."
)

INITIAL_PROMPT = (
    "Question: {question}\n"
    "Suggested search budget: {search_budget}\n"
    "Easy questions usually finish in 2-3 searches; harder chain or comparison questions may need 4-5.\n"
    "For A-or-B comparison questions, keep the explicit options as anchors; the final answer should be one of those options.\n"
    "For bridge questions, do not answer with an intermediate entity copied from the question unless it is the requested final attribute.\n"
    "Do not repeat the same entity-attribute pair.\n"
    "If the answer span is already explicit in the evidence, select a closure skill and answer immediately.\n\n"
    "{skill_index}\n\n"
    "{selection_instruction}"
)

SKILL_HEADER_RE = re.compile(r"^`([a-z0-9][a-z0-9\-]*)`\s*$")


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars].rstrip() + "..."
    return value


def parse_skill_bank(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    current_id = ""
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = SKILL_HEADER_RE.match(raw_line.strip())
        if match:
            if current_id:
                entries[current_id] = " ".join(item.strip() for item in current_lines if item.strip())
            current_id = match.group(1)
            current_lines = []
        elif current_id:
            current_lines.append(raw_line)
    if current_id:
        entries[current_id] = " ".join(item.strip() for item in current_lines if item.strip())
    if not entries:
        raise RuntimeError(f"No SkillBank entries parsed from {path}")
    return entries


def build_skill_index(entries: dict[str, str], max_desc_chars: int) -> str:
    lines = ["Available the final SkillBank index:"]
    for skill_id, desc in entries.items():
        lines.append(f"- {skill_id}: {compact_text(desc, max_desc_chars)}")
    return "\n".join(lines)


def estimate_budget(row: dict[str, Any]) -> int:
    raw = row.get("search_budget") or row.get("max_steps") or 0
    if raw:
        return min(max(1, int(raw)), 5)
    dataset = str(row.get("dataset") or "")
    question = str(row.get("question") or "").lower()
    if dataset in {"nq", "triviaqa", "popqa"}:
        return 2 if len(question.split()) <= 12 else 3
    score = 3
    if any(key in question for key in ("same", "both", "older", "younger", "earlier", "later", " or ")):
        score += 1
    if len(question.split()) >= 18:
        score += 1
    return min(score, 5)


def normalize_dataset_name(name: str) -> str:
    if name == "2wiki":
        return "2wikimultihopqa"
    return name or "unknown"


def convert_split(input_path: Path, output_path: Path, skill_index: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    for idx, row in enumerate(load_jsonl(input_path)):
        question = str(row.get("question") or "").strip()
        if question and not question.endswith("?"):
            question += "?"
        gold = [str(item) for item in (row.get("gold_answers") or row.get("golden_answers") or []) if str(item).strip()]
        dataset = normalize_dataset_name(str(row.get("dataset") or ""))
        budget = estimate_budget(row)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INITIAL_PROMPT.format(
                    question=question,
                    search_budget=budget,
                    skill_index=skill_index,
                    selection_instruction=SELECTION_INSTRUCTION,
                ),
            },
        ]
        rows.append(
            {
                "id": str(row.get("id") or idx),
                "question": question,
                "golden_answers": gold,
                "data_source": dataset,
                "prompt": prompt,
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": {"target": gold}},
                "extra_info": {
                    "split": output_path.stem,
                    "index": idx,
                    "search_budget": budget,
                    "source_dataset": row.get("dataset"),
                    "source_id": row.get("source_id"),
                },
            }
        )
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)
    return {"input": str(input_path), "output": str(output_path), "n": len(rows), "dataset_counts": dataset_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--skill-bank-path", type=Path, default=DEFAULT_SKILL_BANK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-index-desc-chars", type=int, default=180)
    args = parser.parse_args()

    entries = parse_skill_bank(args.skill_bank_path)
    skill_index = build_skill_index(entries, args.max_index_desc_chars)
    summary = {
        "skill_bank_path": str(args.skill_bank_path),
        "n_skills": len(entries),
        "train": convert_split(args.train_jsonl, args.output_dir / "train.parquet", skill_index),
        "dev": convert_split(args.dev_jsonl, args.output_dir / "test.parquet", skill_index),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
