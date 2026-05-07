#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_SKILL_BANK_PATH = Path(
    "/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_4_musique/outputs/final_skill_bank.md"
)

SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
SKILL_HEADER_RE = re.compile(r"^`([a-z0-9][a-z0-9\-]*)`\s*$")

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

ACTION_INSTRUCTION_TEMPLATE = (
    "Action phase. Read the selected the final SkillBank card(s) and follow them for the next action.\n"
    "Selected skill ids: {skill_ids}\n"
    "Now output exactly <skill>{skill_ids}</skill> followed by exactly one "
    "<search>...</search> or <answer>...</answer>. Use the same skill ids in the "
    "<skill> tag. Do not output <select_skill> in this turn. Stop immediately after "
    "the closing action tag."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack stage1 messages into the two-stage the final SkillBank protocol."
    )
    parser.add_argument(
        "--input-train",
        type=Path,
        default=Path("/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/supervised_finetuning/data/stage1/train.jsonl"),
    )
    parser.add_argument(
        "--input-eval",
        type=Path,
        default=Path("/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/supervised_finetuning/data/stage1/eval.jsonl"),
    )
    parser.add_argument("--skill-bank-path", type=Path, default=DEFAULT_SKILL_BANK_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/supervised_finetuning/data/stage2"),
    )
    parser.add_argument("--max-index-desc-chars", type=int, default=180)
    parser.add_argument("--max-card-chars", type=int, default=900)
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_skill_bank_entries(skill_bank_text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    current_skill = ""
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_skill, current_lines
        if current_skill:
            description = "\n".join(current_lines).strip()
            if description:
                entries[current_skill] = description
        current_skill = ""
        current_lines = []

    for line in skill_bank_text.splitlines():
        match = SKILL_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_skill = match.group(1)
            continue
        if current_skill:
            current_lines.append(line)
    flush()
    return entries


def compact_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars].rstrip() + "..."
    return value


def format_skill_index(entries: Dict[str, str], *, max_desc_chars: int) -> str:
    lines = ["Available the final SkillBank index:"]
    for skill_id, description in entries.items():
        lines.append(f"- {skill_id}: {compact_text(description, max_desc_chars)}")
    return "\n".join(lines)


def format_skill_cards(skill_ids: Sequence[str], entries: Dict[str, str], *, max_card_chars: int) -> str:
    cards: List[str] = ["Selected the final SkillBank card(s):"]
    for skill_id in skill_ids:
        description = compact_text(entries.get(skill_id, ""), max_card_chars)
        cards.append(f'<skill_card id="{skill_id}">\n{description}\n</skill_card>')
    return "\n".join(cards)


def parse_skill_ids_from_assistant(content: str) -> List[str]:
    match = SKILL_RE.search(str(content or ""))
    if not match:
        return []
    return dedupe_keep_order(part.strip() for part in match.group(1).split("|"))


def rewrite_user_turn(content: str, skill_index: str) -> str:
    kept_lines: List[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Recommended skills:"):
            continue
        if stripped.startswith("Start with <skill>"):
            continue
        if stripped == "Stop immediately after the closing tag.":
            continue
        kept_lines.append(line)
    base = "\n".join(kept_lines).strip()
    return f"{base}\n\n{skill_index}\n\n{SELECTION_INSTRUCTION}".strip()


def build_action_user_turn(skill_ids: Sequence[str], entries: Dict[str, str], *, max_card_chars: int) -> str:
    skill_ids_text = "|".join(skill_ids)
    return (
        f"{format_skill_cards(skill_ids, entries, max_card_chars=max_card_chars)}\n\n"
        f"{ACTION_INSTRUCTION_TEMPLATE.format(skill_ids=skill_ids_text)}"
    )


def repack_row(
    row: Dict[str, Any],
    *,
    skill_index: str,
    skill_bank_entries: Dict[str, str],
    skill_bank_path: Path,
    max_card_chars: int,
) -> tuple[Dict[str, Any] | None, str, Counter[str]]:
    source_messages = row.get("messages") or []
    if not source_messages:
        return None, "missing_messages", Counter()

    legal_skills = set(skill_bank_entries)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    skill_counter: Counter[str] = Counter()
    assistant_turns = 0

    for message in source_messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "system":
            continue
        if role == "user":
            messages.append({"role": "user", "content": rewrite_user_turn(content, skill_index)})
            continue
        if role != "assistant":
            return None, f"unsupported_role_{role}", Counter()

        skill_ids = parse_skill_ids_from_assistant(content)
        if not skill_ids:
            return None, "missing_skill_tag", Counter()
        unknown_skills = [skill_id for skill_id in skill_ids if skill_id not in legal_skills]
        if unknown_skills:
            return None, "unknown_skill", Counter(unknown_skills)
        if not messages or messages[-1].get("role") != "user":
            return None, "assistant_without_user", Counter()

        assistant_turns += 1
        skill_counter.update(skill_ids)
        messages.append({"role": "assistant", "content": f"<select_skill>{'|'.join(skill_ids)}</select_skill>"})
        messages.append(
            {
                "role": "user",
                "content": build_action_user_turn(
                    skill_ids,
                    skill_bank_entries,
                    max_card_chars=max_card_chars,
                ),
            }
        )
        messages.append({"role": "assistant", "content": content})

    if assistant_turns == 0:
        return None, "no_assistant_turns", Counter()

    packed = dict(row)
    packed["messages"] = messages
    packed["source_format"] = "stage1_packed_chat"
    packed["skill_protocol"] = "two_stage_skillbank"
    packed["skill_bank_path"] = str(skill_bank_path)
    packed["supervision_mode"] = row.get("supervision_mode", "all_assistant")
    packed["stage2_assistant_action_turns"] = assistant_turns
    packed["stage2_total_assistant_turns"] = assistant_turns * 2
    return packed, "", skill_counter


def build_split(
    input_path: Path,
    output_path: Path,
    *,
    skill_index: str,
    skill_bank_entries: Dict[str, str],
    skill_bank_path: Path,
    max_card_chars: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    message_count_hist: Counter[int] = Counter()
    dataset_counts: Counter[str] = Counter()

    for row in load_jsonl(input_path):
        packed, reason, row_skill_counts = repack_row(
            row,
            skill_index=skill_index,
            skill_bank_entries=skill_bank_entries,
            skill_bank_path=skill_bank_path,
            max_card_chars=max_card_chars,
        )
        if packed is None:
            drop_reasons[reason or "unknown"] += 1
            skill_counts.update(row_skill_counts)
            continue
        rows.append(packed)
        skill_counts.update(row_skill_counts)
        message_count_hist[len(packed["messages"])] += 1
        dataset_counts[str(packed.get("dataset", "unknown"))] += 1

    dump_jsonl(output_path, rows)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": sum(dataset_counts.values()) + sum(drop_reasons.values()),
        "output_rows": len(rows),
        "dataset_counts": dict(dataset_counts),
        "drop_reasons": dict(drop_reasons),
        "message_count_hist": dict(sorted(message_count_hist.items())),
        "skill_counts": dict(skill_counts.most_common()),
    }


def main() -> None:
    args = parse_args()
    skill_bank_text = args.skill_bank_path.read_text(encoding="utf-8")
    skill_bank_entries = parse_skill_bank_entries(skill_bank_text)
    if not skill_bank_entries:
        raise RuntimeError(f"No skill entries parsed from {args.skill_bank_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    skill_index = format_skill_index(skill_bank_entries, max_desc_chars=args.max_index_desc_chars)

    train_summary = build_split(
        args.input_train,
        args.output_dir / "train.jsonl",
        skill_index=skill_index,
        skill_bank_entries=skill_bank_entries,
        skill_bank_path=args.skill_bank_path,
        max_card_chars=args.max_card_chars,
    )
    eval_summary = build_split(
        args.input_eval,
        args.output_dir / "eval.jsonl",
        skill_index=skill_index,
        skill_bank_entries=skill_bank_entries,
        skill_bank_path=args.skill_bank_path,
        max_card_chars=args.max_card_chars,
    )
    summary = {
        "strategy": "repack_stage1_full_data_as_two_stage_final_skill_bank_protocol",
        "skill_bank_path": str(args.skill_bank_path),
        "n_skill_bank_entries": len(skill_bank_entries),
        "max_index_desc_chars": args.max_index_desc_chars,
        "max_card_chars": args.max_card_chars,
        "train": train_summary,
        "eval": eval_summary,
        "note": (
            "Each original assistant action turn is split into a supervised "
            "<select_skill> turn, a selected skill-card user turn, and the original "
            "<skill> plus action turn. No Recommended skills line is kept."
        ),
    }
    dump_json(args.output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
