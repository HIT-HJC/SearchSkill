from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SUPPORT_ONLY_SKILLS = {
    "conflict-check",
    "reconstructed-chain-verification",
    "verbatim-evidence-span",
    "answer-grounding-check",
}

HOTPOT_CRITICAL_SKILLS = {
    "bridge-entity-search",
    "relation-chain-decomposition",
    "parallel-attribute-compare",
    "bridge-comparison-planning",
    "multihop-yes-no-verification",
    "temporal-range-extract",
    "temporal-anchor-carry-forward",
    "sequential-hop-checkpointing",
}

MULTIHOP_CORE_SKILLS = {
    "bridge-entity-search",
    "bridge-comparison-planning",
    "bridge-disambiguate-then-hop",
    "relation-chain-decomposition",
    "parallel-attribute-compare",
    "multihop-yes-no-verification",
    "temporal-anchor-carry-forward",
    "temporal-range-extract",
    "sequential-hop-checkpointing",
    "re-anchored-long-hop-decomposition",
    "derived-kinship-inference-join",
    "reconstructed-chain-verification",
}

SINGLEHOP_CORE_SKILLS = {
    "single-entity-relation-lookup",
    "surface-name-resolution",
    "forced-choice-option-resolution",
    "temporal-range-extract",
    "conflict-check",
}

MULTIHOP_DATASETS = {"hotpotqa", "2wiki", "musique"}
SINGLEHOP_DATASETS = {"nq", "triviaqa"}

SYSTEM_PROMPT = (
    "You are participating in a retrieval tool-use evaluation. "
    "You do not have direct access to search results. "
    "Never fabricate or simulate an <information> block yourself. "
    "If you need retrieval, emit a <search>...</search> tag and stop immediately after the first </search>. "
    "Do not output more than one <search> tag in a single response. "
    "When you have enough evidence, emit the final answer inside <answer>...</answer>. "
    "The final answer must be the shortest exact answer span, not a sentence. "
    "You must emit <skill>...</skill> at the start of every turn to declare the skill you are using."
)

USER_PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you still need outside evidence, call the search engine with exactly one <search> query </search> tag. "
    "It will return the top search results between <information> and </information>. "
    "You can search multiple times, but only once per turn. "
    "If you have enough evidence, provide the answer inside <answer> and </answer>, without extra explanation. "
    "Do not describe searching in natural language. "
    "The final answer must be the shortest exact answer span, not a full sentence. "
    "For yes/no questions, output exactly yes or no. "
    "For person/place/work names, output only the name. "
    "For years or numbers, output only the value. "
    "Do not add prefixes such as 'The answer is'. "
    "Recommended skills for this question: {recommended_skills}\n"
    "At the start of every turn, emit <skill>chosen-skill-1|chosen-skill-2</skill> before your <search> or <answer>. "
    "If you use bridge-entity-search, do not jump to the final answer until the bridge entity is grounded in evidence. "
    "If you use parallel-attribute-compare, compare both sides explicitly before answering. "
    "If you use conflict-check, do one extra targeted verification search before answering. "
    "If you use verbatim-evidence-span, copy the answer span from the evidence instead of paraphrasing. "
    "If you use answer-grounding-check, do not finalize an answer that is not explicitly supported by retrieved evidence. "
    "Question: {question}\n"
)

FOLLOWUP_USER_TEMPLATE = (
    "<information>{retrieved}</information>\n\n"
    "Continue the same question. "
    "If you still need outside evidence, output exactly one <search>...</search>. "
    "Otherwise, output the final answer inside <answer>...</answer> with no extra explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the formal SFT dataset from canonical teacher trajectories.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--format-style", choices=("json_action", "retrieval_tags"), default="retrieval_tags")
    parser.add_argument("--sampling-profile", choices=("focus_dataset", "benchmark_balanced_v2"), default="focus_dataset")
    parser.add_argument("--max-retrieved-chars", type=int, default=4000)
    parser.add_argument("--focus-dataset", type=str, default="hotpotqa")
    parser.add_argument("--focus-multiplier", type=int, default=2)
    parser.add_argument("--critical-focus-bonus", type=int, default=1)
    parser.add_argument("--singlehop-base-bonus", type=int, default=1)
    parser.add_argument("--hard-pattern-bonus", type=int, default=1)
    parser.add_argument("--singlehop-skill-bonus", type=int, default=1)
    parser.add_argument("--multihop-skill-bonus", type=int, default=1)
    parser.add_argument("--failure-replay-bonus", type=int, default=1)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--max-skill-tags", type=int, default=2)
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


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    for ch in ".,!?;:'\"()[]{}-_`":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def cleaned_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = [dict(step) for step in steps]
    while len(cleaned) >= 2:
        tail = cleaned[-1]
        prev = cleaned[-2]
        tail_answer = normalize_answer(tail.get("draft_answer", ""))
        prev_answer = normalize_answer(prev.get("draft_answer", ""))
        if (
            tail.get("action") == "answer"
            and prev.get("action") == "answer"
            and tail_answer
            and tail_answer == prev_answer
            and tail.get("primary_skill") in SUPPORT_ONLY_SKILLS
        ):
            cleaned.pop()
            continue
        break
    return cleaned


def quality_score(row: Dict[str, Any]) -> int:
    meta = row.get("metadata_summary") or {}
    steps = row.get("steps") or []
    score = 100
    step_count = len(steps)

    if step_count in {4, 5}:
        score += 18
    elif step_count in {3, 6}:
        score += 12
    elif step_count == 7:
        score += 4
    else:
        score -= 4

    final_skill = steps[-1].get("primary_skill", "") if steps else ""
    if final_skill not in SUPPORT_ONLY_SKILLS:
        score += 8

    if meta.get("estimated_hops", 0) >= 3:
        score += 6
    if meta.get("has_comparison"):
        score += 5
    if meta.get("is_yes_no"):
        score += 4
    if meta.get("has_temporal_anchor"):
        score += 4
    if meta.get("has_superlative"):
        score += 3
    if meta.get("has_kinship"):
        score += 3
    if meta.get("has_forced_choice"):
        score += 2
    if meta.get("looks_relation_chain"):
        score += 2

    if row.get("sample_origin") == "failure_replay":
        score -= 2
    if row.get("failure_info"):
        score -= 4

    return score


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        item = (value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def render_json_action(step: Dict[str, Any]) -> str:
    assistant_payload = {
        "primary_skill": step.get("primary_skill", ""),
        "support_skills": step.get("support_skills", []),
        "action": step.get("action", ""),
        "query": step.get("query", ""),
        "checkpoint": step.get("checkpoint", {}),
        "draft_answer": step.get("draft_answer", ""),
    }
    return json.dumps(assistant_payload, ensure_ascii=False)


def select_skill_tags(step: Dict[str, Any], max_skill_tags: int) -> List[str]:
    primary = step.get("primary_skill", "")
    support = step.get("support_skills", []) or []
    skill_tags = dedupe_keep_order([primary, *support])
    if not skill_tags:
        skill_tags = ["answer-grounding-check"]
    return skill_tags[: max(1, max_skill_tags)]


def normalized_teacher_action(step: Dict[str, Any]) -> str:
    raw_action = (step.get("action") or "").strip().lower()
    query = (step.get("query") or "").strip()
    answer = (step.get("draft_answer") or "").strip()
    if raw_action == "search":
        return "search"
    if raw_action == "verify":
        return "search" if query else "answer"
    if raw_action == "answer":
        return "answer"
    if query:
        return "search"
    if answer:
        return "answer"
    return "search"


def extract_step_answer(step: Dict[str, Any], row: Dict[str, Any]) -> str:
    value = (step.get("draft_answer") or "").strip()
    if value:
        return value
    return (row.get("final_answer") or "").strip()


def render_retrieval_turn(step: Dict[str, Any], row: Dict[str, Any], max_skill_tags: int) -> str:
    skill_text = "|".join(select_skill_tags(step, max_skill_tags))
    action = normalized_teacher_action(step)
    if action == "search":
        query = (step.get("query") or "").strip()
        if not query:
            query = (row.get("question") or "").strip()
        return f"<skill>{skill_text}</skill>\n<search>{query}</search>"
    answer = extract_step_answer(step, row)
    return f"<skill>{skill_text}</skill>\n<answer>{answer}</answer>"


def build_messages(
    row: Dict[str, Any],
    *,
    format_style: str,
    max_retrieved_chars: int,
    max_skill_tags: int,
) -> List[Dict[str, str]]:
    meta = row.get("metadata_summary") or {}
    recommended_skills = dedupe_keep_order(row.get("candidate_primary_skills", []) or [])

    if format_style == "json_action":
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"Question: {row['question']}\n"
                    f"Dataset: {row['dataset']}\n"
                    f"Estimated hops: {meta.get('estimated_hops', 'unknown')}\n"
                    f"Candidate primary skills: {', '.join(recommended_skills)}\n"
                    "Return only the next structured action as JSON."
                ),
            }
        ]
        for step in row.get("steps") or []:
            messages.append({"role": "assistant", "content": render_json_action(step)})
            retrieved = (step.get("retrieved") or "").strip()
            if retrieved:
                messages.append(
                    {
                        "role": "user",
                        "content": "Retriever results:\n" + retrieved[:max_retrieved_chars] + "\nReturn only the next structured action as JSON.",
                    }
                )
        return messages

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                recommended_skills=", ".join(recommended_skills) or "(none)",
                question=row["question"],
            ),
        },
    ]
    for step in row.get("steps") or []:
        messages.append({"role": "assistant", "content": render_retrieval_turn(step, row, max_skill_tags)})
        retrieved = (step.get("retrieved") or "").strip()
        if retrieved:
            messages.append(
                {
                    "role": "user",
                    "content": FOLLOWUP_USER_TEMPLATE.format(retrieved=retrieved[:max_retrieved_chars]),
                }
            )
    return messages


def pack_row(
    row: Dict[str, Any],
    *,
    format_style: str,
    max_retrieved_chars: int,
    max_skill_tags: int,
) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "question": row["question"],
        "gold_answers": row.get("gold_answers", []),
        "messages": build_messages(
            row,
            format_style=format_style,
            max_retrieved_chars=max_retrieved_chars,
            max_skill_tags=max_skill_tags,
        ),
        "final_answer": row.get("final_answer", ""),
        "first_primary_skill": row.get("steps", [{}])[0].get("primary_skill", "") if row.get("steps") else "",
        "metadata_summary": row.get("metadata_summary", {}),
        "quality_score": row.get("quality_score", 0),
        "sample_origin": row.get("sample_origin", "unknown"),
        "canonical_source_label": row.get("canonical_source_label", "unknown"),
        "repeat_factor": row.get("repeat_factor", 1),
    }


def split_train_eval(rows: Sequence[Dict[str, Any]], eval_count: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    remaining_eval = min(eval_count, len(rows))

    for dataset in sorted(grouped.keys()):
        items = list(grouped[dataset])
        rng.shuffle(items)
        take = min(max(1, round(len(items) * remaining_eval / max(1, len(rows)))), len(items), remaining_eval)
        eval_rows.extend(items[:take])
        train_rows.extend(items[take:])
        remaining_eval -= take

    if remaining_eval > 0:
        train_rows.sort(key=lambda row: (row["dataset"], row["id"]))
        eval_rows.extend(train_rows[:remaining_eval])
        train_rows = train_rows[remaining_eval:]

    train_rows.sort(key=lambda row: (row["dataset"], row["id"]))
    eval_rows.sort(key=lambda row: (row["dataset"], row["id"]))
    return train_rows, eval_rows


def repeat_factor(
    row: Dict[str, Any],
    *,
    sampling_profile: str,
    focus_dataset: str,
    focus_multiplier: int,
    critical_focus_bonus: int,
    singlehop_base_bonus: int,
    hard_pattern_bonus: int,
    singlehop_skill_bonus: int,
    multihop_skill_bonus: int,
    failure_replay_bonus: int,
    max_repeat: int,
) -> int:
    meta = row.get("metadata_summary") or {}
    dataset = row.get("dataset")
    first_skill = row.get("steps", [{}])[0].get("primary_skill", "") if row.get("steps") else ""
    hard_pattern = (
        meta.get("has_comparison")
        or meta.get("is_yes_no")
        or meta.get("has_temporal_anchor")
        or meta.get("has_forced_choice")
        or meta.get("looks_relation_chain")
        or meta.get("estimated_hops", 0) >= 3
    )
    repeat = 1
    if sampling_profile == "focus_dataset":
        if dataset == focus_dataset:
            repeat = max(repeat, focus_multiplier)
            is_critical = hard_pattern or first_skill in HOTPOT_CRITICAL_SKILLS
            if is_critical:
                repeat += max(0, critical_focus_bonus)
    else:
        if dataset in SINGLEHOP_DATASETS:
            repeat += max(0, singlehop_base_bonus)
        if hard_pattern:
            repeat += max(0, hard_pattern_bonus)
        if dataset in SINGLEHOP_DATASETS and (
            meta.get("estimated_hops", 0) <= 1
            or meta.get("has_alias_cue")
            or first_skill in SINGLEHOP_CORE_SKILLS
        ):
            repeat += max(0, singlehop_skill_bonus)
        if dataset in MULTIHOP_DATASETS and first_skill in MULTIHOP_CORE_SKILLS:
            repeat += max(0, multihop_skill_bonus)
    if row.get("sample_origin") == "failure_replay":
        repeat += max(0, failure_replay_bonus)
    return min(max_repeat, max(1, repeat))


def expand_train_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    sampling_profile: str,
    focus_dataset: str,
    focus_multiplier: int,
    critical_focus_bonus: int,
    singlehop_base_bonus: int,
    hard_pattern_bonus: int,
    singlehop_skill_bonus: int,
    multihop_skill_bonus: int,
    failure_replay_bonus: int,
    max_repeat: int,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for row in rows:
        repeat = repeat_factor(
            row,
            sampling_profile=sampling_profile,
            focus_dataset=focus_dataset,
            focus_multiplier=focus_multiplier,
            critical_focus_bonus=critical_focus_bonus,
            singlehop_base_bonus=singlehop_base_bonus,
            hard_pattern_bonus=hard_pattern_bonus,
            singlehop_skill_bonus=singlehop_skill_bonus,
            multihop_skill_bonus=multihop_skill_bonus,
            failure_replay_bonus=failure_replay_bonus,
            max_repeat=max_repeat,
        )
        for idx in range(repeat):
            clone = copy.deepcopy(row)
            clone["repeat_factor"] = repeat
            if idx:
                clone["id"] = f"{row['id']}__rep{idx + 1}"
            expanded.append(clone)
    return expanded


def summarize(
    rows: Sequence[Dict[str, Any]],
    *,
    eval_rows: Sequence[Dict[str, Any]],
    expanded_train_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    by_dataset = Counter(row["dataset"] for row in rows)
    by_origin = Counter(row.get("sample_origin", "unknown") for row in rows)
    by_source = Counter(row.get("canonical_source_label", "unknown") for row in rows)
    hop_counts = Counter((row.get("metadata_summary") or {}).get("estimated_hops", 0) for row in rows)
    tag_counts = Counter()
    step_counts = Counter(len(row.get("steps") or []) for row in rows)
    repeat_hist = Counter(row.get("repeat_factor", 1) for row in expanded_train_rows)
    expanded_by_dataset = Counter(row["dataset"] for row in expanded_train_rows)
    for row in rows:
        meta = row.get("metadata_summary") or {}
        for tag in [
            "is_yes_no",
            "has_comparison",
            "has_temporal_anchor",
            "has_superlative",
            "has_kinship",
            "has_forced_choice",
            "has_alias_cue",
            "looks_relation_chain",
        ]:
            if meta.get(tag):
                tag_counts[tag] += 1
    return {
        "format_style": args.format_style,
        "sampling_profile": args.sampling_profile,
        "total_examples": len(rows),
        "train_examples_base": len(rows) - len(eval_rows),
        "train_examples_expanded": len(expanded_train_rows),
        "eval_examples": len(eval_rows),
        "dataset_counts": dict(by_dataset),
        "expanded_train_dataset_counts": dict(expanded_by_dataset),
        "sample_origin_counts": dict(by_origin),
        "canonical_source_counts": dict(by_source),
        "hop_counts": dict(hop_counts),
        "step_counts": dict(step_counts),
        "tag_counts": dict(tag_counts),
        "repeat_factor_hist": dict(repeat_hist),
        "focus_dataset": args.focus_dataset,
        "focus_multiplier": args.focus_multiplier,
        "critical_focus_bonus": args.critical_focus_bonus,
        "singlehop_base_bonus": args.singlehop_base_bonus,
        "hard_pattern_bonus": args.hard_pattern_bonus,
        "singlehop_skill_bonus": args.singlehop_skill_bonus,
        "multihop_skill_bonus": args.multihop_skill_bonus,
        "failure_replay_bonus": args.failure_replay_bonus,
        "max_repeat": args.max_repeat,
        "max_retrieved_chars": args.max_retrieved_chars,
        "average_quality_score": round(sum(row.get("quality_score", 0) for row in rows) / max(1, len(rows)), 2),
    }


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for row in load_jsonl(args.input_path):
        item = dict(row)
        item["steps"] = cleaned_steps(item.get("steps") or [])
        item["quality_score"] = quality_score(item)
        rows.append(item)

    rows = sorted(rows, key=lambda row: (row["dataset"], -row["quality_score"], row["id"]))
    train_rows, eval_rows = split_train_eval(rows, eval_count=args.eval_count, seed=args.seed)
    expanded_train_rows = expand_train_rows(
        train_rows,
        sampling_profile=args.sampling_profile,
        focus_dataset=args.focus_dataset,
        focus_multiplier=args.focus_multiplier,
        critical_focus_bonus=args.critical_focus_bonus,
        singlehop_base_bonus=args.singlehop_base_bonus,
        hard_pattern_bonus=args.hard_pattern_bonus,
        singlehop_skill_bonus=args.singlehop_skill_bonus,
        multihop_skill_bonus=args.multihop_skill_bonus,
        failure_replay_bonus=args.failure_replay_bonus,
        max_repeat=args.max_repeat,
    )

    packed_rows = [
        pack_row(
            row,
            format_style=args.format_style,
            max_retrieved_chars=args.max_retrieved_chars,
            max_skill_tags=args.max_skill_tags,
        )
        for row in rows
    ]
    packed_train = [
        pack_row(
            row,
            format_style=args.format_style,
            max_retrieved_chars=args.max_retrieved_chars,
            max_skill_tags=args.max_skill_tags,
        )
        for row in expanded_train_rows
    ]
    packed_eval = [
        pack_row(
            row,
            format_style=args.format_style,
            max_retrieved_chars=args.max_retrieved_chars,
            max_skill_tags=args.max_skill_tags,
        )
        for row in eval_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "selected_trajectories.jsonl", rows)
    dump_jsonl(args.output_dir / "sft_messages.jsonl", packed_rows)
    dump_jsonl(args.output_dir / "train.jsonl", packed_train)
    dump_jsonl(args.output_dir / "eval.jsonl", packed_eval)
    dump_json(
        args.output_dir / "selection_summary.json",
        summarize(
            rows,
            eval_rows=eval_rows,
            expanded_train_rows=expanded_train_rows,
            args=args,
        ),
    )


if __name__ == "__main__":
    main()
