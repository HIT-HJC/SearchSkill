from __future__ import annotations

import re
import string
from typing import Any


ACTION_RE = re.compile(r"(?m)^\s*<(search|answer)>(.*?)</\1>", re.IGNORECASE)
SKILL_ACTION_RE = re.compile(
    r"(?ms)^\s*<skill>(.*?)</skill>\s*\n\s*<(search|answer)>(.*?)</\2>",
    re.IGNORECASE,
)
SELECT_SKILL_RE = re.compile(r"(?m)^\s*<select_skill>(.*?)</select_skill>", re.IGNORECASE)

MULTIHOP_DATASETS = {"hotpotqa", "2wiki", "2wikimultihopqa", "musique", "bamboogle"}
SKILL_IDS = {
    "single-entity-relation-lookup",
    "surface-name-resolution",
    "multi-constraint-query-anchoring",
    "superlative-ranking-match",
    "forced-choice-option-resolution",
    "bridge-entity-search",
    "bridge-disambiguate-then-hop",
    "relation-chain-decomposition",
    "bridge-comparison-planning",
    "derived-kinship-inference-join",
    "sequential-hop-checkpointing",
    "re-anchored-long-hop-decomposition",
    "parallel-attribute-compare",
    "multihop-yes-no-verification",
    "temporal-anchor-carry-forward",
    "temporal-range-extract",
    "conflict-check",
    "reconstructed-chain-verification",
    "verbatim-evidence-span",
    "answer-grounding-check",
}


def normalize_answer(text: str | None) -> str:
    value = str(text or "").lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def em_check(prediction: str, golden_answers: Any) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred = normalize_answer(prediction)
    return any(pred and pred == normalize_answer(gold) for gold in golden_answers)


def parse_skill_ids(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]


def valid_skill_ids(text: str) -> bool:
    ids = parse_skill_ids(text)
    return bool(ids) and all(skill_id in SKILL_IDS for skill_id in ids)


def dataset_key(data_source: str | None) -> str:
    value = str(data_source or "").lower()
    if value == "2wikimultihopqa":
        return "2wiki"
    return value


def query_key(text: str) -> str:
    return normalize_answer(text)


def trajectory_stats(text: str) -> dict[str, Any]:
    actions = list(ACTION_RE.finditer(text))
    skill_actions = list(SKILL_ACTION_RE.finditer(text))
    select_tags = list(SELECT_SKILL_RE.finditer(text))
    stable_interface = bool(actions) and len(actions) == len(skill_actions)
    stable_interface = stable_interface and all(valid_skill_ids(match.group(1)) for match in skill_actions)
    stable_interface = stable_interface and all(valid_skill_ids(match.group(1)) for match in select_tags)

    search_queries = [
        query_key(match.group(3))
        for match in skill_actions
        if match.group(2).lower() == "search" and query_key(match.group(3))
    ]
    answer_actions = [match for match in skill_actions if match.group(2).lower() == "answer"]
    duplicate_count = len(search_queries) - len(set(search_queries))
    return {
        "stable_interface": stable_interface,
        "search_count": len(search_queries),
        "duplicate_count": duplicate_count,
        "answer": answer_actions[-1].group(3).strip() if answer_actions else "",
    }


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    data_source: str | None = None,
    **_: Any,
) -> float:
    """v9.2-style reward: EM-dominant plus tiny action-space shaping."""
    text = str(solution_str or "")
    target = ground_truth.get("target", [])
    stats = trajectory_stats(text)
    dataset = dataset_key(data_source)

    answer = str(stats["answer"])
    exact = bool(answer and em_check(answer, target))
    reward = 1.0 if exact else 0.0

    if not stats["stable_interface"]:
        return float(reward)

    search_count = int(stats["search_count"])
    duplicate_count = int(stats["duplicate_count"])

    if dataset in MULTIHOP_DATASETS:
        if duplicate_count == 0 and 3 <= search_count <= 4:
            reward += 0.04
        if search_count <= 1 and not exact:
            reward -= 0.04
        if duplicate_count > 0:
            reward -= min(0.04, 0.02 * duplicate_count)
    else:
        if duplicate_count == 0 and search_count <= 2:
            reward += 0.01
        if duplicate_count > 0:
            reward -= min(0.03, 0.015 * duplicate_count)

    return float(reward)
