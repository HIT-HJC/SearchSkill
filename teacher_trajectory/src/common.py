from __future__ import annotations

import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

SKILL_ID_RE = re.compile(r"`([a-z0-9][a-z0-9-]*)`")
YESNO_START_RE = re.compile(r"^(are|is|was|were|do|does|did|has|have|had|can|could|should|would)\b")
SUPERLATIVE_RE = re.compile(r"\b(first|largest|smallest|highest|lowest|oldest|youngest|top)\b")
ALIAS_RE = re.compile(r"\b(real name|full name|nickname|alternate name|also known as|formerly|stage name)\b")
TIME_NUM_RE = re.compile(r"\b(when|what year|date|how many|how much|population|duration|height|age|count)\b")
KINSHIP_RE = re.compile(r"\b(mother|father|spouse|wife|husband|daughter|son|grandfather|grandmother|maternal|paternal|in-law)\b")
FORCED_CHOICE_RE = re.compile(r"\bor\b")
COMPARISON_RE = re.compile(r"\b(compare|same|both|older|younger|earlier|later|more|less|higher|lower)\b")
TEMPORAL_ANCHOR_RE = re.compile(r"\b(before|after|during|when|year|date|season|last|next|former|current)\b")
RELATION_CHAIN_RE = re.compile(
    r"\b(director|author|founder|creator|composer|performer|writer|actor|actress|producer|mother|father|spouse|wife|husband|country|city|state|county|birthplace|date of birth|death|school|university|alma mater)\b.*\b(of|who|whose|where)\b"
)

SUPPORT_ONLY_SKILLS = {
    "conflict-check",
    "reconstructed-chain-verification",
    "verbatim-evidence-span",
    "answer-grounding-check",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_skill_bank(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_skill_ids(path: Path) -> List[str]:
    return sorted(set(SKILL_ID_RE.findall(load_skill_bank(path))))


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("’", "'")
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match_multi(prediction: str, gold_answers: Sequence[str]) -> bool:
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return False
    return any(norm_pred == normalize_answer(gold) for gold in gold_answers if str(gold).strip())


def balanced_take(records: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    if count <= 0 or not records:
        return []
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("signature") or record.get("id") or "default")
        groups[key].append(record)
    rng = random.Random(seed)
    ordered_groups = list(groups.values())
    rng.shuffle(ordered_groups)
    for items in ordered_groups:
        rng.shuffle(items)
    selected: List[Dict[str, Any]] = []
    while ordered_groups and len(selected) < count:
        next_groups: List[List[Dict[str, Any]]] = []
        for items in ordered_groups:
            if items and len(selected) < count:
                selected.append(items.pop())
            if items:
                next_groups.append(items)
        ordered_groups = next_groups
    return selected


def build_task_family(dataset: str) -> str:
    return "singlehop" if dataset in {"nq", "triviaqa"} else "multihop"


def estimate_hops(record: Dict[str, Any]) -> int:
    metadata = record.get("metadata") or {}
    decomposition = metadata.get("question_decomposition") or []
    if decomposition:
        return max(1, len(decomposition))
    hop_count = record.get("hop_count")
    if isinstance(hop_count, int) and hop_count > 0:
        return hop_count
    question = str(record.get("question", "")).lower()
    connector_hits = len(re.findall(r"\b(of|who|whose|where|that|after|before|during|in which)\b", question))
    if connector_hits >= 3:
        return 4
    if connector_hits == 2:
        return 3
    if connector_hits == 1:
        return 2
    return 1


def build_metadata_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata") or {}
    question = str(record.get("question", ""))
    summary = {
        "task_family": build_task_family(str(record.get("dataset", ""))),
        "estimated_hops": estimate_hops(record),
        "question_type": metadata.get("type") or record.get("native_type") or "unknown",
        "level": metadata.get("level"),
        "flags": record.get("flags") or [],
        "wh_word": record.get("wh_word"),
        "entity_bin": record.get("entity_bin"),
        "token_bin": record.get("token_bin"),
        "is_yes_no": bool(YESNO_START_RE.search(question.lower())),
        "has_alias_cue": bool(ALIAS_RE.search(question.lower())),
        "has_superlative": bool(SUPERLATIVE_RE.search(question.lower())),
        "has_kinship": bool(KINSHIP_RE.search(question.lower())),
        "has_forced_choice": bool(FORCED_CHOICE_RE.search(question.lower())),
        "has_comparison": bool(COMPARISON_RE.search(question.lower())),
        "has_temporal_anchor": bool(TEMPORAL_ANCHOR_RE.search(question.lower())),
        "looks_relation_chain": bool(RELATION_CHAIN_RE.search(question.lower())),
    }
    if "failure_info" in record:
        summary["failure_error_tag"] = record["failure_info"].get("error_tag")
    return summary


def suggest_primary_skills(record: Dict[str, Any], legal_skill_ids: Sequence[str]) -> List[str]:
    question = str(record.get("question", ""))
    question_l = question.lower()
    metadata = record.get("metadata") or {}
    meta_summary = build_metadata_summary(record)
    candidates: List[str] = []

    def add(skill_id: str) -> None:
        if skill_id in legal_skill_ids and skill_id not in candidates:
            candidates.append(skill_id)

    if meta_summary["has_alias_cue"]:
        add("surface-name-resolution")
    if meta_summary["has_superlative"]:
        add("superlative-ranking-match")
    if meta_summary["has_forced_choice"] and meta_summary["task_family"] == "singlehop":
        add("forced-choice-option-resolution")
    if meta_summary["has_kinship"]:
        add("derived-kinship-inference-join")

    q_type = str(metadata.get("type") or record.get("native_type") or "").lower()
    hops = int(meta_summary["estimated_hops"])

    if q_type == "comparison":
        if hops >= 3 or meta_summary["has_temporal_anchor"]:
            add("bridge-comparison-planning")
        else:
            add("parallel-attribute-compare")
        if meta_summary["is_yes_no"]:
            add("multihop-yes-no-verification")
    elif q_type == "bridge":
        add("bridge-entity-search")
        add("relation-chain-decomposition")
    elif q_type == "compositional":
        add("relation-chain-decomposition")
        add("sequential-hop-checkpointing")

    if hops >= 4:
        add("re-anchored-long-hop-decomposition")
        add("sequential-hop-checkpointing")
    elif hops == 3:
        add("sequential-hop-checkpointing")
        add("relation-chain-decomposition")
    elif hops == 2 and "bridge-entity-search" in legal_skill_ids:
        add("bridge-entity-search")
        add("relation-chain-decomposition")

    if meta_summary["has_temporal_anchor"] and hops >= 2:
        add("temporal-anchor-carry-forward")

    if TIME_NUM_RE.search(question_l):
        add("temporal-range-extract")

    if len(question.split()) >= 12:
        add("multi-constraint-query-anchoring")

    if meta_summary["task_family"] == "singlehop":
        add("single-entity-relation-lookup")
    else:
        add("bridge-entity-search")
        add("relation-chain-decomposition")

    if not candidates:
        add("single-entity-relation-lookup")
    return candidates[:5]


def suggest_support_skills(legal_skill_ids: Sequence[str]) -> List[str]:
    ordered = [
        "conflict-check",
        "reconstructed-chain-verification",
        "verbatim-evidence-span",
        "answer-grounding-check",
    ]
    return [skill_id for skill_id in ordered if skill_id in legal_skill_ids]


def classify_failure_error(question: str, prediction: str, wrong_skills: Sequence[str]) -> str:
    q = question.lower()
    joined = "|".join(wrong_skills).lower()
    if any(skill in joined for skill in ["verbatim-evidence-span", "answer-grounding-check"]) and not any(
        skill in joined
        for skill in [
            "bridge-entity-search",
            "relation-chain-decomposition",
            "parallel-attribute-compare",
            "sequential-hop-checkpointing",
            "re-anchored-long-hop-decomposition",
        ]
    ):
        return "verifier_as_planner"
    if ALIAS_RE.search(q):
        return "alias_route_error"
    if KINSHIP_RE.search(q):
        return "kinship_route_error"
    if YESNO_START_RE.search(q):
        return "yesno_aggregation_error"
    if prediction.strip().lower() in {"yes", "no"} and not YESNO_START_RE.search(q):
        return "stopped_on_wrong_answer_type"
    if SUPERLATIVE_RE.search(q):
        return "comparison_or_ranking_route_error"
    if len(q.split()) >= 14:
        return "long_question_route_error"
    return "generic_route_error"
