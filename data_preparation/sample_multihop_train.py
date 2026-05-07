#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


DATASET_SPECS = {
    "hotpotqa": {
        "train_paths": [
            "/path/to/hf_cache/datasets/RUC-NLPIR___flash_rag_datasets/hotpotqa/0.0.0/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/flash_rag_datasets-train-00000-of-00002.arrow",
            "/path/to/hf_cache/datasets/RUC-NLPIR___flash_rag_datasets/hotpotqa/0.0.0/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/flash_rag_datasets-train-00001-of-00002.arrow",
        ],
        "eval_path": "/path/to/hf_data/data/hotpotqa/test.jsonl",
    },
    "2wiki": {
        "train_paths": [
            "/path/to/hf_cache/datasets/RUC-NLPIR___flash_rag_datasets/2wikimultihopqa/0.0.0/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/flash_rag_datasets-train.arrow",
        ],
        "eval_path": "/path/to/hf_data/data/2wiki/test.jsonl",
    },
    "musique": {
        "train_paths": [
            "/path/to/hf_cache/datasets/RUC-NLPIR___flash_rag_datasets/musique/0.0.0/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c/flash_rag_datasets-train.arrow",
        ],
        "eval_path": "/path/to/hf_data/data/musique/test.jsonl",
    },
}


REASONING_LABELS = [
    "bridge_reasoning",
    "comparison_reasoning",
    "multi_entity_grounding",
    "relation_chain",
    "set_constraint",
    "temporal_reasoning",
    "numerical_reasoning",
    "verification_or_yes_no",
    "disambiguation_or_alias",
    "compositional_lookup",
]


SKILL_LABELS = [
    "entity_grounding",
    "relation_following",
    "bridge_search",
    "comparison",
    "constraint_filtering",
    "temporal_filtering",
    "aggregation_or_counting",
    "evidence_verification",
    "answer_composition",
]


WH_WORDS = [
    "who",
    "what",
    "when",
    "where",
    "which",
    "whose",
    "whom",
    "how",
    "are",
    "is",
    "was",
    "were",
    "did",
    "do",
    "does",
]


COMPARISON_CUES = [
    "same",
    "earlier",
    "later",
    "older",
    "younger",
    "larger",
    "smaller",
    "higher",
    "lower",
    "more",
    "less",
    "first",
    "last",
    "before",
    "after",
    "both",
]


TEMPORAL_CUES = [
    "year",
    "date",
    "when",
    "before",
    "after",
    "earlier",
    "later",
    "founded",
    "born",
    "died",
    "released",
]


NUMERIC_CUES = [
    "how many",
    "how much",
    "number of",
    "count",
    "population",
    "age",
    "score",
]


DISAMBIGUATION_CUES = [
    "same country",
    "same nationality",
    "same place",
    "same city",
    "same state",
    "same year",
]


CAPITALIZED_SPAN_RE = re.compile(r"(?:[A-Z][\w'&.-]*)(?:\s+(?:[A-Z][\w'&.-]*))*")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
DIGIT_RE = re.compile(r"\d")
JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)
SPACE_RE = re.compile(r"\s+")


@dataclass
class GroupAnnotation:
    group_id: str
    reasoning_types: list[str]
    skill_demands: list[str]
    answer_form: str
    entity_pattern: str
    difficulty: str
    coverage_priority: int
    short_rationale: str
    source: str


@dataclass
class QualityReview:
    record_id: str
    keep: bool
    reason: str
    source: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_cache_rows(paths: list[str]) -> Iterable[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")
        if path.suffix == ".arrow":
            with pa.memory_map(str(path), "r") as source:
                reader = ipc.open_stream(source)
                for batch in reader:
                    for row in batch.to_pylist():
                        yield row
            continue
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    yield row
            continue
        raise ValueError(f"Unsupported cache file: {path}")


def load_eval_ids(path: str) -> set[str]:
    eval_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            eval_ids.add(str(record["id"]))
    return eval_ids


def first_wh_word(question: str) -> str:
    lowered = question.strip().lower()
    for wh_word in WH_WORDS:
        if lowered.startswith(f"{wh_word} "):
            return wh_word
    return "other"


def count_entity_spans(question: str) -> int:
    spans = {span.strip() for span in CAPITALIZED_SPAN_RE.findall(question)}
    spans = {span for span in spans if len(span) > 1}
    return len(spans)


def entity_bucket(entity_count: int) -> str:
    if entity_count <= 1:
        return "single_entity"
    if entity_count == 2:
        return "two_entities"
    if entity_count <= 4:
        return "multi_entity"
    return "dense_entity"


def token_bucket(question: str) -> str:
    length = len(question.split())
    if length <= 8:
        return "short"
    if length <= 14:
        return "medium"
    if length <= 22:
        return "long"
    return "very_long"


def answer_form(answer: str) -> str:
    lowered = answer.strip().lower()
    if lowered in {"yes", "no"}:
        return "boolean"
    if YEAR_RE.search(answer):
        return "year_or_date"
    if DIGIT_RE.search(answer):
        return "number"
    if "," in answer:
        return "list_like"
    if len(answer.split()) >= 3:
        return "long_span"
    return "entity_or_short_span"


def derive_native_summary(dataset: str, row: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    metadata = row.get("metadata", {}) or {}
    if dataset == "hotpotqa":
        supporting_titles = metadata.get("supporting_facts", {}).get("title", []) or []
        native_type = metadata.get("type", "unknown")
        hop_count = len(set(supporting_titles)) or 2
        summary = {
            "native_type": native_type,
            "level": metadata.get("level", "unknown"),
            "supporting_title_count": len(set(supporting_titles)),
            "context_title_count": len(metadata.get("context", {}).get("title", []) or []),
        }
        return native_type, hop_count, summary
    if dataset == "2wiki":
        supporting_titles = metadata.get("supporting_facts", {}).get("title", []) or []
        native_type = metadata.get("type", "unknown")
        hop_count = len(set(supporting_titles)) or 2
        summary = {
            "native_type": native_type,
            "supporting_title_count": len(set(supporting_titles)),
            "context_title_count": len(metadata.get("context", {}).get("title", []) or []),
        }
        return native_type, hop_count, summary
    decomposition = metadata.get("question_decomposition", []) or []
    native_type = f"decomp_{len(decomposition)}"
    hop_count = len(decomposition) or 1
    summary = {
        "native_type": native_type,
        "answerable": bool(metadata.get("answerable", True)),
        "hop_count": hop_count,
    }
    return native_type, hop_count, summary


def derive_flags(question: str, answer: str, native_type: str, hop_count: int) -> list[str]:
    lowered = question.lower()
    flags: set[str] = set()
    if any(cue in lowered for cue in COMPARISON_CUES) or "comparison" in native_type:
        flags.add("comparison")
    if any(cue in lowered for cue in TEMPORAL_CUES):
        flags.add("temporal")
    if any(cue in lowered for cue in NUMERIC_CUES) or DIGIT_RE.search(answer):
        flags.add("numeric")
    if any(cue in lowered for cue in DISAMBIGUATION_CUES):
        flags.add("verification")
    if lowered.startswith(("is ", "are ", "was ", "were ", "did ", "do ", "does ")):
        flags.add("yes_no")
    if " and " in lowered or " or " in lowered or " both " in lowered:
        flags.add("multi_entity")
    if "mother of" in lowered or "father of" in lowered or "spouse of" in lowered or "director of" in lowered:
        flags.add("relation_chain")
    if "which" in lowered or "whose" in lowered or "what" in lowered:
        flags.add("lookup")
    if "same " in lowered or "from the same" in lowered:
        flags.add("same_attribute")
    if "born" in lowered or "founded" in lowered or "released" in lowered:
        flags.add("time_anchor")
    if hop_count >= 3:
        flags.add("longer_hop")
    if count_entity_spans(question) >= 3:
        flags.add("dense_entities")
    return sorted(flags)


def build_signature(dataset: str, wh_word: str, entity_bin: str, token_bin: str, native_type: str, hop_count: int, flags: list[str]) -> str:
    signature = {
        "dataset": dataset,
        "native_type": native_type,
        "hop_count": hop_count,
        "wh_word": wh_word,
        "entity_bin": entity_bin,
        "token_bin": token_bin,
        "flags": flags[:6],
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True)


def light_record(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answers = row.get("golden_answers") or []
    primary_answer = str(answers[0]).strip() if answers else ""
    native_type, hop_count, native_summary = derive_native_summary(dataset, row)
    wh_word = first_wh_word(question)
    entity_count = count_entity_spans(question)
    entity_bin = entity_bucket(entity_count)
    token_bin = token_bucket(question)
    flags = derive_flags(question, primary_answer, native_type, hop_count)
    return {
        "id": str(row["id"]),
        "dataset": dataset,
        "question": question,
        "primary_answer": primary_answer,
        "answer_form_hint": answer_form(primary_answer),
        "native_type": native_type,
        "hop_count": hop_count,
        "wh_word": wh_word,
        "entity_count": entity_count,
        "entity_bin": entity_bin,
        "token_bin": token_bin,
        "flags": flags,
        "native_summary": native_summary,
        "signature": build_signature(dataset, wh_word, entity_bin, token_bin, native_type, hop_count, flags),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def recommend_target_size(num_examples: int, native_type_count: int, hop_bucket_count: int) -> int:
    target = round((0.10 * num_examples) + (1200 * native_type_count) + (1000 * hop_bucket_count))
    target = max(5000, target)
    target = min(num_examples, target)
    return target


def build_light_profiles(dataset: str, root_dir: Path, overwrite_existing: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile_dir = root_dir / "profiles"
    ensure_dir(profile_dir)
    report_path = root_dir / "reports" / f"{dataset}_profile_report.json"
    ensure_dir(report_path.parent)
    light_path = profile_dir / f"{dataset}_train_light.jsonl"
    if light_path.exists() and report_path.exists() and not overwrite_existing:
        return read_jsonl(light_path), json.loads(report_path.read_text(encoding="utf-8"))

    spec = DATASET_SPECS[dataset]
    eval_ids = load_eval_ids(spec["eval_path"])
    examples: list[dict[str, Any]] = []
    type_counter = Counter()
    hop_counter = Counter()
    wh_counter = Counter()
    flag_counter = Counter()
    for row in tqdm(iter_cache_rows(spec["train_paths"]), desc=f"profile:{dataset}"):
        example = light_record(dataset, row)
        example["is_eval_overlap"] = example["id"] in eval_ids
        examples.append(example)
        type_counter[example["native_type"]] += 1
        hop_counter[str(example["hop_count"])] += 1
        wh_counter[example["wh_word"]] += 1
        for flag in example["flags"]:
            flag_counter[flag] += 1
    write_jsonl(light_path, examples)
    report = {
        "dataset": dataset,
        "num_examples": len(examples),
        "num_signatures": len({example["signature"] for example in examples}),
        "native_type_distribution": dict(type_counter.most_common()),
        "hop_distribution": dict(hop_counter.most_common()),
        "wh_distribution": dict(wh_counter.most_common()),
        "flag_distribution": dict(flag_counter.most_common()),
        "recommended_target_size": recommend_target_size(len(examples), len(type_counter), len(hop_counter)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples, report


def choose_representatives(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(records) <= count:
        return records
    ranked = sorted(records, key=lambda item: (len(item["question"].split()), item["question"]))
    positions = sorted({round(index * (len(ranked) - 1) / max(count - 1, 1)) for index in range(count)})
    reps = [ranked[position] for position in positions]
    while len(reps) < count:
        candidate = rng.choice(ranked)
        if candidate not in reps:
            reps.append(candidate)
    return reps[:count]


def build_groups(examples: list[dict[str, Any]], representatives: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["signature"]].append(example)
    groups: list[dict[str, Any]] = []
    for signature, records in grouped.items():
        first = records[0]
        group_id = hashlib.md5(signature.encode("utf-8")).hexdigest()[:16]
        reps = choose_representatives(records, representatives, seed)
        groups.append(
            {
                "group_id": group_id,
                "dataset": first["dataset"],
                "signature": json.loads(signature),
                "size": len(records),
                "native_type": first["native_type"],
                "hop_count": first["hop_count"],
                "heuristic_flags": first["flags"],
                "representatives": [
                    {
                        "id": rep["id"],
                        "question": rep["question"],
                        "primary_answer": rep["primary_answer"],
                        "answer_form_hint": rep["answer_form_hint"],
                        "entity_bin": rep["entity_bin"],
                        "flags": rep["flags"],
                    }
                    for rep in reps
                ],
            }
        )
    groups.sort(key=lambda item: (-item["size"], item["group_id"]))
    return groups


def responses_request(base_url: str, api_key: str, model: str, reasoning_effort: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "store": False,
            "reasoning": {"effort": reasoning_effort},
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]
    chunks: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") in {"output_text", "text"} and isinstance(node.get("text"), str):
                chunks.append(node["text"])
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload.get("output", payload))
    return "".join(chunks).strip()


def parse_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def group_system_prompt() -> str:
    labels = ", ".join(REASONING_LABELS)
    skills = ", ".join(SKILL_LABELS)
    return (
        "You curate a training set for multi-hop QA skill learning. "
        "Classify each question group using only the controlled label spaces. "
        "Return valid JSON only. "
        f"Allowed reasoning_types: [{labels}]. "
        f"Allowed skill_demands: [{skills}]. "
        "Allowed answer_form: [boolean, entity_or_short_span, year_or_date, number, list_like, long_span]. "
        "Allowed entity_pattern: [single_entity, pairwise_entities, multi_entity, relation_chain, set_constraint, dense_entity]. "
        "Allowed difficulty: [easy, medium, hard]. "
        "coverage_priority must be an integer from 1 to 5. "
        "The decision should favor skill coverage, especially rare reasoning patterns and multi-entity cases."
    )


def group_user_prompt(groups: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Annotate each group for coverage-oriented training subset selection.",
        "groups": groups,
        "output_schema": {
            "annotations": [
                {
                    "group_id": "string",
                    "reasoning_types": ["allowed_label"],
                    "skill_demands": ["allowed_label"],
                    "answer_form": "allowed_label",
                    "entity_pattern": "allowed_label",
                    "difficulty": "allowed_label",
                    "coverage_priority": 3,
                    "short_rationale": "brief",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def heuristic_group_annotation(group: dict[str, Any]) -> GroupAnnotation:
    signature = group["signature"]
    reasoning_types: set[str] = {"compositional_lookup"}
    skill_demands: set[str] = {"answer_composition"}
    entity_pattern = "single_entity"
    flags = set(group.get("heuristic_flags", []))
    native_type = str(group.get("native_type", "unknown"))
    if "comparison" in flags or "comparison" in native_type:
        reasoning_types.add("comparison_reasoning")
        skill_demands.add("comparison")
        entity_pattern = "pairwise_entities"
    if "relation_chain" in flags or "bridge" in native_type or "inference" in native_type or group.get("hop_count", 0) >= 3:
        reasoning_types.add("relation_chain")
        reasoning_types.add("bridge_reasoning")
        skill_demands.add("relation_following")
        skill_demands.add("bridge_search")
        entity_pattern = "relation_chain"
    if "temporal" in flags or "time_anchor" in flags:
        reasoning_types.add("temporal_reasoning")
        skill_demands.add("temporal_filtering")
    if "numeric" in flags:
        reasoning_types.add("numerical_reasoning")
        skill_demands.add("aggregation_or_counting")
    if "yes_no" in flags or "verification" in flags:
        reasoning_types.add("verification_or_yes_no")
        skill_demands.add("evidence_verification")
    if "multi_entity" in flags or signature.get("entity_bin") in {"multi_entity", "dense_entity"}:
        reasoning_types.add("multi_entity_grounding")
        skill_demands.add("entity_grounding")
        entity_pattern = "multi_entity"
    if "same_attribute" in flags:
        reasoning_types.add("set_constraint")
        skill_demands.add("constraint_filtering")
        entity_pattern = "set_constraint"
    coverage_priority = 3
    if group["size"] < 32 or entity_pattern in {"multi_entity", "relation_chain", "set_constraint"}:
        coverage_priority = 4
    if group["hop_count"] >= 4:
        coverage_priority = 5
    return GroupAnnotation(
        group_id=group["group_id"],
        reasoning_types=sorted(reasoning_types),
        skill_demands=sorted(skill_demands),
        answer_form=group["representatives"][0]["answer_form_hint"],
        entity_pattern=entity_pattern,
        difficulty="hard" if group["hop_count"] >= 4 else "medium",
        coverage_priority=coverage_priority,
        short_rationale="heuristic fallback",
        source="heuristic_fallback",
    )


def annotate_group_batch(
    groups: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, GroupAnnotation]:
    if not api_key:
        return {group["group_id"]: heuristic_group_annotation(group) for group in groups}
    try:
        payload = responses_request(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            system_prompt=group_system_prompt(),
            user_prompt=group_user_prompt(groups),
        )
        parsed = parse_json_payload(response_text(payload))
        annotations = parsed["annotations"] if isinstance(parsed, dict) else parsed
        result: dict[str, GroupAnnotation] = {}
        for item in annotations:
            annotation = GroupAnnotation(
                group_id=str(item["group_id"]),
                reasoning_types=[label for label in item.get("reasoning_types", []) if label in REASONING_LABELS],
                skill_demands=[label for label in item.get("skill_demands", []) if label in SKILL_LABELS],
                answer_form=str(item.get("answer_form", "entity_or_short_span")),
                entity_pattern=str(item.get("entity_pattern", "single_entity")),
                difficulty=str(item.get("difficulty", "medium")),
                coverage_priority=max(1, min(5, int(item.get("coverage_priority", 3)))),
                short_rationale=str(item.get("short_rationale", ""))[:240],
                source="gpt_5_4",
            )
            result[annotation.group_id] = annotation
        for group in groups:
            if group["group_id"] not in result:
                result[group["group_id"]] = heuristic_group_annotation(group)
        return result
    except Exception:
        return {group["group_id"]: heuristic_group_annotation(group) for group in groups}


def annotate_groups(
    dataset: str,
    groups: list[dict[str, Any]],
    root_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_workers: int,
    group_batch_size: int,
    max_gpt_groups: int,
    overwrite_existing: bool,
) -> dict[str, GroupAnnotation]:
    annotation_dir = root_dir / "group_annotations"
    ensure_dir(annotation_dir)
    path = annotation_dir / f"{dataset}_group_annotations.jsonl"
    if path.exists() and not overwrite_existing:
        cached = {}
        for record in read_jsonl(path):
            cached[record["group_id"]] = GroupAnnotation(**record)
        if len(cached) == len(groups):
            return cached

    annotations: dict[str, GroupAnnotation] = {
        group["group_id"]: heuristic_group_annotation(group) for group in groups
    }
    if not api_key:
        records = []
        for group in groups:
            annotation = annotations[group["group_id"]]
            records.append(
                {
                    "group_id": annotation.group_id,
                    "reasoning_types": annotation.reasoning_types,
                    "skill_demands": annotation.skill_demands,
                    "answer_form": annotation.answer_form,
                    "entity_pattern": annotation.entity_pattern,
                    "difficulty": annotation.difficulty,
                    "coverage_priority": annotation.coverage_priority,
                    "short_rationale": annotation.short_rationale,
                    "source": annotation.source,
                }
            )
        write_jsonl(path, records)
        return annotations

    groups_for_gpt = select_groups_for_gpt(groups, max_gpt_groups)
    batches = [groups_for_gpt[index : index + group_batch_size] for index in range(0, len(groups_for_gpt), group_batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                annotate_group_batch,
                batch,
                base_url,
                api_key,
                model,
                reasoning_effort,
            )
            for batch in batches
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"annotate:{dataset}"):
            annotations.update(future.result())
    records = []
    for group in groups:
        annotation = annotations[group["group_id"]]
        records.append(
            {
                "group_id": annotation.group_id,
                "reasoning_types": annotation.reasoning_types,
                "skill_demands": annotation.skill_demands,
                "answer_form": annotation.answer_form,
                "entity_pattern": annotation.entity_pattern,
                "difficulty": annotation.difficulty,
                "coverage_priority": annotation.coverage_priority,
                "short_rationale": annotation.short_rationale,
                "source": annotation.source,
            }
        )
    write_jsonl(path, records)
    return annotations


def select_groups_for_gpt(groups: list[dict[str, Any]], max_gpt_groups: int) -> list[dict[str, Any]]:
    if max_gpt_groups <= 0 or len(groups) <= max_gpt_groups:
        return groups
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    def add_group(group: dict[str, Any]) -> None:
        if group["group_id"] not in selected_ids and len(selected) < max_gpt_groups:
            selected_ids.add(group["group_id"])
            selected.append(group)

    native_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hop_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entity_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flag_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        native_buckets[str(group["native_type"])].append(group)
        hop_buckets[str(group["hop_count"])].append(group)
        entity_buckets[str(group["signature"].get("entity_bin", "unknown"))].append(group)
        for flag in group.get("heuristic_flags", []):
            flag_buckets[flag].append(group)

    for bucket in list(native_buckets.values()) + list(hop_buckets.values()) + list(entity_buckets.values()):
        for group in sorted(bucket, key=lambda item: (-item["size"], item["group_id"]))[:8]:
            add_group(group)

    important_flags = ["multi_entity", "relation_chain", "same_attribute", "yes_no", "temporal", "numeric", "dense_entities", "comparison"]
    for flag in important_flags:
        for group in sorted(flag_buckets.get(flag, []), key=lambda item: (-item["size"], item["group_id"]))[:10]:
            add_group(group)

    remaining = [group for group in groups if group["group_id"] not in selected_ids]
    remaining.sort(
        key=lambda item: (
            -(heuristic_group_annotation(item).coverage_priority * 10 + math.log(item["size"] + 1, 2)),
            item["group_id"],
        )
    )
    for group in remaining:
        add_group(group)
        if len(selected) >= max_gpt_groups:
            break
    return selected


def attach_group_metadata(
    examples: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    annotations: dict[str, GroupAnnotation],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_lookup = {group["group_id"]: group for group in groups}
    signature_index = {
        json.dumps(group["signature"], ensure_ascii=False, sort_keys=True): group["group_id"]
        for group in groups
    }
    for example in examples:
        signature = json.dumps(json.loads(example["signature"]), ensure_ascii=False, sort_keys=True)
        group_id = signature_index[signature]
        records_by_group[group_id].append(example)
    augmented: dict[str, dict[str, Any]] = {}
    for group in groups:
        annotation = annotations[group["group_id"]]
        augmented[group["group_id"]] = {
            **group_lookup[group["group_id"]],
            "annotation": annotation,
        }
    return records_by_group, augmented


def target_size_for_dataset(requested: str, report: dict[str, Any]) -> int:
    if requested == "auto":
        return int(report["recommended_target_size"])
    return int(requested)


def atomic_labels(group: dict[str, Any]) -> list[str]:
    annotation: GroupAnnotation = group["annotation"]
    labels = [
        f"native:{group['native_type']}",
        f"hop:{group['hop_count']}",
        f"entity:{annotation.entity_pattern}",
        f"answer:{annotation.answer_form}",
    ]
    labels.extend(f"reasoning:{label}" for label in annotation.reasoning_types)
    labels.extend(f"skill:{label}" for label in annotation.skill_demands)
    return sorted(set(labels))


def group_priority(group: dict[str, Any], label_rarity: dict[str, float]) -> float:
    annotation: GroupAnnotation = group["annotation"]
    rarity_bonus = sum(label_rarity.get(label, 0.0) for label in atomic_labels(group))
    size_bonus = math.log(group["size"] + 1, 2)
    return (annotation.coverage_priority * 2.5) + rarity_bonus + (0.15 * size_bonus)


def choose_examples_from_group(
    group_id: str,
    group_records: dict[str, list[dict[str, Any]]],
    selected_ids: set[str],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    available = [record for record in group_records[group_id] if record["id"] not in selected_ids]
    if not available:
        return []
    rng = random.Random(f"{seed}:{group_id}:{len(selected_ids)}")
    available.sort(key=lambda item: (item["question"], item["id"]))
    rng.shuffle(available)
    return available[:count]


def coverage_sample(
    dataset: str,
    examples: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    annotations: dict[str, GroupAnnotation],
    target_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_by_group, augmented_groups = attach_group_metadata(examples, groups, annotations)
    label_to_groups: dict[str, list[str]] = defaultdict(list)
    label_freq = Counter()
    for group_id, group in augmented_groups.items():
        for label in atomic_labels(group):
            label_to_groups[label].append(group_id)
            label_freq[label] += group["size"]
    label_rarity = {label: 1.0 / math.sqrt(freq) for label, freq in label_freq.items()}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    label_counts = Counter()
    min_label_quota = max(24, min(80, target_size // 100))
    for label in sorted(label_to_groups, key=lambda item: (label_freq[item], item)):
        while label_counts[label] < min_label_quota and len(selected) < target_size:
            candidate_groups = sorted(
                label_to_groups[label],
                key=lambda group_id: (-group_priority(augmented_groups[group_id], label_rarity), group_id),
            )
            added = False
            for group_id in candidate_groups:
                chosen = choose_examples_from_group(group_id, records_by_group, selected_ids, 1, seed)
                if not chosen:
                    continue
                record = chosen[0]
                selected.append(record)
                selected_ids.add(record["id"])
                for atomic_label in atomic_labels(augmented_groups[group_id]):
                    label_counts[atomic_label] += 1
                added = True
                break
            if not added:
                break
    ranked_groups = sorted(
        augmented_groups.values(),
        key=lambda item: (-group_priority(item, label_rarity), item["group_id"]),
    )
    cycle_index = 0
    while len(selected) < target_size and ranked_groups:
        group = ranked_groups[cycle_index % len(ranked_groups)]
        chosen = choose_examples_from_group(group["group_id"], records_by_group, selected_ids, 1, seed)
        if chosen:
            record = chosen[0]
            selected.append(record)
            selected_ids.add(record["id"])
            for atomic_label in atomic_labels(group):
                label_counts[atomic_label] += 1
        cycle_index += 1
        if cycle_index > (len(ranked_groups) * max(4, target_size)):
            break
    selected.sort(key=lambda item: (item["native_type"], item["id"]))
    report = {
        "dataset": dataset,
        "target_size": target_size,
        "selected_size": len(selected),
        "label_coverage": dict(label_counts.most_common()),
        "num_groups": len(groups),
        "num_groups_selected_from": len({item["signature"] for item in selected}),
    }
    return selected, report


def normalize_question(question: str) -> str:
    return SPACE_RE.sub(" ", question).strip().lower()


def suspicious_reasons(record: dict[str, Any]) -> list[str]:
    question = record["question"].strip()
    lowered = question.lower()
    reasons: list[str] = []
    if not record.get("primary_answer", "").strip():
        reasons.append("empty_answer")
    if len(question) < 20:
        reasons.append("very_short_text")
    if len(question.split()) < 4:
        reasons.append("too_few_tokens")
    if lowered.startswith(("is a ", "is an ", "are a ", "are an ", "was a ", "was an ", "were a ", "were an ")):
        reasons.append("fragment_after_aux")
    if first_wh_word(question) == "other" and not question.endswith("?") and (not question or question[:1].islower()):
        reasons.append("non_interrogative_fragment")
    return sorted(set(reasons))


def severe_suspicious(reasons: list[str]) -> bool:
    severe = {"empty_answer", "fragment_after_aux", "non_interrogative_fragment"}
    return any(reason in severe for reason in reasons)


def quality_system_prompt() -> str:
    return (
        "You review candidate training questions for multi-hop QA skill learning. "
        "Return valid JSON only. "
        "For each item, decide keep=true if the question is a valid, self-contained QA example. "
        "Set keep=false for malformed question fragments, broken wording, answer conflicts, or clearly low-quality samples. "
        "Use concise reasons."
    )


def quality_user_prompt(items: list[dict[str, Any]]) -> str:
    payload = {
        "task": "Review suspicious candidate questions.",
        "items": items,
        "output_schema": {
            "reviews": [
                {
                    "record_id": "string",
                    "keep": True,
                    "reason": "brief",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def review_quality_batch(
    records: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, QualityReview]:
    if not api_key:
        return {
            record["id"]: QualityReview(
                record_id=record["id"],
                keep=not severe_suspicious(suspicious_reasons(record)),
                reason="heuristic_fallback",
                source="heuristic_fallback",
            )
            for record in records
        }
    payload_items = [
        {
            "record_id": record["id"],
            "question": record["question"],
            "answer": record.get("primary_answer", ""),
            "native_type": record.get("native_type", ""),
            "hop_count": record.get("hop_count", 0),
            "flags": record.get("flags", []),
            "suspicious_reasons": suspicious_reasons(record),
        }
        for record in records
    ]
    try:
        payload = responses_request(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            system_prompt=quality_system_prompt(),
            user_prompt=quality_user_prompt(payload_items),
        )
        parsed = parse_json_payload(response_text(payload))
        reviews = parsed["reviews"] if isinstance(parsed, dict) else parsed
        result: dict[str, QualityReview] = {}
        for item in reviews:
            review = QualityReview(
                record_id=str(item["record_id"]),
                keep=bool(item.get("keep", True)),
                reason=str(item.get("reason", ""))[:240],
                source="gpt_5_4",
            )
            result[review.record_id] = review
        for record in records:
            if record["id"] not in result:
                result[record["id"]] = QualityReview(
                    record_id=record["id"],
                    keep=not severe_suspicious(suspicious_reasons(record)),
                    reason="heuristic_fallback",
                    source="heuristic_fallback",
                )
        return result
    except Exception:
        return {
            record["id"]: QualityReview(
                record_id=record["id"],
                keep=not severe_suspicious(suspicious_reasons(record)),
                reason="heuristic_fallback",
                source="heuristic_fallback",
            )
            for record in records
        }


def review_suspicious_records(
    suspicious_records: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_workers: int,
    batch_size: int = 24,
) -> dict[str, QualityReview]:
    if not suspicious_records:
        return {}
    batches = [suspicious_records[index : index + batch_size] for index in range(0, len(suspicious_records), batch_size)]
    reviews: dict[str, QualityReview] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                review_quality_batch,
                batch,
                base_url,
                api_key,
                model,
                reasoning_effort,
            )
            for batch in batches
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="quality_review"):
            reviews.update(future.result())
    return reviews


def build_ranked_groups(groups: list[dict[str, Any]], annotations: dict[str, GroupAnnotation]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    records_by_group, augmented_groups = attach_group_metadata([], groups, annotations)
    del records_by_group
    label_to_groups: dict[str, list[str]] = defaultdict(list)
    label_freq = Counter()
    for group_id, group in augmented_groups.items():
        for label in atomic_labels(group):
            label_to_groups[label].append(group_id)
            label_freq[label] += group["size"]
    label_rarity = {label: 1.0 / math.sqrt(freq) for label, freq in label_freq.items()}
    ranked_groups = sorted(
        augmented_groups.values(),
        key=lambda item: (-group_priority(item, label_rarity), item["group_id"]),
    )
    return augmented_groups, ranked_groups


def clean_and_refill_sample(
    dataset: str,
    examples: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    annotations: dict[str, GroupAnnotation],
    initial_selected: list[dict[str, Any]],
    target_size: int,
    seed: int,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_by_group, augmented_groups = attach_group_metadata(examples, groups, annotations)
    label_to_groups: dict[str, list[str]] = defaultdict(list)
    label_freq = Counter()
    for group_id, group in augmented_groups.items():
        for label in atomic_labels(group):
            label_to_groups[label].append(group_id)
            label_freq[label] += group["size"]
    label_rarity = {label: 1.0 / math.sqrt(freq) for label, freq in label_freq.items()}
    ranked_groups = sorted(
        augmented_groups.values(),
        key=lambda item: (-group_priority(item, label_rarity), item["group_id"]),
    )

    deduped_by_id: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dropped_duplicate_id = 0
    for record in initial_selected:
        if record["id"] in seen_ids:
            dropped_duplicate_id += 1
            continue
        seen_ids.add(record["id"])
        deduped_by_id.append(record)

    question_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduped_by_id:
        question_groups[normalize_question(record["question"])].append(record)

    kept_after_question_dedup: list[dict[str, Any]] = []
    rejected_questions: set[str] = set()
    dropped_duplicate_question = 0
    dropped_conflicting_question = 0
    for normalized_question_text, records in question_groups.items():
        answers = {record.get("primary_answer", "").strip() for record in records}
        if len(records) == 1:
            kept_after_question_dedup.append(records[0])
            continue
        if len(answers) > 1:
            dropped_conflicting_question += len(records)
            rejected_questions.add(normalized_question_text)
            continue
        kept_after_question_dedup.append(records[0])
        dropped_duplicate_question += len(records) - 1

    suspicious = [record for record in kept_after_question_dedup if suspicious_reasons(record)]
    quality_reviews = review_suspicious_records(
        suspicious_records=suspicious,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        max_workers=max_workers,
    )
    cleaned: list[dict[str, Any]] = []
    dropped_quality = 0
    quality_review_sources = Counter()
    for record in kept_after_question_dedup:
        review = quality_reviews.get(record["id"])
        if review is None:
            cleaned.append(record)
            continue
        quality_review_sources[review.source] += 1
        if review.keep:
            cleaned.append(record)
        else:
            dropped_quality += 1
            rejected_questions.add(normalize_question(record["question"]))

    selected_ids = {record["id"] for record in cleaned}
    selected_questions = {normalize_question(record["question"]) for record in cleaned}
    refill_attempts = 0
    cycle_index = 0
    while len(cleaned) < target_size and ranked_groups:
        group = ranked_groups[cycle_index % len(ranked_groups)]
        candidate_list = records_by_group[group["group_id"]]
        refill_attempts += 1
        chosen_record = None
        for candidate in candidate_list:
            normalized = normalize_question(candidate["question"])
            if candidate["id"] in selected_ids:
                continue
            if normalized in selected_questions or normalized in rejected_questions:
                continue
            if severe_suspicious(suspicious_reasons(candidate)):
                continue
            chosen_record = candidate
            break
        if chosen_record is not None:
            cleaned.append(chosen_record)
            selected_ids.add(chosen_record["id"])
            selected_questions.add(normalize_question(chosen_record["question"]))
        cycle_index += 1
        if refill_attempts > len(ranked_groups) * max(6, target_size):
            break

    cleaned.sort(key=lambda item: (item["native_type"], item["id"]))
    report = {
        "dataset": dataset,
        "initial_selected_size": len(initial_selected),
        "cleaned_selected_size": len(cleaned),
        "dropped_duplicate_id": dropped_duplicate_id,
        "dropped_duplicate_question": dropped_duplicate_question,
        "dropped_conflicting_question": dropped_conflicting_question,
        "suspicious_reviewed": len(suspicious),
        "dropped_quality": dropped_quality,
        "quality_review_sources": dict(quality_review_sources),
        "refill_added": max(0, len(cleaned) - (len(kept_after_question_dedup) - dropped_quality)),
    }
    return cleaned, report


def materialize_full_records(dataset: str, selected_ids: set[str], root_dir: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in iter_cache_rows(DATASET_SPECS[dataset]["train_paths"]):
        row_id = str(row["id"])
        if row_id in selected_ids:
            selected.append(row)
    selected.sort(key=lambda item: str(item["id"]))
    sample_dir = root_dir / "samples" / dataset
    ensure_dir(sample_dir)
    write_jsonl(sample_dir / "train_sample_full.jsonl", selected)
    return selected


def save_selected_light(dataset: str, selected: list[dict[str, Any]], root_dir: Path) -> None:
    sample_dir = root_dir / "samples" / dataset
    ensure_dir(sample_dir)
    write_jsonl(sample_dir / "train_sample_light.jsonl", selected)


def save_sampling_report(dataset: str, report: dict[str, Any], root_dir: Path) -> None:
    report_dir = root_dir / "reports"
    ensure_dir(report_dir)
    (report_dir / f"{dataset}_sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage-oriented sampler for multihop QA training subsets.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--target-size", default="auto")
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--group-representatives", type=int, default=4)
    parser.add_argument("--group-batch-size", type=int, default=12)
    parser.add_argument("--max-gpt-groups", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--candidate-buffer-ratio", type=float, default=1.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--model-base-url", default="https://w.ciykj.cn")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.root_dir)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    summary_path = args.root_dir / "reports" / "sampling_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("datasets", {})
    else:
        summary = {"datasets": {}}
    for dataset in args.datasets:
        examples, profile_report = build_light_profiles(dataset, args.root_dir, overwrite_existing=args.overwrite_existing)
        groups = build_groups(examples, representatives=args.group_representatives, seed=args.seed)
        annotations = annotate_groups(
            dataset=dataset,
            groups=groups,
            root_dir=args.root_dir,
            base_url=args.model_base_url,
            api_key=api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_workers=args.max_workers,
            group_batch_size=args.group_batch_size,
            max_gpt_groups=args.max_gpt_groups,
            overwrite_existing=args.overwrite_existing,
        )
        target_size = target_size_for_dataset(args.target_size, profile_report)
        selected_light_raw, sampling_report = coverage_sample(
            dataset=dataset,
            examples=examples,
            groups=groups,
            annotations=annotations,
            target_size=target_size,
            seed=args.seed,
        )
        selected_light, cleaning_report = clean_and_refill_sample(
            dataset=dataset,
            examples=examples,
            groups=groups,
            annotations=annotations,
            initial_selected=selected_light_raw,
            target_size=target_size,
            seed=args.seed,
            base_url=args.model_base_url,
            api_key=api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_workers=args.max_workers,
        )
        save_selected_light(dataset, selected_light, args.root_dir)
        materialize_full_records(dataset, {record["id"] for record in selected_light}, args.root_dir)
        save_sampling_report(dataset, {**profile_report, **sampling_report, **cleaning_report}, args.root_dir)
        summary["datasets"][dataset] = {
            "recommended_target_size": profile_report["recommended_target_size"],
            "selected_size": len(selected_light),
            "num_signatures": profile_report["num_signatures"],
        }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
