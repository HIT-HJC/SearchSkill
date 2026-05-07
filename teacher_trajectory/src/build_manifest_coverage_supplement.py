from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from common import (
    build_metadata_summary,
    build_task_family,
    dump_json,
    dump_jsonl,
    load_jsonl,
    load_skill_ids,
    repo_root,
    suggest_primary_skills,
    suggest_support_skills,
)


DATASETS = ["hotpotqa", "2wiki", "musique", "nq", "triviaqa"]
MULTIHOP_DATASETS = {"hotpotqa", "2wiki", "musique"}
SINGLEHOP_DATASETS = {"nq", "triviaqa"}

DEFAULT_COUNTS = {
    "hotpotqa": 320,
    "2wiki": 420,
    "musique": 560,
    "nq": 220,
    "triviaqa": 220,
}

DEFAULT_EXCLUDE_PATHS = [
    "teacher_trajectory/runs/canonical_teacher_set/all/trajectories.raw.jsonl",
    "supervised_finetuning/data/sft_v3/selected_trajectories.jsonl",
    "supervised_finetuning/data/sft_v3_qwen2.5_7b_base/selected_trajectories.jsonl",
]

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
DIGIT_RE = re.compile(r"\d")
COMPARISON_RE = re.compile(
    r"\b(compare|same|both|older|younger|earlier|later|more|less|higher|lower|larger|smaller|first|last)\b",
    re.I,
)
TEMPORAL_RE = re.compile(r"\b(when|year|date|before|after|earlier|later|born|died|released|founded|season)\b", re.I)
NUMERIC_RE = re.compile(r"\b(how many|how much|number of|count|population|age|height|score|total)\b", re.I)
YESNO_RE = re.compile(r"^(is|are|was|were|do|does|did|has|have|had|can|could|would|should)\b", re.I)
ALIAS_RE = re.compile(r"\b(real name|full name|nickname|also known as|formerly|stage name|alias)\b", re.I)
KINSHIP_RE = re.compile(r"\b(mother|father|spouse|wife|husband|daughter|son|grandfather|grandmother)\b", re.I)
RELATION_RE = re.compile(
    r"\b(director|author|founder|creator|composer|performer|writer|actor|actress|producer|country|city|state|county|birthplace|school|university|alma mater)\b",
    re.I,
)
CAPITALIZED_SPAN_RE = re.compile(r"(?:[A-Z][\w'&.-]*)(?:\s+(?:[A-Z][\w'&.-]*))*")
SPACE_RE = re.compile(r"\s+")


QUOTA_LABELS = {
    "hotpotqa": {
        "native:bridge": 140,
        "native:comparison": 115,
        "tag:comparison": 110,
        "tag:temporal": 90,
        "tag:numeric": 85,
        "tag:yes_no": 55,
        "tag:same_attribute": 45,
        "tag:relation_chain": 45,
        "combo:comparison_temporal": 35,
        "combo:comparison_numeric": 35,
        "answer:boolean": 40,
        "answer:number": 40,
        "answer:year_or_date": 45,
    },
    "2wiki": {
        "native:compositional": 115,
        "native:bridge_comparison": 100,
        "native:inference": 55,
        "native:comparison": 95,
        "hop:4": 100,
        "tag:comparison": 130,
        "tag:temporal": 105,
        "tag:numeric": 120,
        "tag:yes_no": 100,
        "tag:same_attribute": 100,
        "tag:relation_chain": 115,
        "combo:yes_no_comparison": 70,
        "combo:multi_entity_relation": 115,
    },
    "musique": {
        "native:decomp_2": 145,
        "native:decomp_4": 150,
        "native:decomp_3": 190,
        "hop:4": 150,
        "hop:3": 190,
        "tag:comparison": 85,
        "tag:temporal": 165,
        "tag:numeric": 210,
        "tag:relation_chain": 85,
        "combo:long_hop_temporal": 110,
        "combo:long_hop_numeric": 130,
        "combo:short_multihop": 70,
    },
    "nq": {
        "answer:entity_or_short_span": 75,
        "answer:number": 55,
        "answer:year_or_date": 55,
        "tag:alias_or_disambiguation": 35,
        "tag:temporal": 65,
        "tag:numeric": 75,
        "tag:quoted_span": 45,
        "tag:person": 45,
        "tag:location": 45,
        "tag:superlative": 30,
        "question:short": 65,
    },
    "triviaqa": {
        "answer:entity_or_short_span": 75,
        "answer:number": 50,
        "answer:year_or_date": 65,
        "tag:alias_or_disambiguation": 65,
        "tag:temporal": 75,
        "tag:numeric": 80,
        "tag:quoted_span": 55,
        "tag:person": 55,
        "tag:location": 55,
        "question:short": 65,
    },
}

LABEL_CAPS = {
    "hotpotqa": {
        "native:comparison": 160,
        "tag:yes_no": 120,
        "answer:boolean": 115,
        "tag:comparison": 220,
    },
    "2wiki": {
        "native:bridge_comparison": 150,
        "native:comparison": 150,
        "tag:yes_no": 165,
        "answer:boolean": 165,
        "tag:comparison": 280,
    },
    "musique": {
        "native:decomp_4": 200,
        "native:decomp_3": 250,
        "tag:comparison": 190,
    },
    "nq": {
        "tag:comparison": 65,
        "tag:relation_chain": 55,
    },
    "triviaqa": {
        "tag:comparison": 65,
        "tag:relation_chain": 55,
    },
}

PRIORITY_LABEL_WEIGHTS = {
    "native:bridge_comparison": 4.0,
    "native:decomp_4": 4.0,
    "native:decomp_3": 3.0,
    "native:inference": 3.0,
    "hop:4": 3.0,
    "hop:3": 2.2,
    "combo:short_multihop": 2.5,
    "combo:long_hop_temporal": 2.5,
    "combo:long_hop_numeric": 2.5,
    "combo:yes_no_comparison": 2.3,
    "combo:multi_entity_relation": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supplemental coverage-oriented manifest for teacher trajectory generation."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skill-bank-path",
        type=Path,
        default=repo_root() / "skill_bank" / "round_4_musique" / "outputs" / "final_skill_bank.md",
    )
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--hotpot-count", type=int, default=DEFAULT_COUNTS["hotpotqa"])
    parser.add_argument("--2wiki-count", dest="two_wiki_count", type=int, default=DEFAULT_COUNTS["2wiki"])
    parser.add_argument("--musique-count", type=int, default=DEFAULT_COUNTS["musique"])
    parser.add_argument("--nq-count", type=int, default=DEFAULT_COUNTS["nq"])
    parser.add_argument("--triviaqa-count", type=int, default=DEFAULT_COUNTS["triviaqa"])
    parser.add_argument(
        "--multihop-source",
        choices=["full", "pruned"],
        default="full",
        help="Use the larger data_preparation samples or the trajectory_pruning subset for multihop datasets.",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        help="Extra jsonl path to exclude by dataset/id and normalized question.",
    )
    parser.add_argument("--cap-per-signature", type=int, default=4)
    parser.add_argument("--random-fill-ratio", type=float, default=0.18)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return SPACE_RE.sub(" ", value).strip()


def answer_form(answer: str) -> str:
    value = str(answer or "").strip()
    lowered = value.lower()
    if lowered in {"yes", "no"}:
        return "boolean"
    if YEAR_RE.search(value):
        return "year_or_date"
    if DIGIT_RE.search(value):
        return "number"
    if "," in value:
        return "list_like"
    if len(value.split()) >= 3:
        return "long_span"
    return "entity_or_short_span"


def primary_answer(row: Dict[str, Any]) -> str:
    for key in ("primary_answer", "final_answer"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    answers = row.get("gold_answers") or row.get("golden_answers") or row.get("gold") or []
    for answer in answers:
        value = str(answer or "").strip()
        if value:
            return value
    return ""


def question_flags(question: str, native_type: str, answer: str) -> Set[str]:
    lowered = question.lower()
    flags: Set[str] = set()
    if COMPARISON_RE.search(question) or "comparison" in native_type:
        flags.add("comparison")
    if TEMPORAL_RE.search(question) or YEAR_RE.search(answer):
        flags.add("temporal")
    if NUMERIC_RE.search(question) or DIGIT_RE.search(answer):
        flags.add("numeric")
    if YESNO_RE.search(question):
        flags.add("yes_no")
    if "same " in lowered or "from the same" in lowered:
        flags.add("same_attribute")
    if ALIAS_RE.search(question):
        flags.add("alias_or_disambiguation")
    if KINSHIP_RE.search(question):
        flags.add("kinship")
    if RELATION_RE.search(question) or "bridge" in native_type or "decomp" in native_type:
        flags.add("relation_chain")
    if "which" in lowered or "whose" in lowered or "what" in lowered:
        flags.add("lookup")
    if "first" in lowered or "largest" in lowered or "smallest" in lowered or "highest" in lowered or "oldest" in lowered:
        flags.add("superlative")
    entity_count = len({span.strip() for span in CAPITALIZED_SPAN_RE.findall(question) if len(span.strip()) > 1})
    if entity_count >= 3:
        flags.add("multi_entity")
    if entity_count >= 5:
        flags.add("dense_entities")
    return flags


def coverage_labels(row: Dict[str, Any]) -> Set[str]:
    dataset = str(row.get("dataset") or "")
    question = str(row.get("question") or "").strip()
    native_type = str(row.get("native_type") or (row.get("metadata_summary") or {}).get("question_type") or "unknown")
    hop_count = int(row.get("hop_count") or (row.get("metadata_summary") or {}).get("estimated_hops") or 1)
    if dataset in SINGLEHOP_DATASETS:
        hop_count = 1
    answer = primary_answer(row)
    form = str(row.get("answer_form_hint") or answer_form(answer))
    inherited_flags = {str(flag) for flag in row.get("flags") or [] if str(flag).strip()}
    derived_flags = question_flags(question, native_type, answer)
    flags = {normalize_flag(flag) for flag in inherited_flags} | derived_flags
    token_count = len(question.split())

    labels = {
        f"dataset:{dataset}",
        f"native:{native_type}",
        f"hop:{min(hop_count, 4)}",
        f"answer:{form}",
    }
    if token_count <= 9:
        labels.add("question:short")
    if token_count >= 18:
        labels.add("question:long")
    for flag in flags:
        labels.add(f"tag:{flag}")
    if dataset in MULTIHOP_DATASETS and hop_count >= 2 and token_count <= 9:
        labels.add("combo:short_multihop")
    if dataset in MULTIHOP_DATASETS and hop_count >= 3 and "temporal" in flags:
        labels.add("combo:long_hop_temporal")
    if dataset in MULTIHOP_DATASETS and hop_count >= 3 and "numeric" in flags:
        labels.add("combo:long_hop_numeric")
    if "comparison" in flags and "temporal" in flags:
        labels.add("combo:comparison_temporal")
    if "comparison" in flags and "numeric" in flags:
        labels.add("combo:comparison_numeric")
    if "yes_no" in flags and "comparison" in flags:
        labels.add("combo:yes_no_comparison")
    if "multi_entity" in flags and "relation_chain" in flags:
        labels.add("combo:multi_entity_relation")
    if form in {"boolean", "year_or_date", "number", "entity_or_short_span"}:
        labels.add("tag:answer_boundary_sensitive")
    return labels


def normalize_flag(flag: str) -> str:
    value = str(flag or "").strip().lower()
    mapping = {
        "alias": "alias_or_disambiguation",
        "verification": "verification",
        "time_anchor": "temporal",
        "direct_lookup": "lookup",
        "longer_hop": "long_hop",
        "location": "location",
        "person": "person",
        "quoted_span": "quoted_span",
        "title_work": "title_work",
        "organization": "organization",
    }
    return mapping.get(value, value)


def sample_pair_paths(dataset: str, multihop_source: str) -> Tuple[Path, Path]:
    samples_root = repo_root() / "data_preparation" / "samples"
    if dataset in MULTIHOP_DATASETS and multihop_source == "pruned":
        base = samples_root / "trajectory_pruning" / dataset
    else:
        base = samples_root / dataset
    return base / "train_sample_light.jsonl", base / "train_sample_full.jsonl"


def load_candidate_pool(
    dataset: str,
    *,
    multihop_source: str,
    legal_skill_ids: Sequence[str],
    excluded_ids: Set[Tuple[str, str]],
    excluded_questions: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    light_path, full_path = sample_pair_paths(dataset, multihop_source)
    light_by_id = {str(row["id"]): row for row in load_jsonl(light_path)}
    records: List[Dict[str, Any]] = []
    skipped_id = 0
    skipped_question = 0
    for full_row in load_jsonl(full_path):
        row_id = str(full_row.get("id") or "")
        light = light_by_id.get(row_id, {})
        question = str(full_row.get("question") or light.get("question") or "").strip()
        norm_question = normalize_text(question)
        if (dataset, row_id) in excluded_ids:
            skipped_id += 1
            continue
        if norm_question in excluded_questions:
            skipped_question += 1
            continue
        merged: Dict[str, Any] = {}
        merged.update(light)
        merged.update(full_row)
        merged["id"] = row_id
        merged["dataset"] = dataset
        merged["question"] = question
        merged["sample_origin"] = "coverage_supplement_full" if dataset in SINGLEHOP_DATASETS or multihop_source == "full" else "coverage_supplement_pruned"
        merged["task_family"] = build_task_family(dataset)
        merged["gold_answers"] = full_row.get("golden_answers") or full_row.get("gold_answers") or []
        merged["metadata_summary"] = build_metadata_summary(merged)
        merged["candidate_primary_skills"] = suggest_primary_skills(merged, legal_skill_ids)
        merged["suggested_support_skills"] = suggest_support_skills(legal_skill_ids)
        labels = sorted(coverage_labels(merged))
        merged["coverage_labels"] = labels
        records.append(merged)
    return records, {
        "light_path": str(light_path),
        "full_path": str(full_path),
        "loaded": len(records),
        "skipped_existing_id": skipped_id,
        "skipped_existing_question": skipped_question,
    }


def load_exclusions(paths: Sequence[Path]) -> Tuple[Set[Tuple[str, str]], Set[str], Dict[str, Any]]:
    excluded_ids: Set[Tuple[str, str]] = set()
    excluded_questions: Set[str] = set()
    stats: Dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            stats[str(path)] = {"exists": False, "rows": 0}
            continue
        rows = 0
        for row in load_jsonl(path):
            rows += 1
            dataset = str(row.get("dataset") or "")
            row_id = str(row.get("id") or row.get("source_example_id") or "")
            if dataset and row_id:
                excluded_ids.add((dataset, row_id))
            question = normalize_text(str(row.get("question") or ""))
            if question:
                excluded_questions.add(question)
        stats[str(path)] = {"exists": True, "rows": rows}
    return excluded_ids, excluded_questions, stats


def label_frequency(records: Sequence[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in records:
        counter.update(row.get("coverage_labels") or [])
    return counter


def selected_label_count(selected: Sequence[Dict[str, Any]], label: str) -> int:
    return sum(1 for row in selected if label in set(row.get("coverage_labels") or []))


def candidate_score(
    row: Dict[str, Any],
    *,
    quotas: Dict[str, int],
    selected_counts: Counter[str],
    label_counts: Counter[str],
    seed: int,
) -> Tuple[float, float, int]:
    labels = set(row.get("coverage_labels") or [])
    unmet_score = 0.0
    for label, quota in quotas.items():
        if label not in labels or selected_counts[label] >= quota:
            continue
        weight = PRIORITY_LABEL_WEIGHTS.get(label, 1.0)
        unmet_score += weight * (1.0 + (quota - selected_counts[label]) / max(1, quota))
    rare_score = sum(1.0 / math.sqrt(label_counts[label]) for label in labels if label_counts[label] > 0)
    jitter = stable_hash(f"{seed}:{row.get('dataset')}:{row.get('id')}") % 100000
    return (unmet_score, rare_score, -jitter)


def select_dataset_records(
    records: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    target_count: int,
    seed: int,
    cap_per_signature: int,
    random_fill_ratio: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed + stable_hash(dataset) % 100000)
    quotas = dict(QUOTA_LABELS.get(dataset, {}))
    caps = dict(LABEL_CAPS.get(dataset, {}))
    label_counts = label_frequency(records)
    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    selected_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    records_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        for label in row.get("coverage_labels") or []:
            records_by_label[label].append(row)

    def can_take(
        row: Dict[str, Any],
        *,
        strict_signature_cap: bool = True,
        strict_label_caps: bool = True,
    ) -> bool:
        row_id = str(row.get("id"))
        if row_id in selected_ids:
            return False
        signature = str(row.get("signature") or row_id)
        if strict_signature_cap and signature_counts[signature] >= cap_per_signature:
            return False
        if strict_label_caps:
            labels = set(row.get("coverage_labels") or [])
            for label, cap in caps.items():
                if label in labels and selected_counts[label] >= cap:
                    return False
        return True

    def take(row: Dict[str, Any]) -> None:
        selected.append(row)
        selected_ids.add(str(row.get("id")))
        signature_counts[str(row.get("signature") or row.get("id"))] += 1
        selected_counts.update(row.get("coverage_labels") or [])

    # First pass: satisfy coarse coverage quotas with diverse examples.
    for label, quota in sorted(quotas.items(), key=lambda item: (-PRIORITY_LABEL_WEIGHTS.get(item[0], 1.0), item[0])):
        if len(selected) >= target_count:
            break
        candidates = list(records_by_label.get(label, []))
        rng.shuffle(candidates)
        while selected_counts[label] < quota and len(selected) < target_count:
            candidates = [row for row in candidates if can_take(row)]
            if not candidates:
                candidates = [
                    row for row in records_by_label.get(label, []) if can_take(row, strict_signature_cap=False)
                ]
            if not candidates:
                candidates = [
                    row
                    for row in records_by_label.get(label, [])
                    if can_take(row, strict_signature_cap=False, strict_label_caps=False)
                ]
            if not candidates:
                break
            candidates.sort(
                key=lambda row: candidate_score(
                    row,
                    quotas=quotas,
                    selected_counts=selected_counts,
                    label_counts=label_counts,
                    seed=seed,
                ),
                reverse=True,
            )
            take(candidates.pop(0))

    # Second pass: keep a random/regular slice so the supplement does not become too narrow.
    random_target = min(target_count, int(round(target_count * max(0.0, min(0.5, random_fill_ratio)))))
    general_pool = [row for row in records if can_take(row)]
    rng.shuffle(general_pool)
    while len(selected) < random_target and general_pool:
        take(general_pool.pop())

    # Final pass: fill remaining slots by unmet coverage and rarity.
    while len(selected) < target_count:
        candidates = [row for row in records if can_take(row)]
        if not candidates:
            candidates = [row for row in records if can_take(row, strict_signature_cap=False)]
        if not candidates:
            candidates = [row for row in records if can_take(row, strict_signature_cap=False, strict_label_caps=False)]
        if not candidates:
            break
        candidates.sort(
            key=lambda row: candidate_score(
                row,
                quotas=quotas,
                selected_counts=selected_counts,
                label_counts=label_counts,
                seed=seed,
            ),
            reverse=True,
        )
        take(candidates[0])

    coverage = {
        label: {
            "quota": quota,
            "available": label_counts.get(label, 0),
            "selected": selected_counts.get(label, 0),
            "met": selected_counts.get(label, 0) >= min(quota, label_counts.get(label, 0)),
        }
        for label, quota in quotas.items()
    }
    return selected, {
        "target_count": target_count,
        "selected_count": len(selected),
        "candidate_count": len(records),
        "cap_per_signature": cap_per_signature,
        "random_fill_ratio": random_fill_ratio,
        "quota_coverage": coverage,
        "label_caps": caps,
        "selected_label_top": dict(selected_counts.most_common(80)),
        "candidate_label_top": dict(label_counts.most_common(80)),
        "signature_count": len(signature_counts),
    }


def path_from_repo(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "sample_origin": row["sample_origin"],
        "task_family": row["task_family"],
        "question": row["question"],
        "gold_answers": row.get("gold_answers", []),
        "metadata": row.get("metadata", {}),
        "native_type": row.get("native_type"),
        "hop_count": row.get("hop_count"),
        "wh_word": row.get("wh_word"),
        "entity_bin": row.get("entity_bin"),
        "token_bin": row.get("token_bin"),
        "answer_form_hint": row.get("answer_form_hint") or answer_form(primary_answer(row)),
        "flags": row.get("flags", []),
        "signature": row.get("signature"),
        "metadata_summary": row.get("metadata_summary", {}),
        "candidate_primary_skills": row.get("candidate_primary_skills", []),
        "suggested_support_skills": row.get("suggested_support_skills", []),
        "coverage_labels": row.get("coverage_labels", []),
    }


def main() -> None:
    args = parse_args()
    counts = {
        "hotpotqa": args.hotpot_count,
        "2wiki": args.two_wiki_count,
        "musique": args.musique_count,
        "nq": args.nq_count,
        "triviaqa": args.triviaqa_count,
    }
    exclude_paths = [path_from_repo(value) for value in DEFAULT_EXCLUDE_PATHS]
    exclude_paths.extend(path_from_repo(value) for value in args.exclude_path)
    excluded_ids, excluded_questions, exclusion_stats = load_exclusions(exclude_paths)
    legal_skill_ids = load_skill_ids(args.skill_bank_path)

    manifest: List[Dict[str, Any]] = []
    dataset_summaries: Dict[str, Any] = {}
    source_stats: Dict[str, Any] = {}
    for dataset in DATASETS:
        count = int(counts.get(dataset, 0))
        if count <= 0:
            continue
        pool, stats = load_candidate_pool(
            dataset,
            multihop_source=args.multihop_source,
            legal_skill_ids=legal_skill_ids,
            excluded_ids=excluded_ids,
            excluded_questions=excluded_questions,
        )
        source_stats[dataset] = stats
        selected, summary = select_dataset_records(
            pool,
            dataset=dataset,
            target_count=count,
            seed=args.seed,
            cap_per_signature=args.cap_per_signature,
            random_fill_ratio=args.random_fill_ratio,
        )
        manifest.extend(compact_row(row) for row in selected)
        dataset_summaries[dataset] = summary

    manifest.sort(key=lambda row: (row["dataset"], row["id"]))
    output_dir = args.output_dir
    if not args.dry_run:
        dump_jsonl(output_dir / "manifest.jsonl", manifest)
    overall_labels: Counter[str] = Counter()
    for row in manifest:
        overall_labels.update(row.get("coverage_labels") or [])
    summary = {
        "script": "build_manifest_coverage_supplement.py",
        "purpose": "supplemental train-only coverage manifest for teacher trajectory generation",
        "output_dir": str(output_dir),
        "skill_bank_path": str(args.skill_bank_path),
        "seed": args.seed,
        "multihop_source": args.multihop_source,
        "total_examples": len(manifest),
        "counts_requested": counts,
        "counts_selected": dict(Counter(row["dataset"] for row in manifest)),
        "source_stats": source_stats,
        "exclusion_stats": exclusion_stats,
        "excluded_id_count": len(excluded_ids),
        "excluded_question_count": len(excluded_questions),
        "dataset_summaries": dataset_summaries,
        "overall_label_top": dict(overall_labels.most_common(120)),
        "notes": [
            "Only train-pool questions are sampled.",
            "Existing canonical_teacher_set and sft_v3 questions are excluded by dataset/id and normalized question.",
            "Labels are coarse coverage labels from dataset metadata, question cues, and answer form.",
        ],
    }
    if not args.dry_run:
        dump_json(output_dir / "manifest_summary.json", summary)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
