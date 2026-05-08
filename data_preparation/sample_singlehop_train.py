#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "nq": {
        "train_path": "__HF_DATA_ROOT__/data/nq/train.jsonl",
        "eval_path": "__HF_DATA_ROOT__/data/nq/test.jsonl",
        "default_target_size": 3000,
        "cap_per_signature": 12,
        "protect_signature_freq_leq": 3,
    },
    "triviaqa": {
        "train_path": "__HF_DATA_ROOT__/data/triviaqa/train.jsonl",
        "eval_path": "__HF_DATA_ROOT__/data/triviaqa/test.jsonl",
        "default_target_size": 3000,
        "cap_per_signature": 12,
        "protect_signature_freq_leq": 3,
    },
}


def apply_data_roots(hf_data_root: str) -> None:
    hf_data_root = hf_data_root.rstrip("/\\")
    for spec in DATASET_SPECS.values():
        spec["train_path"] = spec["train_path"].replace("__HF_DATA_ROOT__", hf_data_root)
        spec["eval_path"] = spec["eval_path"].replace("__HF_DATA_ROOT__", hf_data_root)


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

TEMPORAL_CUES = [
    "when",
    "what year",
    "what date",
    "what month",
    "what day",
    "born",
    "died",
    "released",
    "founded",
    "first aired",
    "started",
]

NUMERIC_CUES = [
    "how many",
    "how much",
    "how old",
    "how tall",
    "how long",
    "how far",
    "how big",
    "number of",
    "population",
    "area of",
    "height of",
    "length of",
]

LOCATION_CUES = [
    "where",
    "located",
    "location",
    "country",
    "city",
    "state",
    "capital",
    "place of birth",
    "birthplace",
    "place of death",
]

PERSON_CUES = [
    "who",
    "president",
    "director",
    "author",
    "writer",
    "actor",
    "actress",
    "singer",
    "founder",
    "inventor",
]

TITLE_CUES = [
    "film",
    "movie",
    "album",
    "song",
    "book",
    "novel",
    "show",
    "series",
    "episode",
]

ORG_CUES = [
    "company",
    "organization",
    "institution",
    "network",
    "university",
    "team",
    "club",
]

ALIAS_CUES = [
    "also known as",
    "nickname",
    "real name",
    "full name",
    "stage name",
    "pen name",
]

CAPITALIZED_SPAN_RE = re.compile(r"(?:[A-Z][\w'&.-]*)(?:\s+(?:[A-Z][\w'&.-]*))*")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
DIGIT_RE = re.compile(r"\d")
SPACE_RE = re.compile(r"\s+")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_dataset_names(values: Sequence[str]) -> List[str]:
    names: List[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        if name not in DATASET_SPECS:
            raise ValueError(f"Unsupported dataset: {name}")
        names.append(name)
    if not names:
        raise ValueError("At least one dataset is required.")
    return names


def normalize_question(question: str) -> str:
    return SPACE_RE.sub(" ", question).strip().lower()


def load_eval_ids(path: Path) -> set[str]:
    eval_ids: set[str] = set()
    if not path.exists():
        return eval_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            eval_ids.add(str(record.get("id", "")))
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
    if length <= 6:
        return "short"
    if length <= 10:
        return "medium"
    if length <= 16:
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
    if len(answer.split()) >= 4:
        return "long_span"
    return "entity_or_short_span"


def derive_flags(question: str, answer: str) -> List[str]:
    lowered = question.lower()
    flags: set[str] = {"direct_lookup"}
    if any(cue in lowered for cue in TEMPORAL_CUES):
        flags.add("temporal")
    if any(cue in lowered for cue in NUMERIC_CUES) or DIGIT_RE.search(answer):
        flags.add("numeric")
    if any(cue in lowered for cue in LOCATION_CUES):
        flags.add("location")
    if any(cue in lowered for cue in PERSON_CUES):
        flags.add("person")
    if any(cue in lowered for cue in TITLE_CUES):
        flags.add("title_work")
    if any(cue in lowered for cue in ORG_CUES):
        flags.add("organization")
    if any(cue in lowered for cue in ALIAS_CUES):
        flags.add("alias")
    if lowered.startswith(("is ", "are ", "was ", "were ", "did ", "do ", "does ")):
        flags.add("yes_no")
    if "\"" in question or "'" in question:
        flags.add("quoted_span")
    if len(question.split()) >= 14:
        flags.add("long_question")
    return sorted(flags)


def build_signature(
    dataset: str,
    wh_word: str,
    entity_bin: str,
    token_bin: str,
    answer_form_hint: str,
    flags: List[str],
) -> str:
    signature = {
        "dataset": dataset,
        "wh_word": wh_word,
        "entity_bin": entity_bin,
        "token_bin": token_bin,
        "answer_form_hint": answer_form_hint,
        "flags": flags[:6],
    }
    return json.dumps(signature, ensure_ascii=False, sort_keys=True)


def light_record(dataset: str, row: Dict[str, Any], eval_ids: set[str]) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answers = row.get("golden_answers") or row.get("answers") or row.get("answer") or []
    if isinstance(answers, list):
        primary_answer = str(answers[0]).strip() if answers else ""
    else:
        primary_answer = str(answers).strip()
    wh_word = first_wh_word(question)
    entity_count = count_entity_spans(question)
    entity_bin = entity_bucket(entity_count)
    token_bin = token_bucket(question)
    answer_form_hint = answer_form(primary_answer)
    flags = derive_flags(question, primary_answer)
    return {
        "id": str(row.get("id", "")),
        "dataset": dataset,
        "question": question,
        "primary_answer": primary_answer,
        "answer_form_hint": answer_form_hint,
        "wh_word": wh_word,
        "entity_count": entity_count,
        "entity_bin": entity_bin,
        "token_bin": token_bin,
        "flags": flags,
        "signature": build_signature(dataset, wh_word, entity_bin, token_bin, answer_form_hint, flags),
        "is_eval_overlap": str(row.get("id", "")) in eval_ids,
    }


def build_profiles(dataset: str, root_dir: Path, overwrite_existing: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    profile_dir = root_dir / "profiles"
    report_dir = root_dir / "reports"
    ensure_dir(profile_dir)
    ensure_dir(report_dir)
    light_path = profile_dir / f"{dataset}_train_light.jsonl"
    report_path = report_dir / f"{dataset}_profile_report.json"
    if light_path.exists() and report_path.exists() and not overwrite_existing:
        return load_jsonl(light_path), json.loads(report_path.read_text(encoding="utf-8"))

    spec = DATASET_SPECS[dataset]
    train_rows = load_jsonl(Path(spec["train_path"]))
    eval_ids = load_eval_ids(Path(spec["eval_path"]))
    examples = [light_record(dataset, row, eval_ids) for row in train_rows]
    answer_counter = Counter(example["answer_form_hint"] for example in examples)
    wh_counter = Counter(example["wh_word"] for example in examples)
    entity_counter = Counter(example["entity_bin"] for example in examples)
    token_counter = Counter(example["token_bin"] for example in examples)
    flag_counter = Counter(flag for example in examples for flag in example["flags"])
    report = {
        "dataset": dataset,
        "num_examples": len(examples),
        "num_signatures": len({example["signature"] for example in examples}),
        "answer_form_distribution": dict(answer_counter.most_common()),
        "wh_distribution": dict(wh_counter.most_common()),
        "entity_distribution": dict(entity_counter.most_common()),
        "token_distribution": dict(token_counter.most_common()),
        "flag_distribution": dict(flag_counter.most_common()),
        "default_target_size": int(spec["default_target_size"]),
    }
    write_jsonl(light_path, examples)
    write_json(report_path, report)
    return examples, report


def build_labels(row: Dict[str, Any]) -> List[str]:
    labels = [
        f"wh:{row.get('wh_word', 'unknown')}",
        f"answer:{row.get('answer_form_hint', 'unknown')}",
        f"entity:{row.get('entity_bin', 'unknown')}",
        f"token:{row.get('token_bin', 'unknown')}",
    ]
    for flag in row.get("flags", []):
        labels.append(f"flag:{flag}")
    return labels


def compute_rare_label_threshold(total: int) -> int:
    return max(12, math.ceil(total * 0.002))


def choose_target_size(dataset: str, requested: str) -> int:
    if requested != "auto":
        return int(requested)
    return int(DATASET_SPECS[dataset]["default_target_size"])


def score_row(row: Dict[str, Any], label_counts: Dict[str, int]) -> Tuple[float, int, str]:
    rarity_score = sum(1.0 / label_counts[label] for label in build_labels(row) if label_counts[label] > 0)
    question_len = len(str(row.get("question", "")).split())
    return (-rarity_score, -question_len, str(row.get("id", "")))


def sample_dataset(dataset: str, examples: List[Dict[str, Any]], target_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    spec = DATASET_SPECS[dataset]
    signature_counts = Counter(example["signature"] for example in examples)
    label_counts = Counter(label for example in examples for label in build_labels(example))
    rare_label_threshold = compute_rare_label_threshold(len(examples))

    protected_ids: set[str] = set()
    for example in examples:
        if signature_counts[example["signature"]] <= int(spec["protect_signature_freq_leq"]):
            protected_ids.add(example["id"])
            continue
        if any(label_counts[label] <= rare_label_threshold for label in build_labels(example)):
            protected_ids.add(example["id"])

    by_signature: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_signature[example["signature"]].append(example)

    selected_ids = set(protected_ids)
    cap_per_signature = int(spec["cap_per_signature"])
    for signature, rows in by_signature.items():
        protected_rows = [row for row in rows if row["id"] in protected_ids]
        unprotected_rows = [row for row in rows if row["id"] not in protected_ids]
        slots = max(cap_per_signature, len(protected_rows))
        if len(protected_rows) < slots and unprotected_rows:
            ranked = sorted(unprotected_rows, key=lambda row: score_row(row, label_counts))
            for row in ranked[: slots - len(protected_rows)]:
                selected_ids.add(row["id"])

    initial = [row for row in examples if row["id"] in selected_ids]
    if len(initial) < target_size:
        remaining = [row for row in examples if row["id"] not in selected_ids]
        remaining = sorted(remaining, key=lambda row: score_row(row, label_counts))
        for row in remaining:
            if len(initial) >= target_size:
                break
            selected_ids.add(row["id"])
            initial.append(row)

    deduped: List[Dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in initial:
        normalized = normalize_question(row["question"])
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        deduped.append(row)

    if len(deduped) > target_size:
        deduped = sorted(deduped, key=lambda row: score_row(row, label_counts))[:target_size]

    kept_label_counts = Counter(label for row in deduped for label in build_labels(row))
    report = {
        "dataset": dataset,
        "original_size": len(examples),
        "target_size": target_size,
        "selected_size": len(deduped),
        "selected_ratio": len(deduped) / len(examples) if examples else 0.0,
        "unique_signatures_original": len(signature_counts),
        "unique_signatures_selected": len({row["signature"] for row in deduped}),
        "cap_per_signature": cap_per_signature,
        "protect_signature_freq_leq": int(spec["protect_signature_freq_leq"]),
        "rare_label_threshold": rare_label_threshold,
        "protected_examples": len(protected_ids),
        "answer_form_selected": dict(Counter(row["answer_form_hint"] for row in deduped).most_common()),
        "wh_selected": dict(Counter(row["wh_word"] for row in deduped).most_common()),
        "entity_selected": dict(Counter(row["entity_bin"] for row in deduped).most_common()),
        "token_selected": dict(Counter(row["token_bin"] for row in deduped).most_common()),
        "flag_selected": dict(Counter(flag for row in deduped for flag in row["flags"]).most_common()),
        "label_coverage_selected": dict(kept_label_counts.most_common(30)),
    }
    return sorted(deduped, key=lambda row: row["id"]), report


def materialize_full_records(dataset: str, selected_ids: set[str]) -> List[Dict[str, Any]]:
    train_rows = load_jsonl(Path(DATASET_SPECS[dataset]["train_path"]))
    selected = [row for row in train_rows if str(row.get("id", "")) in selected_ids]
    selected.sort(key=lambda row: str(row.get("id", "")))
    return selected


def save_selected_light(root_dir: Path, dataset: str, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(root_dir / "samples" / dataset / "train_sample_light.jsonl", rows)


def save_selected_full(root_dir: Path, dataset: str, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(root_dir / "samples" / dataset / "train_sample_full.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-oriented sampler for single-hop QA training subsets.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--target-size", default="auto")
    parser.add_argument("--hf-data-root", default=os.environ.get("HF_DATA", "__HF_DATA_ROOT__"))
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    apply_data_roots(args.hf_data_root)

    root_dir = args.root_dir
    ensure_dir(root_dir)
    ensure_dir(root_dir / "profiles")
    ensure_dir(root_dir / "reports")
    ensure_dir(root_dir / "samples")
    ensure_dir(root_dir / "logs")

    dataset_names = parse_dataset_names(args.datasets)
    summary: Dict[str, Any] = {"datasets": {}}

    for dataset in dataset_names:
        examples, profile_report = build_profiles(dataset, root_dir, overwrite_existing=args.overwrite_existing)
        chosen_target_size = choose_target_size(dataset, args.target_size)
        target_size = min(chosen_target_size, len(examples))
        selected_light, sampling_report = sample_dataset(dataset, examples, target_size=target_size)
        selected_full = materialize_full_records(dataset, {row["id"] for row in selected_light})
        save_selected_light(root_dir, dataset, selected_light)
        save_selected_full(root_dir, dataset, selected_full)
        write_json(root_dir / "reports" / f"{dataset}_sampling_report.json", {**profile_report, **sampling_report})
        summary["datasets"][dataset] = {
            "default_target_size": profile_report["default_target_size"],
            "selected_size": len(selected_light),
            "num_examples": profile_report["num_examples"],
            "num_signatures": profile_report["num_signatures"],
            "target_size": chosen_target_size,
        }

    write_json(root_dir / "reports" / "sampling_summary_singlehop.json", summary)


if __name__ == "__main__":
    main()
