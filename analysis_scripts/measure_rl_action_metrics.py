#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DATASET_PATHS = {
    "hotpotqa": (
        "hotpot_toolstar_stage2_7b_base_gpu021_2gpu/merged/hotpot_toolstar_stage2_7b_base_gpu021_2gpu_trace.jsonl",
        "rl_v9_base_step0060_eval7_gpu028_4gpu_rerun_0501/hotpotqa/merged/hotpotqa_trace.jsonl",
    ),
    "2wiki": (
        "2wiki_toolstar_stage2_7b_base_gpu021_2gpu/merged/2wiki_toolstar_stage2_7b_base_gpu021_2gpu_trace.jsonl",
        "rl_v9_base_step0060_eval7_gpu028_4gpu_rerun_0501/2wiki/merged/2wiki_trace.jsonl",
    ),
    "musique": (
        "musique_toolstar_stage2_7b_base_gpu021_2gpu/merged/musique_toolstar_stage2_7b_base_gpu021_2gpu_trace.jsonl",
        "rl_v9_base_step0060_eval7_gpu028_4gpu_rerun_0501/musique/merged/musique_trace.jsonl",
    ),
    "bamboogle": (
        "bamboogle_toolstar_stage2_7b_base_gpu021_2gpu/merged/bamboogle_toolstar_stage2_7b_base_gpu021_2gpu_trace.jsonl",
        "rl_v9_base_step0060_eval7_gpu028_4gpu_rerun_0501/bamboogle/merged/bamboogle_trace.jsonl",
    ),
}

WORD_RE = re.compile(r"[A-Za-z0-9]+")
TITLE_RE = re.compile(r'Title:\s*"([^"]+)"')

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def tokens(text: str | None, *, keep_stopwords: bool = False) -> set[str]:
    values = {token.lower() for token in WORD_RE.findall(text or "")}
    if keep_stopwords:
        return values
    return {token for token in values if token not in STOPWORDS and len(token) > 1}


def normalize(text: str | None) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def load_by_question(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[normalize(row.get("question"))] = row
    return rows


def steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("steps") or []
    return value if isinstance(value, list) else []


def query_steps(row: dict[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for step in steps(row):
        query = step.get("query")
        if not query:
            continue
        output.append(
            {
                "query": str(query),
                "retrieved": str(step.get("retrieved") or ""),
            }
        )
    return output


def contains_gold(text: str | None, golds: list[str] | None) -> bool:
    normalized_text = normalize(text)
    return any(normalize(gold) and normalize(gold) in normalized_text for gold in golds or [])


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metric_names() -> list[str]:
    return [
        "searches",
        "query_question_overlap",
        "query_novelty",
        "duplicate_query_rate",
        "title_conditioned_rate",
        "evidence_conditioned_rate",
        "gold_found_rate",
        "first_gold_step",
        "gold_by_step2",
        "post_gold_extra_searches",
        "answer_grounded_rate",
        "bridge_actions",
        "relation_chain_actions",
    ]


def action_metrics(row: dict[str, Any]) -> dict[str, float]:
    qsteps = query_steps(row)
    question_tokens = tokens(row.get("question"))
    golds = row.get("gold") or []
    all_retrieved = "\n".join(step["retrieved"] for step in qsteps)

    query_token_sets = [tokens(step["query"]) for step in qsteps]
    query_norms = [normalize(step["query"]) for step in qsteps]
    query_question_overlap = mean([jaccard(query_tokens, question_tokens) for query_tokens in query_token_sets])

    novelty_scores: list[float] = []
    for prev_tokens, cur_tokens in zip(query_token_sets, query_token_sets[1:]):
        novelty_scores.append(1.0 - jaccard(prev_tokens, cur_tokens))

    title_conditioned = 0
    evidence_conditioned = 0
    transition_count = max(0, len(qsteps) - 1)
    for idx in range(1, len(qsteps)):
        prev_titles = " ".join(TITLE_RE.findall(qsteps[idx - 1]["retrieved"]))
        prev_title_tokens = tokens(prev_titles) - question_tokens
        prev_evidence_tokens = tokens(qsteps[idx - 1]["retrieved"]) - question_tokens
        cur_query_tokens = query_token_sets[idx]
        if cur_query_tokens & prev_title_tokens:
            title_conditioned += 1
        if cur_query_tokens & prev_evidence_tokens:
            evidence_conditioned += 1

    first_gold_step = 0
    for idx, step in enumerate(qsteps, start=1):
        if contains_gold(step["retrieved"], golds):
            first_gold_step = idx
            break

    selected_skills: list[str] = []
    for step in steps(row):
        skill_selection = step.get("skill_selection") or {}
        selected_skills.extend(str(skill) for skill in (step.get("selected_skills") or skill_selection.get("selected_skills") or []))

    return {
        "searches": float(len(qsteps)),
        "query_question_overlap": query_question_overlap,
        "query_novelty": mean(novelty_scores),
        "duplicate_query_rate": float(len(query_norms) != len(set(query_norms))) if query_norms else 0.0,
        "title_conditioned_rate": title_conditioned / transition_count if transition_count else 0.0,
        "evidence_conditioned_rate": evidence_conditioned / transition_count if transition_count else 0.0,
        "gold_found_rate": float(first_gold_step > 0),
        "first_gold_step": float(first_gold_step),
        "gold_by_step2": float(0 < first_gold_step <= 2),
        "post_gold_extra_searches": float(max(0, len(qsteps) - first_gold_step)) if first_gold_step else 0.0,
        "answer_grounded_rate": float(contains_gold(row.get("prediction"), golds) or contains_gold(all_retrieved, [row.get("prediction") or ""])),
        "bridge_actions": float(sum(1 for skill in selected_skills if re.search(r"bridge|anchor", skill))),
        "relation_chain_actions": float(sum(1 for skill in selected_skills if skill == "relation-chain-decomposition")),
    }


def group_name(sft_em: int, rl_em: int) -> str:
    if sft_em and rl_em:
        return "both_correct"
    if sft_em and not rl_em:
        return "sft_only"
    if rl_em and not sft_em:
        return "rl_only"
    return "both_wrong"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_subset(label: str, subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metric_names():
        sft_values = [row[f"sft_{metric}"] for row in subset]
        rl_values = [row[f"rl_{metric}"] for row in subset]
        deltas = [row[f"rl_{metric}"] - row[f"sft_{metric}"] for row in subset]
        rows.append(
            {
                "subset": label,
                "metric": metric,
                "n": len(subset),
                "sft_mean": round(mean(sft_values), 4),
                "rl_mean": round(mean(rl_values), 4),
                "delta": round(mean(deltas), 4),
                "rl_higher_pct": round(mean([float(delta > 0) for delta in deltas]) * 100, 2),
                "rl_lower_pct": round(mean([float(delta < 0) for delta in deltas]) * 100, 2),
                "same_pct": round(mean([float(delta == 0) for delta in deltas]) * 100, 2),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure numerical action-trajectory changes between SFT and RL.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paired_rows: list[dict[str, Any]] = []
    for dataset, (sft_rel, rl_rel) in DATASET_PATHS.items():
        sft_rows = load_by_question(args.eval_root / sft_rel)
        rl_rows = load_by_question(args.eval_root / rl_rel)
        for key in sorted(set(sft_rows) & set(rl_rows)):
            sft_row = sft_rows[key]
            rl_row = rl_rows[key]
            sft_metrics = action_metrics(sft_row)
            rl_metrics = action_metrics(rl_row)
            row: dict[str, Any] = {
                "dataset": dataset,
                "group": group_name(int(bool(sft_row.get("em"))), int(bool(rl_row.get("em")))),
                "question": sft_row.get("question"),
                "sft_em": int(bool(sft_row.get("em"))),
                "rl_em": int(bool(rl_row.get("em"))),
            }
            for metric in metric_names():
                row[f"sft_{metric}"] = sft_metrics[metric]
                row[f"rl_{metric}"] = rl_metrics[metric]
                row[f"delta_{metric}"] = rl_metrics[metric] - sft_metrics[metric]
            paired_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_trajectory_pairwise.csv", paired_rows)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize_subset("overall", paired_rows))
    for dataset in DATASET_PATHS:
        summary_rows.extend(summarize_subset(dataset, [row for row in paired_rows if row["dataset"] == dataset]))
    for group in ("both_correct", "sft_only", "rl_only", "both_wrong"):
        summary_rows.extend(summarize_subset(group, [row for row in paired_rows if row["group"] == group]))
    write_csv(args.output_dir / "action_trajectory_summary.csv", summary_rows)

    key_metrics = [
        "title_conditioned_rate",
        "evidence_conditioned_rate",
        "query_question_overlap",
        "query_novelty",
        "gold_found_rate",
        "first_gold_step",
        "gold_by_step2",
        "post_gold_extra_searches",
        "answer_grounded_rate",
    ]
    key_rows = [row for row in summary_rows if row["subset"] in ("overall", "rl_only", "sft_only", "both_wrong") and row["metric"] in key_metrics]
    write_csv(args.output_dir / "action_trajectory_key_metrics.csv", key_rows)

    print(
        json.dumps(
            {
                "n_pairs": len(paired_rows),
                "output_dir": str(args.output_dir),
                "outputs": [
                    "action_trajectory_pairwise.csv",
                    "action_trajectory_summary.csv",
                    "action_trajectory_key_metrics.csv",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
