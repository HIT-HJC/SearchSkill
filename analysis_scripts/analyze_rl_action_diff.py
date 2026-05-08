#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_PATHS = {
    "hotpotqa": {
        "sft": "sft_stage2_7b_base_eval/hotpotqa/merged/hotpotqa_trace.jsonl",
        "rl": "rl_stage2_7b_base_eval/hotpotqa/merged/hotpotqa_trace.jsonl",
    },
    "2wiki": {
        "sft": "sft_stage2_7b_base_eval/2wiki/merged/2wiki_trace.jsonl",
        "rl": "rl_stage2_7b_base_eval/2wiki/merged/2wiki_trace.jsonl",
    },
    "musique": {
        "sft": "sft_stage2_7b_base_eval/musique/merged/musique_trace.jsonl",
        "rl": "rl_stage2_7b_base_eval/musique/merged/musique_trace.jsonl",
    },
    "bamboogle": {
        "sft": "sft_stage2_7b_base_eval/bamboogle/merged/bamboogle_trace.jsonl",
        "rl": "rl_stage2_7b_base_eval/bamboogle/merged/bamboogle_trace.jsonl",
    },
}

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str | None) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def contains_gold(text: str | None, golds: list[str] | None) -> bool:
    normalized_text = normalize(text)
    for gold in golds or []:
        normalized_gold = normalize(gold)
        if normalized_gold and normalized_gold in normalized_text:
            return True
    return False


def get_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    steps = row.get("steps") or []
    return steps if isinstance(steps, list) else []


def get_queries(row: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for step in get_steps(row):
        query = step.get("query")
        if query:
            queries.append(str(query))
    return queries


def get_retrieved_text(row: dict[str, Any]) -> str:
    retrieved: list[str] = []
    for step in get_steps(row):
        text = step.get("retrieved")
        if text:
            retrieved.append(str(text))
    return "\n".join(retrieved)


def get_skills(row: dict[str, Any]) -> list[str]:
    skills: list[str] = []
    for step in get_steps(row):
        skill_selection = step.get("skill_selection") or {}
        selected = step.get("selected_skills") or skill_selection.get("selected_skills") or []
        skills.extend(str(skill) for skill in selected)
    return skills


def duplicate_query_rate(row: dict[str, Any]) -> float:
    queries = [normalize(query) for query in get_queries(row)]
    if not queries:
        return 0.0
    return float(len(queries) != len(set(queries)))


def summarize_row(row: dict[str, Any], bridge_pattern: re.Pattern[str]) -> dict[str, Any]:
    skills = get_skills(row)
    queries = get_queries(row)
    return {
        "em": int(bool(row.get("em"))),
        "gold_in_evidence": int(contains_gold(get_retrieved_text(row), row.get("gold") or [])),
        "pred_contains_gold": int(contains_gold(row.get("prediction"), row.get("gold") or [])),
        "searches": int(row.get("searches_used") if row.get("searches_used") is not None else len(queries)),
        "duplicate_query": duplicate_query_rate(row),
        "bridge_actions": sum(1 for skill in skills if bridge_pattern.search(skill)),
        "relation_chain_actions": sum(1 for skill in skills if skill == "relation-chain-decomposition"),
        "skills": skills,
        "queries": queries,
        "question": row.get("question"),
        "gold": row.get("gold"),
        "prediction": row.get("prediction"),
    }


def group_name(sft: dict[str, Any], rl: dict[str, Any]) -> str:
    if sft["em"] and rl["em"]:
        return "both_correct"
    if sft["em"] and not rl["em"]:
        return "sft_only"
    if rl["em"] and not sft["em"]:
        return "rl_only"
    return "both_wrong"


def pct(value: float) -> float:
    return round(100.0 * value, 2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(str(row[column]) for column in columns) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired supervised_finetuning/RL action and evidence changes.")
    parser.add_argument("--eval-root", type=Path, required=True, help="Root directory containing eval outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-pattern", default=r"bridge|anchor", help="Regex defining bridge/action-planning skills.")
    parser.add_argument("--max-cases-per-group", type=int, default=50)
    args = parser.parse_args()

    bridge_pattern = re.compile(args.bridge_pattern)
    pairs: list[dict[str, Any]] = []
    for dataset, rel_paths in DATASET_PATHS.items():
        sft_rows = {normalize(row.get("question")): row for row in load_jsonl(args.eval_root / rel_paths["sft"])}
        rl_rows = {normalize(row.get("question")): row for row in load_jsonl(args.eval_root / rel_paths["rl"])}
        common_questions = sorted(set(sft_rows) & set(rl_rows))
        if not common_questions:
            raise RuntimeError(f"No paired questions found for {dataset}")
        for key in common_questions:
            sft_metrics = summarize_row(sft_rows[key], bridge_pattern)
            rl_metrics = summarize_row(rl_rows[key], bridge_pattern)
            pairs.append(
                {
                    "dataset": dataset,
                    "question_key": key,
                    "group": group_name(sft_metrics, rl_metrics),
                    "sft": sft_metrics,
                    "rl": rl_metrics,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    flip_rows: list[dict[str, Any]] = []
    for dataset in list(DATASET_PATHS) + ["overall"]:
        subset = pairs if dataset == "overall" else [pair for pair in pairs if pair["dataset"] == dataset]
        counts = Counter(pair["group"] for pair in subset)
        n = len(subset)
        flip_rows.append(
            {
                "dataset": dataset,
                "n": n,
                "sft_em": pct(safe_mean([pair["sft"]["em"] for pair in subset])),
                "rl_em": pct(safe_mean([pair["rl"]["em"] for pair in subset])),
                "both_correct": counts["both_correct"],
                "sft_only": counts["sft_only"],
                "rl_only": counts["rl_only"],
                "both_wrong": counts["both_wrong"],
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group in ["both_correct", "sft_only", "rl_only", "both_wrong"]:
        subset = [pair for pair in pairs if pair["group"] == group]
        group_rows.append(
            {
                "group": group,
                "n": len(subset),
                "sft_gold_ev": pct(safe_mean([pair["sft"]["gold_in_evidence"] for pair in subset])),
                "rl_gold_ev": pct(safe_mean([pair["rl"]["gold_in_evidence"] for pair in subset])),
                "delta_gold_ev": pct(
                    safe_mean([pair["rl"]["gold_in_evidence"] - pair["sft"]["gold_in_evidence"] for pair in subset])
                ),
                "sft_bridge": round(safe_mean([pair["sft"]["bridge_actions"] for pair in subset]), 3),
                "rl_bridge": round(safe_mean([pair["rl"]["bridge_actions"] for pair in subset]), 3),
                "sft_relation": round(safe_mean([pair["sft"]["relation_chain_actions"] for pair in subset]), 3),
                "rl_relation": round(safe_mean([pair["rl"]["relation_chain_actions"] for pair in subset]), 3),
                "sft_searches": round(safe_mean([pair["sft"]["searches"] for pair in subset]), 3),
                "rl_searches": round(safe_mean([pair["rl"]["searches"] for pair in subset]), 3),
                "sft_dup_query": pct(safe_mean([pair["sft"]["duplicate_query"] for pair in subset])),
                "rl_dup_query": pct(safe_mean([pair["rl"]["duplicate_query"] for pair in subset])),
            }
        )

    skill_counts: dict[str, Counter[str]] = {"sft": Counter(), "rl": Counter()}
    for pair in pairs:
        skill_counts["sft"].update(pair["sft"]["skills"])
        skill_counts["rl"].update(pair["rl"]["skills"])
    all_skills = sorted(set(skill_counts["sft"]) | set(skill_counts["rl"]))
    action_rows = [
        {
            "skill": skill,
            "sft_count": skill_counts["sft"][skill],
            "rl_count": skill_counts["rl"][skill],
            "delta": skill_counts["rl"][skill] - skill_counts["sft"][skill],
        }
        for skill in all_skills
    ]
    action_rows.sort(key=lambda row: abs(int(row["delta"])), reverse=True)

    cause_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    cases_path = args.output_dir / "flip_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            group = pair["group"]
            if group == "rl_only":
                if pair["rl"]["gold_in_evidence"] > pair["sft"]["gold_in_evidence"]:
                    cause = "rl_gain_gold_evidence_appears"
                elif pair["rl"]["gold_in_evidence"] and pair["sft"]["gold_in_evidence"]:
                    cause = "rl_gain_both_have_gold_evidence"
                else:
                    cause = "rl_gain_no_literal_gold_proxy"
            elif group == "sft_only":
                if pair["sft"]["gold_in_evidence"] > pair["rl"]["gold_in_evidence"]:
                    cause = "rl_loss_gold_evidence_disappears"
                elif pair["sft"]["gold_in_evidence"] and pair["rl"]["gold_in_evidence"]:
                    cause = "rl_loss_both_have_gold_evidence"
                elif pair["rl"]["pred_contains_gold"]:
                    cause = "rl_loss_prediction_contains_gold"
                else:
                    cause = "rl_loss_no_literal_gold_proxy"
            else:
                continue
            cause_counts[cause] += 1
            if case_counts[group] >= args.max_cases_per_group:
                continue
            case_counts[group] += 1
            handle.write(
                json.dumps(
                    {
                        "dataset": pair["dataset"],
                        "group": group,
                        "cause_proxy": cause,
                        "question": pair["sft"]["question"],
                        "gold": pair["sft"]["gold"],
                        "sft_prediction": pair["sft"]["prediction"],
                        "rl_prediction": pair["rl"]["prediction"],
                        "sft_gold_in_evidence": pair["sft"]["gold_in_evidence"],
                        "rl_gold_in_evidence": pair["rl"]["gold_in_evidence"],
                        "sft_queries": pair["sft"]["queries"],
                        "rl_queries": pair["rl"]["queries"],
                        "sft_skills": pair["sft"]["skills"],
                        "rl_skills": pair["rl"]["skills"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    cause_rows = [{"cause_proxy": cause, "count": count} for cause, count in cause_counts.most_common()]

    write_csv(
        args.output_dir / "flip_summary.csv",
        flip_rows,
        ["dataset", "n", "sft_em", "rl_em", "both_correct", "sft_only", "rl_only", "both_wrong"],
    )
    write_csv(
        args.output_dir / "group_action_evidence.csv",
        group_rows,
        [
            "group",
            "n",
            "sft_gold_ev",
            "rl_gold_ev",
            "delta_gold_ev",
            "sft_bridge",
            "rl_bridge",
            "sft_relation",
            "rl_relation",
            "sft_searches",
            "rl_searches",
            "sft_dup_query",
            "rl_dup_query",
        ],
    )
    write_csv(args.output_dir / "action_distribution.csv", action_rows, ["skill", "sft_count", "rl_count", "delta"])
    write_csv(args.output_dir / "flip_cause_proxy.csv", cause_rows, ["cause_proxy", "count"])

    write_latex_table(
        args.output_dir / "flip_summary.tex",
        flip_rows,
        ["dataset", "n", "sft_em", "rl_em", "both_correct", "sft_only", "rl_only", "both_wrong"],
    )
    write_latex_table(
        args.output_dir / "group_action_evidence.tex",
        group_rows,
        ["group", "n", "sft_gold_ev", "rl_gold_ev", "delta_gold_ev", "sft_bridge", "rl_bridge", "sft_relation", "rl_relation"],
    )

    manifest = {
        "eval_root": str(args.eval_root),
        "output_dir": str(args.output_dir),
        "bridge_pattern": args.bridge_pattern,
        "n_pairs": len(pairs),
        "outputs": [
            "flip_summary.csv",
            "group_action_evidence.csv",
            "action_distribution.csv",
            "flip_cause_proxy.csv",
            "flip_cases.jsonl",
            "flip_summary.tex",
            "group_action_evidence.tex",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
