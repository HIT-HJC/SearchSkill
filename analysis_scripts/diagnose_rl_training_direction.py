#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
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


def normalize(text: str | None) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def load_by_question(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[normalize(row.get("question"))] = row
    return rows


def contains_gold(text: str | None, golds: list[str] | None) -> bool:
    normalized_text = normalize(text)
    return any(normalize(gold) and normalize(gold) in normalized_text for gold in golds or [])


def steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("steps") or []
    return value if isinstance(value, list) else []


def queries(row: dict[str, Any]) -> list[str]:
    return [str(step.get("query")) for step in steps(row) if step.get("query")]


def retrieved_text(row: dict[str, Any]) -> str:
    return "\n".join(str(step.get("retrieved") or "") for step in steps(row))


def skills(row: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    for step in steps(row):
        skill_selection = step.get("skill_selection") or {}
        selected.extend(str(skill) for skill in (step.get("selected_skills") or skill_selection.get("selected_skills") or []))
    return selected


def metrics(row: dict[str, Any]) -> dict[str, Any]:
    selected = skills(row)
    gold = row.get("gold") or []
    return {
        "em": int(bool(row.get("em"))),
        "gold_ev": int(contains_gold(retrieved_text(row), gold)),
        "pred_gold": int(contains_gold(row.get("prediction"), gold)),
        "bridge": sum(1 for skill in selected if re.search(r"bridge|anchor", skill)),
        "relation": sum(1 for skill in selected if skill == "relation-chain-decomposition"),
        "grounding": sum(1 for skill in selected if any(tag in skill for tag in ("ground", "evidence", "verbatim"))),
        "searches": int(row.get("searches_used") if row.get("searches_used") is not None else len(queries(row))),
        "question": row.get("question"),
        "gold": gold,
        "prediction": row.get("prediction"),
        "queries": queries(row),
        "skills": selected,
    }


def group_name(sft: dict[str, Any], rl: dict[str, Any]) -> str:
    if sft["em"] and rl["em"]:
        return "both_correct"
    if sft["em"] and not rl["em"]:
        return "sft_only"
    if rl["em"] and not sft["em"]:
        return "rl_only"
    return "both_wrong"


def change(value: int | float) -> str:
    return "up" if value > 0 else "down" if value < 0 else "same"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> float:
    return round(value * 100, 2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert supervised_finetuning/RL paired trace differences into training-direction diagnostics.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for dataset, (sft_rel, rl_rel) in DATASET_PATHS.items():
        sft_rows = load_by_question(args.eval_root / sft_rel)
        rl_rows = load_by_question(args.eval_root / rl_rel)
        for key in sorted(set(sft_rows) & set(rl_rows)):
            pairs.append((dataset, metrics(sft_rows[key]), metrics(rl_rows[key])))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    action_change_rows: list[dict[str, Any]] = []
    for action in ("bridge", "relation", "grounding", "searches"):
        for direction in ("up", "same", "down"):
            subset = [(dataset, sft, rl) for dataset, sft, rl in pairs if change(rl[action] - sft[action]) == direction]
            if not subset:
                continue
            action_change_rows.append(
                {
                    "action": action,
                    "direction": direction,
                    "n": len(subset),
                    "em_gain": sum(1 for _, sft, rl in subset if rl["em"] > sft["em"]),
                    "em_loss": sum(1 for _, sft, rl in subset if rl["em"] < sft["em"]),
                    "net_em": sum(rl["em"] - sft["em"] for _, sft, rl in subset),
                    "evidence_gain": sum(1 for _, sft, rl in subset if rl["gold_ev"] > sft["gold_ev"]),
                    "evidence_loss": sum(1 for _, sft, rl in subset if rl["gold_ev"] < sft["gold_ev"]),
                    "net_evidence": sum(rl["gold_ev"] - sft["gold_ev"] for _, sft, rl in subset),
                    "avg_sft_action": round(mean([sft[action] for _, sft, _ in subset]), 3),
                    "avg_rl_action": round(mean([rl[action] for _, _, rl in subset]), 3),
                }
            )

    bucket_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset, sft, rl in pairs:
        group = group_name(sft, rl)
        if group == "rl_only" and rl["gold_ev"] > sft["gold_ev"]:
            bucket = "benefit_retrieval: RL recovers missing gold evidence"
        elif group == "rl_only" and rl["gold_ev"] and sft["gold_ev"]:
            bucket = "benefit_answering: same gold evidence, RL answers better"
        elif group == "sft_only" and sft["gold_ev"] > rl["gold_ev"]:
            bucket = "risk_retrieval: RL loses previously found gold evidence"
        elif group == "sft_only" and sft["gold_ev"] and rl["gold_ev"]:
            bucket = "risk_answering: gold evidence present, RL answers worse"
        elif group == "both_wrong" and rl["gold_ev"] and not sft["gold_ev"]:
            bucket = "opportunity_retrieval: RL finds new gold evidence but answer still wrong"
        elif group == "both_wrong" and rl["gold_ev"]:
            bucket = "opportunity_answering: RL has gold evidence but still answers wrong"
        elif group == "both_wrong" and sft["gold_ev"] and not rl["gold_ev"]:
            bucket = "risk_pool: RL drops gold evidence without EM recovery"
        else:
            continue
        bucket_counts[bucket] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append(
                {
                    "dataset": dataset,
                    "group": group,
                    "question": sft["question"],
                    "gold": sft["gold"],
                    "sft_prediction": sft["prediction"],
                    "rl_prediction": rl["prediction"],
                    "sft_gold_ev": sft["gold_ev"],
                    "rl_gold_ev": rl["gold_ev"],
                    "sft_bridge": sft["bridge"],
                    "rl_bridge": rl["bridge"],
                    "sft_relation": sft["relation"],
                    "rl_relation": rl["relation"],
                    "sft_queries": sft["queries"],
                    "rl_queries": rl["queries"],
                }
            )

    bucket_rows = [{"bucket": bucket, "count": count} for bucket, count in bucket_counts.most_common()]

    opportunity_rows: list[dict[str, Any]] = []
    for dataset in DATASET_PATHS:
        subset = [(sft, rl) for ds, sft, rl in pairs if ds == dataset]
        both_wrong = [(sft, rl) for sft, rl in subset if not sft["em"] and not rl["em"]]
        opportunity_rows.append(
            {
                "dataset": dataset,
                "n": len(subset),
                "both_wrong": len(both_wrong),
                "rl_gold_ev_but_wrong": sum(1 for _, rl in both_wrong if rl["gold_ev"]),
                "rl_new_gold_ev_but_wrong": sum(1 for sft, rl in both_wrong if rl["gold_ev"] and not sft["gold_ev"]),
                "answer_training_pool_pct": pct(mean([rl["gold_ev"] for _, rl in both_wrong])),
                "new_evidence_pool_pct": pct(mean([1 if rl["gold_ev"] and not sft["gold_ev"] else 0 for sft, rl in both_wrong])),
            }
        )

    write_csv(args.output_dir / "benefit_risk_by_action_change.csv", action_change_rows)
    write_csv(args.output_dir / "training_direction_buckets.csv", bucket_rows)
    write_csv(args.output_dir / "answer_stage_opportunity.csv", opportunity_rows)
    (args.output_dir / "training_direction_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"n_pairs": len(pairs), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
