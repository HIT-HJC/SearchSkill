#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


SEARCH_R1_ROOT = Path("/path/to/Search-R1")
if str(SEARCH_R1_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_R1_ROOT))

from verl.utils.reward_score.searchskill import compute_score, trajectory_stats  # noqa: E402


def build_solution_from_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join(str(turn.get("generated") or "").strip() for turn in turns if turn.get("generated"))


def load_rollouts(trace_path: Path, limit_tasks: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as f:
        for task_idx, line in enumerate(f, start=1):
            if limit_tasks is not None and task_idx > limit_tasks:
                break
            task_trace = json.loads(line)
            task = task_trace.get("task") or {}
            for rollout_idx, rollout in enumerate(task_trace.get("rollouts") or []):
                score = rollout.get("score") or {}
                solution = build_solution_from_turns(rollout.get("turns") or [])
                final_stats = trajectory_stats(solution)
                final_reward = compute_score(
                    solution,
                    {"target": task.get("gold_answers") or []},
                    data_source=task.get("dataset"),
                )
                old_reward = float(rollout.get("reward", score.get("reward", 0.0)) or 0.0)
                rows.append(
                    {
                        "task_id": task.get("id"),
                        "dataset": task.get("dataset"),
                        "rollout_idx": rollout_idx,
                        "old_reward": old_reward,
                        "final_reward": float(final_reward),
                        "delta": float(final_reward) - old_reward,
                        "old_exact": bool(score.get("exact_match")),
                        "old_search_count": int(score.get("search_count") or 0),
                        "old_dup_count": int(score.get("duplicate_search_count") or score.get("duplicate_query_count") or 0),
                        "old_malformed": bool(score.get("malformed")),
                        "final_stable": bool(final_stats.get("stable_interface")),
                        "final_search_count": int(final_stats.get("search_count") or 0),
                        "final_dup_count": int(final_stats.get("duplicate_count") or 0),
                        "old_answer": score.get("final_answer"),
                        "final_answer": final_stats.get("answer"),
                    }
                )
    return rows


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "reinforcement_learning/previous_policy/"
            "runs/previous_policy_action_samples/sample_traces.jsonl"
        ),
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reinforcement_learning/reward_alignment_v9_2_vs_final.jsonl"),
    )
    args = parser.parse_args()

    rows = load_rollouts(args.trace, args.limit_tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mismatches = [row for row in rows if not close(row["old_reward"], row["final_reward"])]
    old_positive = [row for row in rows if row["old_reward"] > 0]
    final_positive = [row for row in rows if row["final_reward"] > 0]
    print(f"rows={len(rows)}")
    print(f"match={len(rows) - len(mismatches)} mismatch={len(mismatches)} mismatch_rate={len(mismatches)/max(1,len(rows)):.3f}")
    print(f"old_mean={mean(row['old_reward'] for row in rows):.4f} final_mean={mean(row['final_reward'] for row in rows):.4f}")
    print(f"old_positive={len(old_positive)} final_positive={len(final_positive)}")
    print(f"old_exact={sum(row['old_exact'] for row in rows)}")
    print(f"final_stable={sum(row['final_stable'] for row in rows)}")
    print("top_mismatches:")
    for row in sorted(mismatches, key=lambda item: abs(item["delta"]), reverse=True)[:20]:
        print(
            json.dumps(
                {
                    "task_id": row["task_id"],
                    "dataset": row["dataset"],
                    "rollout_idx": row["rollout_idx"],
                    "old_reward": row["old_reward"],
                    "final_reward": row["final_reward"],
                    "old_exact": row["old_exact"],
                    "old_search_count": row["old_search_count"],
                    "final_search_count": row["final_search_count"],
                    "old_answer": row["old_answer"],
                    "final_answer": row["final_answer"],
                },
                ensure_ascii=False,
            )
        )
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
