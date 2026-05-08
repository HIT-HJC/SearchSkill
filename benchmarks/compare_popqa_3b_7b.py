#!/usr/bin/env python3
import json
from pathlib import Path


p3 = Path("outputs/zerosearch_qwen25_3b_base_localretriever_standard_20260430/singlehop_benchmark1000/popqa/popqa_zerosearch_qwen25_3b_base_t00_categoryhard70_gpu025_g0_job1319639.trace.jsonl")
p7 = Path("outputs/zerosearch_qwen25_7b_base_localretriever_standard_20260430/singlehop_categoryhard70/popqa/popqa_zerosearch_qwen25_7b_base_t00_categoryhard70_gpu025_g0_job1319639.trace.jsonl")


def load(path: Path):
    return [json.loads(x) for x in path.open("r", encoding="utf-8") if x.strip()]


rows3 = load(p3)
rows7 = load(p7)
assert len(rows3) == len(rows7), (len(rows3), len(rows7))

b3_only = []
b7_only = []
both = 0
none = 0

for a, b in zip(rows3, rows7):
    e3 = int(a.get("em", 0) == 1)
    e7 = int(b.get("em", 0) == 1)
    if e3 and not e7:
        b3_only.append(
            {
                "q": a.get("question", ""),
                "gold": a.get("gold"),
                "p3": a.get("prediction"),
                "p7": b.get("prediction"),
            }
        )
    elif e7 and not e3:
        b7_only.append(
            {
                "q": a.get("question", ""),
                "gold": a.get("gold"),
                "p3": a.get("prediction"),
                "p7": b.get("prediction"),
            }
        )
    elif e3 and e7:
        both += 1
    else:
        none += 1

print(json.dumps(
    {
        "n": len(rows3),
        "3b_only": len(b3_only),
        "7b_only": len(b7_only),
        "both": both,
        "none": none,
        "sample_3b_only": b3_only[:10],
        "sample_7b_only": b7_only[:10],
    },
    ensure_ascii=False,
    indent=2,
))
