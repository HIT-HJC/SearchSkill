#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple


QTYPE_SET = {"who", "what", "when", "where", "which", "how", "why", "whom", "whose"}
MONTH_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
CONSTRAINT_WORDS = {
    "and",
    "or",
    "before",
    "after",
    "between",
    "first",
    "last",
    "earliest",
    "latest",
    "except",
    "not",
    "according",
    "during",
    "including",
    "without",
    "both",
    "either",
}


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_question(obj: dict) -> str:
    for k in ("question", "query", "Question", "instruction"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def get_answers(obj: dict) -> List[str]:
    if isinstance(obj.get("answers"), list):
        return [str(x).strip() for x in obj["answers"] if str(x).strip()]
    if isinstance(obj.get("answer"), list):
        return [str(x).strip() for x in obj["answer"] if str(x).strip()]
    if isinstance(obj.get("answer"), str) and obj["answer"].strip():
        return [obj["answer"].strip()]
    if isinstance(obj.get("gold_answers"), list):
        return [str(x).strip() for x in obj["gold_answers"] if str(x).strip()]
    return []


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def infer_qtype(question: str) -> str:
    toks = tokenize(question)
    if not toks:
        return "other"
    if toks[0] == "how" and len(toks) > 1:
        return f"how_{toks[1]}"
    if toks[0] in QTYPE_SET:
        return toks[0]
    return "other"


def infer_answer_type(answer: str) -> str:
    a = answer.strip()
    low = a.lower()
    if not a:
        return "unknown"
    if low in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.search(r"\b\d{4}\b", low) or any(m in low for m in MONTH_WORDS):
        return "date"
    if re.fullmatch(r"[-+]?[\d,]+(\.\d+)?", low):
        return "number"
    if re.search(r"\b\d+\b", low):
        return "number"
    token_count = len(a.split())
    if token_count <= 2:
        return "short_entity"
    if token_count <= 6:
        return "entity"
    return "phrase"


def len_bin(length: int, q25: int, q50: int, q75: int) -> str:
    if length <= q25:
        return "len_q1"
    if length <= q50:
        return "len_q2"
    if length <= q75:
        return "len_q3"
    return "len_q4"


def percentile(vals: List[int], p: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return s[i]


def difficulty_score(question: str, answers: List[str]) -> float:
    toks = tokenize(question)
    qlen = len(toks)
    c = sum(1 for t in toks if t in CONSTRAINT_WORDS)
    punct_bonus = 1.0 if "," in question or ";" in question else 0.0
    answer_bonus = 0.4 * max(0, len(answers) - 1)
    return 0.08 * qlen + 0.9 * c + punct_bonus + answer_bonus


def rebalance_quotas(sizes: Dict[str, int], target: int) -> Dict[str, int]:
    total = sum(sizes.values())
    if total <= target:
        return dict(sizes)
    raw = {k: sizes[k] / total * target for k in sizes}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    rem = target - sum(floor.values())
    order = sorted(sizes.keys(), key=lambda k: (raw[k] - floor[k], sizes[k]), reverse=True)
    for k in order[:rem]:
        floor[k] += 1
    return floor


def sample_dataset(
    dataset: str,
    source_path: str,
    out_dir: str,
    sample_size: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    rows = read_jsonl(source_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"{dataset}: empty source {source_path}")

    qlens = []
    metas = []
    for i, obj in enumerate(rows):
        q = get_question(obj)
        ans = get_answers(obj)
        toks = tokenize(q)
        qlen = len(toks)
        qlens.append(qlen)
        metas.append((i, q, ans, qlen))

    q25 = percentile(qlens, 0.25)
    q50 = percentile(qlens, 0.50)
    q75 = percentile(qlens, 0.75)

    strata: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    strat_counter = Counter()
    for i, q, ans, qlen in metas:
        qtype = infer_qtype(q)
        atype = infer_answer_type(ans[0] if ans else "")
        lbin = len_bin(qlen, q25, q50, q75)
        key = f"{qtype}|{atype}|{lbin}"
        score = difficulty_score(q, ans)
        strata[key].append((i, score))
        strat_counter[key] += 1

    quotas = rebalance_quotas(dict(strat_counter), min(sample_size, n))

    selected = set()
    for key, q in quotas.items():
        items = strata[key]
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        pool_size = min(len(items_sorted), max(q + 8, int(q * 1.8)))
        pool = items_sorted[:pool_size]
        if q >= len(pool):
            pick = [idx for idx, _ in pool]
        else:
            pick = [idx for idx, _ in rng.sample(pool, q)]
        selected.update(pick)

    if len(selected) < min(sample_size, n):
        remaining = []
        for key, items in strata.items():
            for idx, score in items:
                if idx not in selected:
                    remaining.append((idx, score))
        remaining.sort(key=lambda x: x[1], reverse=True)
        need = min(sample_size, n) - len(selected)
        for idx, _ in remaining[:need]:
            selected.add(idx)

    selected_indices = sorted(selected)[: min(sample_size, n)]
    sampled_rows = [rows[i] for i in selected_indices]

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), sampled_rows)
    with open(os.path.join(out_dir, "sample_indices.json"), "w", encoding="utf-8") as f:
        json.dump(selected_indices, f, ensure_ascii=False, indent=2)

    sampled_qtypes = Counter()
    for obj in sampled_rows:
        sampled_qtypes[infer_qtype(get_question(obj))] += 1

    return {
        "dataset": dataset,
        "source": source_path,
        "source_size": n,
        "sample_size": len(sampled_rows),
        "output": os.path.join(out_dir, "test.jsonl"),
        "indices_file": os.path.join(out_dir, "sample_indices.json"),
        "q_len_percentiles": {"q25": q25, "q50": q50, "q75": q75},
        "sample_qtype_distribution": sampled_qtypes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="benchmarks/singlehop root")
    ap.add_argument("--data-root", required=True, help="hf_data/data root")
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = ["nq", "triviaqa", "popqa"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(args.root, f"_backup_sample_1000_{ts}")
    os.makedirs(backup_root, exist_ok=True)

    report = {
        "strategy": "stratified_balanced_plus_harder",
        "seed": args.seed,
        "target_sample_size": args.sample_size,
        "generated_at": ts,
        "backup_root": backup_root,
        "datasets": [],
    }

    dataset_seed_offset = {"nq": 11, "triviaqa": 29, "popqa": 47}
    for d in datasets:
        source = os.path.join(args.data_root, d, "test.jsonl")
        out_dir = os.path.join(args.root, d, "sample_1000")
        if os.path.exists(out_dir):
            shutil.copytree(out_dir, os.path.join(backup_root, d), dirs_exist_ok=True)
            if args.overwrite:
                for fn in os.listdir(out_dir):
                    fp = os.path.join(out_dir, fn)
                    if os.path.isfile(fp):
                        os.remove(fp)
        meta = sample_dataset(
            dataset=d,
            source_path=source,
            out_dir=out_dir,
            sample_size=args.sample_size,
            seed=args.seed + dataset_seed_offset[d],
        )
        report["datasets"].append(meta)

    manifest_path = os.path.join(args.root, "sample_1000_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: dict(o))

    print(json.dumps({"ok": True, "manifest": manifest_path, "backup_root": backup_root}, ensure_ascii=False))


if __name__ == "__main__":
    main()
