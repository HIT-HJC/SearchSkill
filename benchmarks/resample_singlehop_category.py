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
from typing import Dict, List


BASE_QTYPES = ["who", "what", "when", "where", "which", "how", "other"]
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
    "than",
    "vs",
    "versus",
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


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def infer_qtype(question: str) -> str:
    toks = tokenize(question)
    if not toks:
        return "other"
    if toks[0] in {"who", "what", "when", "where", "which"}:
        return toks[0]
    if toks[0] == "how":
        return "how"
    return "other"


def get_answers(obj: dict) -> List[str]:
    if isinstance(obj.get("answers"), list):
        vals = [str(x).strip() for x in obj["answers"] if str(x).strip()]
        if vals:
            return vals
    if isinstance(obj.get("answer"), list):
        vals = [str(x).strip() for x in obj["answer"] if str(x).strip()]
        if vals:
            return vals
    if isinstance(obj.get("answer"), str) and obj["answer"].strip():
        return [obj["answer"].strip()]
    if isinstance(obj.get("gold_answers"), list):
        vals = [str(x).strip() for x in obj["gold_answers"] if str(x).strip()]
        if vals:
            return vals
    return []


def infer_answer_type(answers: List[str]) -> str:
    if not answers:
        return "unknown"
    a = answers[0].lower()
    if re.search(r"\b\d{4}\b", a):
        return "date"
    if re.search(r"\b\d+\b", a):
        return "number"
    n = len(a.split())
    if n <= 2:
        return "short_entity"
    if n <= 6:
        return "entity"
    return "phrase"


def hard_score(question: str, answers: List[str]) -> float:
    toks = tokenize(question)
    qlen = len(toks)
    c = sum(1 for t in toks if t in CONSTRAINT_WORDS)
    qtype = infer_qtype(question)
    atype = infer_answer_type(answers)

    score = 0.0
    score += min(qlen, 24) * 0.08
    score += c * 0.8
    if qtype in {"where", "what", "other", "when"}:
        score += 0.8
    if atype in {"entity", "number", "date"}:
        score += 0.4
    if "?" in question and question.count("?") > 1:
        score += 0.3
    return score


def proportional_quota(counts: Dict[str, int], target: int, min_per_class: int) -> Dict[str, int]:
    total = sum(counts.values())
    if total <= target:
        return dict(counts)

    raw = {k: counts[k] / total * target for k in counts}
    q = {k: int(math.floor(raw[k])) for k in counts}

    # Keep small but existing classes from disappearing.
    for k, c in counts.items():
        if c >= min_per_class and q[k] < min_per_class:
            q[k] = min_per_class

    # Cap by available count.
    for k, c in counts.items():
        if q[k] > c:
            q[k] = c

    s = sum(q.values())
    if s < target:
        remain = target - s
        order = sorted(
            counts.keys(),
            key=lambda k: (raw[k] - math.floor(raw[k]), counts[k] - q[k]),
            reverse=True,
        )
        i = 0
        while remain > 0 and order:
            k = order[i % len(order)]
            if q[k] < counts[k]:
                q[k] += 1
                remain -= 1
            i += 1
            if i > 100000:
                break
    elif s > target:
        extra = s - target
        order = sorted(
            counts.keys(),
            key=lambda k: (q[k] - raw[k], q[k]),
            reverse=True,
        )
        i = 0
        while extra > 0 and order:
            k = order[i % len(order)]
            if q[k] > 0:
                q[k] -= 1
                extra -= 1
            i += 1
            if i > 100000:
                break

    return q


def sample_one_dataset(
    dataset: str,
    source_path: str,
    out_dir: str,
    sample_size: int,
    seed: int,
    min_per_class: int,
    hard_ratio: float,
):
    rng = random.Random(seed)
    rows = read_jsonl(source_path)
    n = len(rows)
    target = min(sample_size, n)

    by_cls = defaultdict(list)
    scored = {}
    for i, obj in enumerate(rows):
        q = get_question(obj)
        cls = infer_qtype(q)
        by_cls[cls].append(i)
        scored[i] = hard_score(q, get_answers(obj))

    source_dist = {k: len(v) for k, v in by_cls.items()}
    quotas = proportional_quota(source_dist, target, min_per_class=min_per_class)

    selected = []
    for cls, q in quotas.items():
        idxs = by_cls[cls]
        if q >= len(idxs):
            selected.extend(idxs)
            continue

        idxs_sorted = sorted(idxs, key=lambda x: scored[x], reverse=True)
        hard_n = min(q, max(0, int(round(q * hard_ratio))))
        hard_pick = idxs_sorted[:hard_n]
        remain_pool = idxs_sorted[hard_n:]
        rand_n = q - len(hard_pick)
        rand_pick = rng.sample(remain_pool, rand_n) if rand_n > 0 and remain_pool else []
        selected.extend(hard_pick)
        selected.extend(rand_pick)

    # If quota adjustment still under target, fill randomly from leftovers.
    if len(selected) < target:
        selected_set = set(selected)
        rest = [i for i in range(n) if i not in selected_set]
        need = target - len(selected)
        selected.extend(rng.sample(rest, need))

    selected = sorted(selected[:target])
    sampled_rows = [rows[i] for i in selected]

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), sampled_rows)
    with open(os.path.join(out_dir, "sample_indices.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    sample_dist = Counter(infer_qtype(get_question(x)) for x in sampled_rows)

    return {
        "dataset": dataset,
        "source": source_path,
        "source_size": n,
        "sample_size": target,
        "output": os.path.join(out_dir, "test.jsonl"),
        "indices_file": os.path.join(out_dir, "sample_indices.json"),
        "source_qtype_distribution": dict(sorted(source_dist.items(), key=lambda kv: kv[0])),
        "sample_qtype_distribution": dict(sorted(sample_dist.items(), key=lambda kv: kv[0])),
        "quotas": dict(sorted(quotas.items(), key=lambda kv: kv[0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="benchmarks/singlehop root")
    ap.add_argument("--data-root", required=True, help="hf_data/data root")
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument("--hard-ratio", type=float, default=0.7)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = ["nq", "triviaqa", "popqa"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(args.root, f"_backup_sample_1000_{ts}")
    os.makedirs(backup_root, exist_ok=True)

    report = {
        "strategy": "category_stratified_proportional",
        "seed": args.seed,
        "target_sample_size": args.sample_size,
        "min_per_class": args.min_per_class,
        "generated_at": ts,
        "backup_root": backup_root,
        "datasets": [],
    }

    seed_offsets = {"nq": 11, "triviaqa": 29, "popqa": 47}
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

        meta = sample_one_dataset(
            dataset=d,
            source_path=source,
            out_dir=out_dir,
            sample_size=args.sample_size,
            seed=args.seed + seed_offsets[d],
            min_per_class=args.min_per_class,
            hard_ratio=args.hard_ratio,
        )
        report["datasets"].append(meta)

    manifest_path = os.path.join(args.root, "sample_1000_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, "manifest": manifest_path, "backup_root": backup_root}, ensure_ascii=False))


if __name__ == "__main__":
    main()
