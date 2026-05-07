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


CONSTRAINT_WORDS = {
    "and", "or", "before", "after", "between", "first", "last", "earliest",
    "latest", "except", "not", "according", "during", "including", "without",
    "both", "either", "than", "vs", "versus", "while", "although",
}


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_question(obj):
    for k in ("question", "query", "Question", "instruction"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def get_answers(obj):
    for k in ("answers", "gold_answers"):
        v = obj.get(k)
        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                return vals
    v = obj.get("answer")
    if isinstance(v, list):
        vals = [str(x).strip() for x in v if str(x).strip()]
        if vals:
            return vals
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def toks(text):
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def qtype(question):
    tt = toks(question)
    if not tt:
        return "other"
    if tt[0] in {"who", "what", "when", "where", "which"}:
        return tt[0]
    if tt[0] == "how":
        return "how"
    return "other"


def template(question):
    q = question.lower().strip()
    if q.startswith("what is the capital of"):
        return "capital"
    if q.startswith("in what city was") or q.startswith("what city was"):
        return "born_city"
    if "occupation" in q:
        return "occupation"
    if q.startswith("what genre") or "what genre is" in q:
        return "genre"
    return "other"


def answer_type(answers):
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


def hard_score(question, answers, ds):
    tt = toks(question)
    qlen = len(tt)
    cons = sum(1 for t in tt if t in CONSTRAINT_WORDS)
    qt = qtype(question)
    at = answer_type(answers)
    tmpl = template(question)

    s = 0.0
    s += min(qlen, 30) * 0.09
    s += cons * 0.9
    s += question.count(",") * 0.2
    s += question.count("(") * 0.3
    if qt in {"what", "where", "other", "when"}:
        s += 0.8
    if at in {"entity", "number", "date"}:
        s += 0.4

    if ds == "popqa":
        if tmpl in {"capital", "born_city", "occupation", "genre"}:
            s -= 2.0
        else:
            s += 0.6
    elif ds == "triviaqa":
        if qt == "other":
            s += 0.3
        if qlen >= 12:
            s += 0.6
    return s


def proportional_quota(counts, target, min_per_class):
    total = sum(counts.values())
    raw = {k: counts[k] / total * target for k in counts}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k, c in counts.items():
        if c >= min_per_class and out[k] < min_per_class:
            out[k] = min(min_per_class, c)
    for k, c in counts.items():
        out[k] = min(out[k], c)
    cur = sum(out.values())
    if cur < target:
        order = sorted(counts, key=lambda k: (raw[k] - math.floor(raw[k]), counts[k] - out[k]), reverse=True)
        i = 0
        while cur < target and order:
            k = order[i % len(order)]
            if out[k] < counts[k]:
                out[k] += 1
                cur += 1
            i += 1
    elif cur > target:
        order = sorted(counts, key=lambda k: (out[k] - raw[k], out[k]), reverse=True)
        i = 0
        while cur > target and order:
            k = order[i % len(order)]
            if out[k] > 0:
                out[k] -= 1
                cur -= 1
            i += 1
    return out


def sample_dataset(ds, src, out_dir, sample_size, seed, hard_ratio, min_per_class):
    rng = random.Random(seed)
    rows = read_jsonl(src)
    by_cls = defaultdict(list)
    meta = {}
    for i, obj in enumerate(rows):
        q = get_question(obj)
        by_cls[qtype(q)].append(i)
        meta[i] = (q, get_answers(obj), template(q))

    quotas = proportional_quota({k: len(v) for k, v in by_cls.items()}, sample_size, min_per_class)
    selected = []
    easy_caps = {"capital": 10, "born_city": 15, "occupation": 15, "genre": 30} if ds == "popqa" else {}
    easy_used = Counter()

    for cls, qn in quotas.items():
        idxs = by_cls[cls]
        ranked = sorted(idxs, key=lambda i: hard_score(meta[i][0], meta[i][1], ds), reverse=True)
        hard_n = min(qn, int(round(qn * hard_ratio)))
        picked = []
        for i in ranked:
            tmpl = meta[i][2]
            if tmpl in easy_caps and easy_used[tmpl] >= easy_caps[tmpl]:
                continue
            picked.append(i)
            if tmpl in easy_caps:
                easy_used[tmpl] += 1
            if len(picked) >= hard_n:
                break
        remaining = [i for i in idxs if i not in set(picked)]
        rng.shuffle(remaining)
        for i in remaining:
            if len(picked) >= qn:
                break
            tmpl = meta[i][2]
            if tmpl in easy_caps and easy_used[tmpl] >= easy_caps[tmpl]:
                continue
            picked.append(i)
            if tmpl in easy_caps:
                easy_used[tmpl] += 1
        selected.extend(picked)

    selected = list(dict.fromkeys(selected))
    if len(selected) < sample_size:
        all_ranked = sorted(range(len(rows)), key=lambda i: hard_score(meta[i][0], meta[i][1], ds), reverse=True)
        chosen = set(selected)
        for i in all_ranked:
            if len(selected) >= sample_size:
                break
            if i in chosen:
                continue
            tmpl = meta[i][2]
            if tmpl in easy_caps and easy_used[tmpl] >= easy_caps[tmpl]:
                continue
            selected.append(i)
            chosen.add(i)
            if tmpl in easy_caps:
                easy_used[tmpl] += 1

    selected = sorted(selected[:sample_size])
    sampled = [rows[i] for i in selected]
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), sampled)
    with open(os.path.join(out_dir, "sample_indices.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    tmpl_dist = Counter(template(get_question(x)) for x in sampled)
    cls_dist = Counter(qtype(get_question(x)) for x in sampled)
    return {
        "dataset": ds,
        "source": src,
        "sample_size": len(sampled),
        "output": os.path.join(out_dir, "test.jsonl"),
        "template_distribution": dict(tmpl_dist),
        "qtype_distribution": dict(cls_dist),
        "easy_caps": easy_caps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(args.root, f"_backup_triviaqa_popqa_harder_{ts}")
    os.makedirs(backup_root, exist_ok=True)
    report = {
        "strategy": "category_stratified_harder_dataset_specific",
        "generated_at": ts,
        "backup_root": backup_root,
        "datasets": [],
    }

    configs = {
        "triviaqa": {"hard_ratio": 0.9, "min_per_class": 14},
        "popqa": {"hard_ratio": 0.9, "min_per_class": 20},
    }
    for ds, cfg in configs.items():
        out_dir = os.path.join(args.root, ds, "sample_1000")
        if os.path.exists(out_dir):
            shutil.copytree(out_dir, os.path.join(backup_root, ds), dirs_exist_ok=True)
            if args.overwrite:
                for fn in os.listdir(out_dir):
                    fp = os.path.join(out_dir, fn)
                    if os.path.isfile(fp):
                        os.remove(fp)
        meta = sample_dataset(
            ds,
            os.path.join(args.data_root, ds, "test.jsonl"),
            out_dir,
            args.sample_size,
            args.seed + (17 if ds == "triviaqa" else 31),
            cfg["hard_ratio"],
            cfg["min_per_class"],
        )
        report["datasets"].append(meta)

    manifest = os.path.join(args.root, "triviaqa_popqa_harder_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, "manifest": manifest, "backup_root": backup_root}, ensure_ascii=False))


if __name__ == "__main__":
    main()
