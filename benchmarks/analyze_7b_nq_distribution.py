#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


QTYPE_SET = {"who", "what", "when", "where", "which", "how", "why", "whom", "whose"}
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


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def qtype(question: str):
    toks = tokenize(question)
    if not toks:
        return "other"
    if toks[0] == "how" and len(toks) > 1:
        return f"how_{toks[1]}"
    if toks[0] in QTYPE_SET:
        return toks[0]
    return "other"


def answer_type(answers):
    if not answers:
        return "unknown"
    a = str(answers[0]).strip().lower()
    if a in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.search(r"\b\d{4}\b", a) or any(m in a for m in MONTH_WORDS):
        return "date"
    if re.fullmatch(r"[-+]?[\d,]+(\.\d+)?", a) or re.search(r"\b\d+\b", a):
        return "number"
    n = len(a.split())
    if n <= 2:
        return "short_entity"
    if n <= 6:
        return "entity"
    return "phrase"


def constraint_level(question: str):
    toks = tokenize(question)
    c = sum(1 for t in toks if t in CONSTRAINT_WORDS)
    if c == 0:
        return "c0"
    if c == 1:
        return "c1"
    return "c2plus"


def len_bin(question: str):
    n = len(tokenize(question))
    if n <= 7:
        return "len_short"
    if n <= 11:
        return "len_mid"
    return "len_long"


def parse_trace(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            rows.append(obj)
    return rows


def safe_question(obj):
    for k in ("question", "query", "Question", "instruction"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def safe_answers(obj):
    if isinstance(obj.get("gold"), list):
        vals = [str(x).strip() for x in obj["gold"] if str(x).strip()]
        if vals:
            return vals
    if isinstance(obj.get("gold"), str) and obj["gold"].strip():
        return [obj["gold"].strip()]
    for k in ("answers", "answer", "gold_answers"):
        v = obj.get(k)
        if isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                return vals
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    return []


def main():
    files = [
        Path("outputs/zerosearch_qwen25_7b_base_localretriever_standard_20260430/singlehop/nq/nq_zerosearch_qwen25_7b_base_t00_gpu_public_g0_job_public.trace.jsonl"),
        Path("outputs/zerosearch_qwen25_7b_base_localretriever_standard_20260430/singlehop/nq_resume_3117_3610/nq_zerosearch_qwen25_7b_base_t00_resume_3117_3610_gpu_public_g0_job_public.trace.jsonl"),
    ]

    rows = []
    for p in files:
        if p.exists():
            rows.extend(parse_trace(p))

    total = len(rows)
    corr = sum(int(x.get("em", 0) == 1) for x in rows)

    by_qtype = defaultdict(lambda: [0, 0])
    by_atype = defaultdict(lambda: [0, 0])
    by_len = defaultdict(lambda: [0, 0])
    by_cons = defaultdict(lambda: [0, 0])
    joint = defaultdict(lambda: [0, 0])

    for r in rows:
        em = int(r.get("em", 0) == 1)
        q = safe_question(r)
        a = safe_answers(r)
        qt = qtype(q)
        at = answer_type(a)
        lb = len_bin(q)
        cs = constraint_level(q)
        by_qtype[qt][0] += em
        by_qtype[qt][1] += 1
        by_atype[at][0] += em
        by_atype[at][1] += 1
        by_len[lb][0] += em
        by_len[lb][1] += 1
        by_cons[cs][0] += em
        by_cons[cs][1] += 1
        joint[f"{qt}|{at}|{lb}|{cs}"][0] += em
        joint[f"{qt}|{at}|{lb}|{cs}"][1] += 1

    def fmt(d):
        out = []
        for k, (c, n) in sorted(d.items(), key=lambda kv: kv[1][1], reverse=True):
            out.append({"bucket": k, "n": n, "acc": (c / n if n else 0.0), "err": (1 - c / n if n else 0.0)})
        return out

    report = {
        "total": total,
        "correct": corr,
        "acc": (corr / total if total else 0.0),
        "by_qtype": fmt(by_qtype),
        "by_answer_type": fmt(by_atype),
        "by_len": fmt(by_len),
        "by_constraint": fmt(by_cons),
        "top_hard_joint": sorted(
            [{"bucket": k, "n": n, "acc": (c / n if n else 0.0), "err": (1 - c / n if n else 0.0)} for k, (c, n) in joint.items() if n >= 20],
            key=lambda x: (x["err"], x["n"]),
            reverse=True,
        )[:30],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
