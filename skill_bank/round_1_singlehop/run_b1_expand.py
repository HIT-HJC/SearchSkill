#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests


ROUND_DIR = Path(__file__).resolve().parent
ROOT_DIR = ROUND_DIR.parent
DEFAULT_B0_PATH = ROOT_DIR / "inputs" / "seed_skill_bank.md"
DEFAULT_PACKETS_PATH = ROUND_DIR / "artifacts" / "skill_discovery_packets.jsonl"
DEFAULT_SUMMARY_PATH = ROUND_DIR / "artifacts" / "skill_expansion_summary.json"
DEFAULT_OUTPUT_BANK_PATH = ROUND_DIR / "outputs" / "round1_skill_bank.md"
DEFAULT_OUTPUT_META_PATH = ROUND_DIR / "outputs" / "round1_skill_bank_metadata.json"
DEFAULT_RAW_RESPONSE_PATH = ROUND_DIR / "logs" / "b1_expand_raw_response.json"
DEFAULT_REQUEST_PATH = ROUND_DIR / "logs" / "b1_expand_request.json"


BUCKET_SPECS: List[Tuple[str, str]] = [
    (
        "direct_entity_lookup",
        "Direct factual lookup with a single primary target. Fast answer when one focused search should surface the fact.",
    ),
    (
        "alias_or_renaming_lookup",
        "Questions asking for a real name, alternate name, former name, nickname, or renamed entity.",
    ),
    (
        "temporal_attribute_lookup",
        "Questions asking for a year, date, season, period, or temporal boundary tied to one entity or event.",
    ),
    (
        "numeric_attribute_lookup",
        "Questions asking for counts, heights, lengths, populations, rankings, or other numeric attributes.",
    ),
    (
        "location_targeted_lookup",
        "Questions asking where something happened, is located, was born, was held, or is based.",
    ),
    (
        "person_identity_lookup",
        "Questions asking who or whose, especially person-role and identity resolution questions.",
    ),
    (
        "organization_attribute_lookup",
        "Questions centered on teams, companies, universities, institutions, or other organizations.",
    ),
    (
        "yes_no_targeted_verification",
        "Binary questions that need one precise grounded verification search before answering yes or no.",
    ),
    (
        "multi_constraint_entity_match",
        "Single-hop questions with multiple constraints, long wording, or multiple named entities that require careful query anchoring.",
    ),
    (
        "exact_span_finalization",
        "Cases where the answer must preserve exact titles, aliases, units, or quoted spans from evidence.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an expanded B1 single-hop SkillBank using GPT-5.4 via a responses-compatible provider."
    )
    parser.add_argument("--b0-path", type=Path, default=DEFAULT_B0_PATH)
    parser.add_argument("--packets-path", type=Path, default=DEFAULT_PACKETS_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-bank-path", type=Path, default=DEFAULT_OUTPUT_BANK_PATH)
    parser.add_argument("--output-meta-path", type=Path, default=DEFAULT_OUTPUT_META_PATH)
    parser.add_argument("--raw-response-path", type=Path, default=DEFAULT_RAW_RESPONSE_PATH)
    parser.add_argument("--request-path", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--base-url", type=str, default="https://w.ciykj.cn")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="xhigh")
    parser.add_argument("--verbosity", type=str, default="high")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--bucket-example-limit", type=int, default=8)
    parser.add_argument("--top-profile-limit", type=int, default=18)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def unique_examples(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = set()
    results = []
    for example in packet.get("representative_examples", []):
        question = example.get("question", "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        results.append(example)
    return results


def packet_matches(bucket_id: str, packet: Dict[str, Any]) -> bool:
    profile = packet.get("profile", {})
    flags = set(profile.get("flags", []))
    wh_word = (profile.get("wh_word") or "").lower()
    answer_form = (profile.get("answer_form_hint") or "").lower()
    entity_bin = (profile.get("entity_bin") or "").lower()
    token_bin = (profile.get("token_bin") or "").lower()

    if bucket_id == "direct_entity_lookup":
        return "direct_lookup" in flags
    if bucket_id == "alias_or_renaming_lookup":
        return "alias" in flags
    if bucket_id == "temporal_attribute_lookup":
        return "temporal" in flags or wh_word == "when"
    if bucket_id == "numeric_attribute_lookup":
        return "numeric" in flags or answer_form == "number" or wh_word == "how"
    if bucket_id == "location_targeted_lookup":
        return "location" in flags or wh_word == "where"
    if bucket_id == "person_identity_lookup":
        return "person" in flags or wh_word in {"who", "whose"}
    if bucket_id == "organization_attribute_lookup":
        return "organization" in flags
    if bucket_id == "yes_no_targeted_verification":
        return "yes_no" in flags or wh_word in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had"}
    if bucket_id == "multi_constraint_entity_match":
        return entity_bin != "single_entity" or token_bin == "long"
    if bucket_id == "exact_span_finalization":
        return "quoted_span" in flags or answer_form in {"long_span", "list_like"}
    return False


def signature_overview(packet: Dict[str, Any]) -> Dict[str, Any]:
    profile = packet.get("profile", {})
    return {
        "dataset": packet.get("dataset"),
        "group_size": packet.get("group_size"),
        "wh_word": profile.get("wh_word"),
        "answer_form_hint": profile.get("answer_form_hint"),
        "entity_bin": profile.get("entity_bin"),
        "token_bin": profile.get("token_bin"),
        "flags": profile.get("flags", []),
    }


def build_summary(
    packets: List[Dict[str, Any]],
    bucket_example_limit: int,
    top_profile_limit: int,
) -> Dict[str, Any]:
    dataset_packet_counts: Dict[str, int] = defaultdict(int)
    dataset_group_sizes: Dict[str, int] = defaultdict(int)
    bucket_data: Dict[str, Dict[str, Any]] = {
        bucket_id: {
            "bucket_id": bucket_id,
            "description": description,
            "matched_packet_count": 0,
            "matched_group_size": 0,
            "datasets": defaultdict(int),
            "examples": [],
        }
        for bucket_id, description in BUCKET_SPECS
    }

    top_profiles = []
    for packet in sorted(packets, key=lambda item: item.get("group_size", 0), reverse=True)[:top_profile_limit]:
        profile = signature_overview(packet)
        profile["sample_questions"] = [example.get("question") for example in unique_examples(packet)[:3]]
        top_profiles.append(profile)

    for packet in packets:
        dataset = packet.get("dataset", "unknown")
        group_size = int(packet.get("group_size", 0))
        dataset_packet_counts[dataset] += 1
        dataset_group_sizes[dataset] += group_size
        examples = unique_examples(packet)
        for bucket_id, _ in BUCKET_SPECS:
            if not packet_matches(bucket_id, packet):
                continue
            bucket = bucket_data[bucket_id]
            bucket["matched_packet_count"] += 1
            bucket["matched_group_size"] += group_size
            bucket["datasets"][dataset] += 1
            for example in examples:
                if len(bucket["examples"]) >= bucket_example_limit:
                    break
                if any(existing["question"] == example.get("question") for existing in bucket["examples"]):
                    continue
                bucket["examples"].append(
                    {
                        "dataset": dataset,
                        "question": example.get("question"),
                        "primary_answer": example.get("primary_answer"),
                        "golden_answers": example.get("golden_answers", [])[:5],
                        "flags": example.get("flags", []),
                    }
                )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "packet_count": len(packets),
        "dataset_packet_counts": dict(dataset_packet_counts),
        "dataset_group_sizes": dict(dataset_group_sizes),
        "bucket_summaries": [
            {
                **bucket,
                "datasets": dict(bucket["datasets"]),
            }
            for bucket in bucket_data.values()
            if bucket["matched_packet_count"] > 0
        ],
        "top_profiles": top_profiles,
    }


def render_bucket_summary(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("Overall packet coverage:")
    lines.append(json.dumps(
        {
            "packet_count": summary["packet_count"],
            "dataset_packet_counts": summary["dataset_packet_counts"],
            "dataset_group_sizes": summary["dataset_group_sizes"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    lines.append("")
    lines.append("Bucket summaries:")
    for bucket in summary["bucket_summaries"]:
        lines.append(f"- {bucket['bucket_id']}: {bucket['description']}")
        lines.append(
            json.dumps(
                {
                    "matched_packet_count": bucket["matched_packet_count"],
                    "matched_group_size": bucket["matched_group_size"],
                    "datasets": bucket["datasets"],
                    "examples": bucket["examples"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    lines.append("")
    lines.append("Top frequent profiles:")
    lines.append(json.dumps(summary["top_profiles"], ensure_ascii=False, indent=2))
    return "\n".join(lines)


def extract_output_text(response_json: Dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]
    texts: List[str] = []
    for item in response_json.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    texts.append(content["text"])
        elif item.get("type") in {"output_text", "text"} and item.get("text"):
            texts.append(item["text"])
    return "\n".join(texts).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [text.strip()]
    unfenced = re.sub(r"^```(?:json)?\s*", "", text.strip())
    unfenced = re.sub(r"\s*```$", "", unfenced)
    candidates.append(unfenced.strip())
    base = unfenced.strip()
    start = base.find("{")
    end = base.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(base[start : end + 1].strip())
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed, _ = decoder.raw_decode(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse a JSON object from model output.")


def build_prompts(bank_text: str, summary: Dict[str, Any]) -> Tuple[str, str]:
    system_prompt = (
        "You are an expert retrieval-skill designer building a search-oriented skill bank for a smaller student model. "
        "You may aggressively add new reusable skills when the evidence shows a stable retrieval pattern. "
        "Do not stay conservative just because the base bank is small. "
        "However, every new skill must have crisp boundaries, a distinct retrieval/query pattern, a verification rule, and a clear avoid condition. "
        "Avoid near-duplicate skills, entity-specific hacks, and skills that merely restate the answer type."
    )

    user_prompt = f"""
We are evolving a base retrieval skill bank into `round1_skill_bank` using single-hop training data from NQ and TriviaQA.

Base bank:
```markdown
{bank_text.strip()}
```

Evidence summary from grouped single-hop packets:
{render_bucket_summary(summary)}

Task:
1. Preserve the useful multihop core skills from seed unless a rewrite is clearly necessary.
2. Expand the bank with as many high-quality single-hop skills as the evidence supports.
3. Prefer specialized additions over overloading one generic single-hop skill when the retrieval strategy or verification pattern is meaningfully different.
4. Keep the bank reusable for downstream trajectory generation and later multihop rounds.

Strong guidance for new skills:
- Good additions usually correspond to repeatable retrieval behaviors such as alias resolution, temporal attribute lookup, numeric attribute lookup, binary verification, multi-constraint entity matching, or exact-span finalization.
- A new skill is justified when it changes how the model should search, verify, or decide not to use the skill.
- Do not invent skills that are too narrow, too dataset-specific, or differ only by answer surface form with no retrieval difference.

Formatting requirements:
- Return valid JSON only.
- Use this schema:
{{
  "strategy_summary": "short paragraph",
  "new_skills": [
    {{
      "skill_id": "kebab-case-id",
      "skill_text": "2-4 sentences. Must include when to use, how to search, how to verify, and when not to use.",
      "reason": "why this new skill is justified",
      "supporting_buckets": ["bucket-id"]
    }}
  ],
  "refined_skills": [
    {{
      "skill_id": "existing-skill-id",
      "skill_text": "updated text",
      "reason": "why refine"
    }}
  ],
  "final_bank_markdown": "# Retrieval Skill Bank B1\\n\\n`skill-id`\\nSkill text...\\n"
}}

Bank construction rules:
- Final bank should usually end up with roughly 10 to 16 skills.
- Keep the original multihop skills unless there is a concrete refinement.
- New skill IDs must be distinct, reusable, and not tied to a single entity or dataset.
- Each skill text should read like an actionable retrieval policy, not a taxonomy label.
- `final_bank_markdown` must include the full final bank, not only the new skills.
""".strip()
    return system_prompt, user_prompt


def call_responses_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str,
    verbosity: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": verbosity},
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return {"payload": payload, "response_json": response.json()}


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment.")

    bank_text = args.b0_path.read_text(encoding="utf-8")
    packets = list(load_jsonl(args.packets_path))
    summary = build_summary(
        packets,
        bucket_example_limit=args.bucket_example_limit,
        top_profile_limit=args.top_profile_limit,
    )
    dump_json(args.summary_path, summary)

    system_prompt, user_prompt = build_prompts(bank_text, summary)
    api_result = call_responses_api(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
    )

    sanitized_payload = dict(api_result["payload"])
    dump_json(args.request_path, sanitized_payload)
    dump_json(args.raw_response_path, api_result["response_json"])

    output_text = extract_output_text(api_result["response_json"])
    parsed = extract_json_object(output_text)
    final_bank_markdown = parsed.get("final_bank_markdown", "").strip()
    if not final_bank_markdown.startswith("#"):
        raise ValueError("Model output did not include a valid final bank markdown block.")

    dump_text(args.output_bank_path, final_bank_markdown + "\n")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "usage": api_result["response_json"].get("usage", {}),
        "strategy_summary": parsed.get("strategy_summary"),
        "new_skills": parsed.get("new_skills", []),
        "refined_skills": parsed.get("refined_skills", []),
        "summary_path": str(args.summary_path),
        "raw_response_path": str(args.raw_response_path),
    }
    dump_json(args.output_meta_path, metadata)

    print(f"Wrote summary to {args.summary_path}")
    print(f"Wrote final bank to {args.output_bank_path}")
    print(f"Wrote metadata to {args.output_meta_path}")
    print(json.dumps(metadata.get("usage", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
