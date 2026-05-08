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
DEFAULT_B2_PATH = ROOT_DIR / "round_2_hotpotqa" / "outputs" / "round2_skill_bank.md"
DEFAULT_PACKETS_PATH = ROUND_DIR / "artifacts" / "skill_discovery_packets.jsonl"
DEFAULT_SUMMARY_PATH = ROUND_DIR / "artifacts" / "skill_expansion_summary.json"
DEFAULT_OUTPUT_BANK_PATH = ROUND_DIR / "outputs" / "round3_skill_bank.md"
DEFAULT_OUTPUT_META_PATH = ROUND_DIR / "outputs" / "round3_skill_bank_metadata.json"
DEFAULT_RAW_RESPONSE_PATH = ROUND_DIR / "logs" / "b3_expand_raw_response.json"
DEFAULT_REQUEST_PATH = ROUND_DIR / "logs" / "b3_expand_request.json"


BUCKET_SPECS: List[Tuple[str, str]] = [
    (
        "compositional_core",
        "Classic 2Wiki compositional questions where one relation leads to an intermediate entity and then to the answer.",
    ),
    (
        "comparison_core",
        "Pairwise comparison questions that require retrieving evidence for both sides before deciding.",
    ),
    (
        "bridge_comparison_core",
        "Bridge-comparison questions where a hidden intermediate entity must be found before a comparison or selection can be made.",
    ),
    (
        "inference_core",
        "Inference-style questions where the answer is not exposed by one direct bridge and requires joining multiple supported relations.",
    ),
    (
        "relation_chain_composition",
        "Compositional questions with explicit relation-chain structure that benefit from edge-by-edge retrieval instead of a broad bridge query.",
    ),
    (
        "four_hop_composition",
        "Longer 4-hop questions where the retrieval policy must preserve intermediate checkpoints and avoid skipping hidden steps.",
    ),
    (
        "same_attribute_compare",
        "Comparison questions where two candidates must be checked against the same attribute or relation in a like-for-like way.",
    ),
    (
        "yes_no_verification",
        "Yes/no or verification-heavy 2Wiki questions where the answer depends on confirming a full relational claim rather than one page lookup.",
    ),
    (
        "temporal_numeric_composition",
        "Compositional questions whose endpoint is a date, year, count, ranking, or other numeric value after one or more intermediate hops.",
    ),
    (
        "dense_entity_disambiguation",
        "Questions with dense entities, multiple candidates, or alias-heavy intermediate pages that require stronger compositional disambiguation.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an expanded B3 2Wiki SkillBank using GPT-5.4 via a responses-compatible provider."
    )
    parser.add_argument("--b2-path", type=Path, default=DEFAULT_B2_PATH)
    parser.add_argument("--packets-path", type=Path, default=DEFAULT_PACKETS_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-bank-path", type=Path, default=DEFAULT_OUTPUT_BANK_PATH)
    parser.add_argument("--output-meta-path", type=Path, default=DEFAULT_OUTPUT_META_PATH)
    parser.add_argument("--raw-response-path", type=Path, default=DEFAULT_RAW_RESPONSE_PATH)
    parser.add_argument("--request-path", type=Path, default=DEFAULT_REQUEST_PATH)
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="xhigh")
    parser.add_argument("--verbosity", type=str, default="high")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=15000)
    parser.add_argument("--bucket-example-limit", type=int, default=10)
    parser.add_argument("--top-profile-limit", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
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
    results: List[Dict[str, Any]] = []
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
    native_type = (profile.get("native_type") or "").lower()
    entity_bin = (profile.get("entity_bin") or "").lower()
    hop_count = int(profile.get("hop_count") or 0)

    if bucket_id == "compositional_core":
        return native_type == "compositional"
    if bucket_id == "comparison_core":
        return native_type == "comparison" or "comparison" in flags
    if bucket_id == "bridge_comparison_core":
        return native_type == "bridge_comparison"
    if bucket_id == "inference_core":
        return native_type == "inference"
    if bucket_id == "relation_chain_composition":
        return "relation_chain" in flags
    if bucket_id == "four_hop_composition":
        return hop_count >= 4
    if bucket_id == "same_attribute_compare":
        return (native_type == "comparison" or "comparison" in flags) and "same_attribute" in flags
    if bucket_id == "yes_no_verification":
        return "yes_no" in flags or "verification" in flags
    if bucket_id == "temporal_numeric_composition":
        return bool({"temporal", "numeric", "time_anchor"} & flags)
    if bucket_id == "dense_entity_disambiguation":
        return (
            ("verification" in flags or "dense_entities" in flags)
            and entity_bin in {"multi_entity", "dense_entity"}
        )
    return False


def packet_profile(packet: Dict[str, Any]) -> Dict[str, Any]:
    profile = packet.get("profile", {})
    return {
        "dataset": packet.get("dataset"),
        "group_size": packet.get("group_size"),
        "native_type": profile.get("native_type"),
        "hop_count": profile.get("hop_count"),
        "wh_word": profile.get("wh_word"),
        "answer_form_hint": profile.get("answer_form_hint"),
        "entity_bin": profile.get("entity_bin"),
        "token_bin": profile.get("token_bin"),
        "flags": profile.get("flags", []),
        "native_summary": profile.get("native_summary", {}),
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
        profile = packet_profile(packet)
        profile["sample_questions"] = [example.get("question") for example in unique_examples(packet)[:3]]
        top_profiles.append(profile)

    for packet in packets:
        dataset = packet.get("dataset", "unknown")
        group_size = int(packet.get("group_size", 0))
        dataset_packet_counts[dataset] += 1
        dataset_group_sizes[dataset] += group_size
        examples = unique_examples(packet)
        for bucket_id, _description in BUCKET_SPECS:
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
                        "native_type": example.get("native_type"),
                        "metadata_type": example.get("metadata_type"),
                        "metadata_level": example.get("metadata_level"),
                        "supporting_titles": example.get("supporting_titles", []),
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
    lines.append(
        json.dumps(
            {
                "packet_count": summary["packet_count"],
                "dataset_packet_counts": summary["dataset_packet_counts"],
                "dataset_group_sizes": summary["dataset_group_sizes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
        "You are an expert retrieval-skill designer evolving a search-oriented skill bank for a smaller student model. "
        "You should actively add reusable multihop skills when the evidence shows a stable retrieval pattern. "
        "Preserve the useful single-hop and HotpotQA multihop spine from earlier rounds. "
        "Every compositional skill must specify when to use it, how to search across hops, how to verify the final relation, and when not to use it. "
        "Pay special attention to routing boundaries so new compositional skills do not over-trigger on shorter bridge questions."
    )

    user_prompt = f"""
We are evolving `round2_skill_bank` into `round3_skill_bank` using 2Wiki training evidence.

Current bank:
```markdown
{bank_text.strip()}
```

2Wiki grouped evidence summary:
{render_bucket_summary(summary)}

Task:
1. Keep the B1 single-hop and B2 hotpot-style multihop gains intact.
2. Expand the bank with high-quality compositional skills for relation chains, bridge-comparison hybrids, inference joins, and longer-hop retrieval when justified.
3. Refine existing skills if their boundaries are currently too broad, too vague, or fail to preserve intermediate entities in longer chains.
4. Write stronger avoid conditions than before so later routing can keep shorter bridge questions on the right path.

Important guidance:
- Good B3 additions usually correspond to reusable search plans such as relation-chain decomposition, bridge-comparison planning, inference joins with checkpointed intermediates, 4-hop checkpointing, temporal/numeric endpoints after composition, or compositional disambiguation.
- Add a new skill only if it changes search planning, verification behavior, or the decision to stop searching.
- Avoid near-duplicate skills that differ only by answer surface form or by a tiny wording variation.
- Preserve compatibility with the later MuSiQue round.

Formatting requirements:
- Return valid JSON only.
- Use this schema:
{{
  "strategy_summary": "short paragraph",
  "new_skills": [
    {{
      "skill_id": "kebab-case-id",
      "skill_text": "2-5 sentences. Must include when to use, how to search, how to verify, and when not to use.",
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
  "final_bank_markdown": "# Retrieval Skill Bank B3\\n\\n`skill-id`\\nSkill text...\\n"
}}

Bank construction rules:
- Final bank should usually end up with roughly 16 to 22 skills.
- Keep the useful B1 and B2 skills unless a targeted rewrite is clearly better.
- New skill IDs must be reusable and not tied to a single dataset example.
- Each skill text should read like an actionable retrieval policy.
- `final_bank_markdown` must include the full final bank, not only the additions.
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
    base = base_url.rstrip("/")
    url = f"{base}/responses" if base.endswith("/v1") else f"{base}/v1/responses"
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
        url,
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

    bank_text = args.b2_path.read_text(encoding="utf-8")
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

    dump_json(args.request_path, dict(api_result["payload"]))
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
    print(json.dumps(metadata.get('usage', {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
