from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from common import (
    SUPPORT_ONLY_SKILLS,
    append_jsonl,
    dump_json,
    exact_match_multi,
    load_jsonl,
    load_skill_bank,
    load_skill_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-5.4 teacher rollout over a trajectory manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill-bank-path", type=Path, required=True)
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="xhigh")
    parser.add_argument("--verbosity", type=str, default="medium")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--api-retry-backoff", type=float, default=8.0)
    parser.add_argument("--retriever-host", type=str, default=os.environ.get("RETRIEVER_HOST", "127.0.0.1"))
    parser.add_argument("--retriever-port", type=int, default=int(os.environ.get("RETRIEVER_PORT", "8000")))
    parser.add_argument("--retriever-topk", type=int, default=3)
    parser.add_argument("--retriever-timeout", type=int, default=45)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


class RetrieverClient:
    def __init__(self, host: str, port: int, topk: int, timeout: int):
        self.url = f"http://{host}:{port}/retrieve"
        self.topk = topk
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def search(self, query: str) -> str:
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        rows = response.json()["result"][0]
        passages: List[str] = []
        for idx, row in enumerate(rows, start=1):
            contents = row["document"]["contents"]
            parts = contents.split("\n")
            title = parts[0].strip().strip('"')
            text = "\n".join(parts[1:]).strip()
            passages.append(f'Doc {idx}(Title: "{title}") {text}')
        return "\n".join(passages)


def extract_text(obj: Any) -> str:
    chunks: List[str] = []
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for item in obj:
            text = extract_text(item)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            chunks.append(obj["text"])
        for key in ("output", "content", "items"):
            if key in obj:
                text = extract_text(obj[key])
                if text:
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in: {text[:400]}")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Unclosed JSON object in: {text[:400]}")


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
    max_retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": verbosity},
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }
    session = requests.Session()
    base = base_url.rstrip("/")
    url = f"{base}/responses" if base.endswith("/v1") else f"{base}/v1/responses"
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout_seconds,
            )
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"retryable status {response.status_code}: {clip_text(response.text, 300)}",
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {408, 429, 500, 502, 503, 504} or attempt >= max_retries:
                raise
            last_error = exc
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            last_error = exc
        sleep_seconds = retry_backoff * attempt
        print(
            f"[warn] responses api attempt {attempt}/{max_retries} failed: {last_error}. "
            f"retrying in {sleep_seconds:.1f}s",
            flush=True,
        )
        time.sleep(sleep_seconds)
    raise RuntimeError(f"responses api failed after {max_retries} attempts") from last_error


def clip_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_current_record(
    path: Path,
    record: Dict[str, Any],
    *,
    processed_count: int,
    total_requested: int,
) -> None:
    dump_json(
        path,
        {
            "updated_at": iso_now(),
            "record_id": record["id"],
            "dataset": record["dataset"],
            "sample_origin": record["sample_origin"],
            "question": record["question"],
            "candidate_primary_skills": record.get("candidate_primary_skills", []),
            "failure_info": record.get("failure_info"),
            "processed_count": processed_count,
            "total_requested": total_requested,
            "remaining_count": max(0, total_requested - processed_count),
        },
    )


def write_current_step(
    path: Path,
    *,
    record: Dict[str, Any],
    phase: str,
    step_index: int,
    completed_steps: List[Dict[str, Any]],
    query: str = "",
    action: str = "",
    checkpoint: Dict[str, Any] | None = None,
    note: str = "",
) -> None:
    dump_json(
        path,
        {
            "updated_at": iso_now(),
            "record_id": record["id"],
            "dataset": record["dataset"],
            "question": record["question"],
            "phase": phase,
            "step_index": step_index,
            "completed_step_count": len(completed_steps),
            "action": action,
            "query": query,
            "checkpoint": checkpoint or {},
            "note": note,
        },
    )


def build_rollout_prompts(
    record: Dict[str, Any],
    skill_bank_text: str,
    step_index: int,
    trace_steps: List[Dict[str, Any]],
    latest_information: str,
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert retrieval teacher generating high-quality executable supervision traces for a smaller retrieval student. "
        "Return one JSON object only. "
        "Choose exactly one primary_skill from the provided retrieval skill bank. "
        "Support skills are optional, but they must stay secondary and never replace the main planning skill. "
        "Do not use verifier-only skills as the primary skill before the final answer. "
        "Use action=search to discover the next entity or attribute, action=verify to do a tighter confirmation search, and action=answer only when the current evidence is enough. "
        "Queries should be concise and targeted. "
        "Answers must be the shortest exact span supported by evidence."
    )
    state = {
        "dataset": record["dataset"],
        "sample_origin": record["sample_origin"],
        "task_family": record["task_family"],
        "question": record["question"],
        "metadata_summary": record.get("metadata_summary", {}),
        "candidate_primary_skills": record.get("candidate_primary_skills", []),
        "suggested_support_skills": record.get("suggested_support_skills", []),
        "failure_info": record.get("failure_info"),
        "step_index": step_index,
        "trace_so_far": trace_steps,
        "latest_information": latest_information,
    }
    user_prompt = (
        f"Skill bank:\n```markdown\n{skill_bank_text}\n```\n\n"
        "Current state:\n"
        f"```json\n{json.dumps(state, ensure_ascii=False, indent=2)}\n```\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "primary_skill": "skill-id",\n'
        '  "support_skills": ["skill-id"],\n'
        '  "action": "search|verify|answer",\n'
        '  "query": "query or empty string",\n'
        '  "checkpoint": {\n'
        '    "resolved": "current resolved entity/value checkpoint",\n'
        '    "remaining_goal": "what remains to solve"\n'
        "  },\n"
        '  "draft_answer": "short string or empty",\n'
        '  "rationale": "1-2 short sentences",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "- The primary skill must be a real skill id from the bank.\n"
        "- Use at most 2 support_skills.\n"
        "- If action is answer, leave query empty.\n"
        "- If action is search or verify, query must be non-empty.\n"
        "- Prefer a bridge, chain, comparison, or checkpoint skill as primary planning, not verbatim extraction.\n"
    )
    return system_prompt, user_prompt


def build_finalizer_prompts(record: Dict[str, Any], trace_steps: List[Dict[str, Any]], evidence_text: str, draft_answer: str) -> tuple[str, str]:
    system_prompt = (
        "You are finalizing an answer span for an exact-match retrieval dataset. "
        "Return one JSON object only. "
        "Prefer the shortest exact answer span that is explicitly supported by the evidence."
    )
    user_prompt = (
        f"Question: {record['question']}\n"
        f"Draft answer: {draft_answer}\n"
        f"Trace summary:\n{json.dumps(trace_steps, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        'Return JSON with this schema:\n{ "final_answer": "short exact answer span", "normalization_skill": "verbatim-evidence-span|answer-grounding-check", "reason": "short reason" }'
    )
    return system_prompt, user_prompt


def validate_trajectory(record: Dict[str, Any], trajectory: Dict[str, Any], legal_skill_ids: List[str]) -> Dict[str, Any]:
    unknown_skills: List[str] = []
    support_only_primary_steps: List[int] = []
    search_steps = 0
    for idx, step in enumerate(trajectory["steps"]):
        primary = step.get("primary_skill") or ""
        if primary not in legal_skill_ids:
            unknown_skills.append(primary)
        if step.get("action") != "answer" and primary in SUPPORT_ONLY_SKILLS:
            support_only_primary_steps.append(idx)
        for skill_id in step.get("support_skills", []):
            if skill_id not in legal_skill_ids:
                unknown_skills.append(skill_id)
        if step.get("action") in {"search", "verify"}:
            search_steps += 1
    first_primary = trajectory["steps"][0]["primary_skill"] if trajectory["steps"] else ""
    validation = {
        "em": exact_match_multi(trajectory["final_answer"], record.get("gold_answers", [])),
        "has_search": search_steps > 0,
        "unknown_skills": sorted(set(skill for skill in unknown_skills if skill)),
        "support_only_primary_steps": support_only_primary_steps,
        "route_matches_candidates": first_primary in record.get("candidate_primary_skills", []),
    }
    validation["passed"] = (
        validation["em"]
        and validation["has_search"]
        and not validation["unknown_skills"]
        and not validation["support_only_primary_steps"]
    )
    return validation


def run_one_example(
    *,
    record: Dict[str, Any],
    skill_bank_text: str,
    legal_skill_ids: List[str],
    retriever: RetrieverClient,
    api_key: str,
    args: argparse.Namespace,
    current_step_path: Path,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    evidence_blocks: List[str] = []
    latest_information = ""
    final_answer = ""
    raw_outputs: List[str] = []

    for step_index in range(args.max_steps):
        write_current_step(
            current_step_path,
            record=record,
            phase="rollout_model_call",
            step_index=step_index,
            completed_steps=steps,
            note="waiting for model step output",
        )
        system_prompt, user_prompt = build_rollout_prompts(
            record=record,
            skill_bank_text=skill_bank_text,
            step_index=step_index,
            trace_steps=steps,
            latest_information=clip_text(latest_information, 7000),
        )
        response_json = call_responses_api(
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
            max_retries=args.api_max_retries,
            retry_backoff=args.api_retry_backoff,
        )
        output_text = extract_text(response_json)
        raw_outputs.append(output_text)
        parsed = extract_json_object(output_text)
        step = {
            "step": step_index,
            "primary_skill": str(parsed.get("primary_skill", "")).strip(),
            "support_skills": [str(item).strip() for item in parsed.get("support_skills", []) if str(item).strip()],
            "action": str(parsed.get("action", "")).strip(),
            "query": str(parsed.get("query", "")).strip(),
            "checkpoint": parsed.get("checkpoint") or {},
            "draft_answer": str(parsed.get("draft_answer", "")).strip(),
            "rationale": clip_text(str(parsed.get("rationale", "")).strip(), 300),
            "confidence": parsed.get("confidence"),
        }
        if step["action"] in {"search", "verify"} and step["query"]:
            write_current_step(
                current_step_path,
                record=record,
                phase="retriever_call",
                step_index=step_index,
                completed_steps=steps,
                query=step["query"],
                action=step["action"],
                checkpoint=step["checkpoint"],
                note="waiting for retriever results",
            )
            latest_information = retriever.search(step["query"])
            step["retrieved"] = latest_information
            evidence_blocks.append(latest_information)
        else:
            step["retrieved"] = ""
        steps.append(step)
        if step["action"] == "answer":
            final_answer = step["draft_answer"]
            break
        time.sleep(args.sleep_seconds)

    merged_evidence = "\n\n".join(evidence_blocks[-3:])
    if steps:
        write_current_step(
            current_step_path,
            record=record,
            phase="finalizer_model_call",
            step_index=len(steps),
            completed_steps=steps,
            note="waiting for final answer normalization",
        )
        finalizer_system, finalizer_user = build_finalizer_prompts(
            record=record,
            trace_steps=steps,
            evidence_text=clip_text(merged_evidence, 12000),
            draft_answer=final_answer or steps[-1].get("draft_answer", ""),
        )
        response_json = call_responses_api(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=finalizer_system,
            user_prompt=finalizer_user,
            reasoning_effort=args.reasoning_effort,
            verbosity="low",
            temperature=0.0,
            max_output_tokens=200,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.api_max_retries,
            retry_backoff=args.api_retry_backoff,
        )
        output_text = extract_text(response_json)
        raw_outputs.append(output_text)
        parsed = extract_json_object(output_text)
        normalized_answer = str(parsed.get("final_answer", "")).strip()
        if normalized_answer:
            final_answer = normalized_answer
            steps.append(
                {
                    "step": len(steps),
                    "primary_skill": "verbatim-evidence-span",
                    "support_skills": ["answer-grounding-check"],
                    "action": "answer",
                    "query": "",
                    "checkpoint": {},
                    "draft_answer": final_answer,
                    "rationale": clip_text(str(parsed.get("reason", "")).strip(), 200),
                    "confidence": 1.0,
                    "retrieved": "",
                }
            )

    trajectory = {
        "id": record["id"],
        "dataset": record["dataset"],
        "sample_origin": record["sample_origin"],
        "question": record["question"],
        "gold_answers": record.get("gold_answers", []),
        "candidate_primary_skills": record.get("candidate_primary_skills", []),
        "metadata_summary": record.get("metadata_summary", {}),
        "failure_info": record.get("failure_info"),
        "steps": steps,
        "final_answer": final_answer,
        "raw_outputs": raw_outputs,
    }
    trajectory["validation"] = validate_trajectory(record, trajectory, legal_skill_ids)
    write_current_step(
        current_step_path,
        record=record,
        phase="record_complete",
        step_index=len(trajectory["steps"]),
        completed_steps=trajectory["steps"],
        action="answer",
        note=f"passed={trajectory['validation']['passed']}",
    )
    return trajectory


def build_initial_summary(
    *,
    manifest_path: Path,
    skill_bank_path: Path,
    model: str,
    base_url: str,
    total_requested: int,
    existing_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = {
        "manifest_path": str(manifest_path),
        "skill_bank_path": str(skill_bank_path),
        "model": model,
        "base_url": base_url,
        "total_requested": total_requested,
        "processed": 0,
        "passed": 0,
        "failed": 0,
        "runtime_errors": 0,
        "datasets": {},
    }
    for row in existing_rows:
        dataset = row.get("dataset", "unknown")
        summary["processed"] += 1
        summary["datasets"].setdefault(dataset, {"processed": 0, "passed": 0})
        summary["datasets"][dataset]["processed"] += 1
        if row.get("validation", {}).get("passed"):
            summary["passed"] += 1
            summary["datasets"][dataset]["passed"] += 1
        else:
            summary["failed"] += 1
    return summary


def build_runtime_failure(record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "sample_origin": record["sample_origin"],
        "question": record["question"],
        "gold_answers": record.get("gold_answers", []),
        "candidate_primary_skills": record.get("candidate_primary_skills", []),
        "metadata_summary": record.get("metadata_summary", {}),
        "failure_info": record.get("failure_info"),
        "steps": [],
        "final_answer": "",
        "raw_outputs": [],
        "error": {
            "type": type(exc).__name__,
            "message": clip_text(str(exc), 500),
        },
        "validation": {
            "em": False,
            "has_search": False,
            "unknown_skills": [],
            "support_only_primary_steps": [],
            "route_matches_candidates": False,
            "passed": False,
            "runtime_error": True,
        },
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in the environment.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "trajectories.raw.jsonl"
    filtered_path = output_dir / "trajectories.filtered.jsonl"
    failed_path = output_dir / "trajectories.failed.jsonl"
    summary_path = output_dir / "run_summary.json"
    current_record_path = output_dir / "current_record.json"
    current_step_path = output_dir / "current_step.json"

    processed_rows_by_id: Dict[str, Dict[str, Any]] = {}
    if args.resume and raw_path.exists():
        for row in load_jsonl(raw_path):
            row_id = row.get("id")
            if row_id:
                processed_rows_by_id[row_id] = row
    processed_ids = set(processed_rows_by_id)

    skill_bank_text = load_skill_bank(args.skill_bank_path)
    legal_skill_ids = load_skill_ids(args.skill_bank_path)
    retriever = RetrieverClient(args.retriever_host, args.retriever_port, args.retriever_topk, args.retriever_timeout)

    manifest_rows = list(load_jsonl(args.manifest_path))
    if args.max_examples > 0:
        manifest_rows = manifest_rows[: args.max_examples]

    summary = build_initial_summary(
        manifest_path=args.manifest_path,
        skill_bank_path=args.skill_bank_path,
        model=args.model,
        base_url=args.base_url,
        total_requested=len(manifest_rows),
        existing_rows=list(processed_rows_by_id.values()),
    )
    dump_json(summary_path, summary)

    for record in manifest_rows:
        if record["id"] in processed_ids:
            continue
        write_current_record(
            current_record_path,
            record,
            processed_count=len(processed_ids),
            total_requested=len(manifest_rows),
        )
        try:
            trajectory = run_one_example(
                record=record,
                skill_bank_text=skill_bank_text,
                legal_skill_ids=legal_skill_ids,
                retriever=retriever,
                api_key=api_key,
                args=args,
                current_step_path=current_step_path,
            )
        except Exception as exc:
            append_jsonl(failed_path, build_runtime_failure(record, exc))
            summary["runtime_errors"] += 1
            dump_json(summary_path, summary)
            write_current_step(
                current_step_path,
                record=record,
                phase="runtime_error",
                step_index=0,
                completed_steps=[],
                note=f"{type(exc).__name__}: {clip_text(str(exc), 300)}",
            )
            print(f"[warn] runtime failure on {record['id']}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.sleep_seconds)
            continue
        append_jsonl(raw_path, trajectory)
        processed_ids.add(record["id"])
        summary["processed"] += 1
        dataset = trajectory["dataset"]
        summary["datasets"].setdefault(dataset, {"processed": 0, "passed": 0})
        summary["datasets"][dataset]["processed"] += 1
        if trajectory["validation"]["passed"]:
            append_jsonl(filtered_path, trajectory)
            summary["passed"] += 1
            summary["datasets"][dataset]["passed"] += 1
        else:
            append_jsonl(failed_path, trajectory)
            summary["failed"] += 1
        dump_json(summary_path, summary)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
