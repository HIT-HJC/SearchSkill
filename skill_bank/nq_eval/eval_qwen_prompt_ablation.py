#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import torch
import transformers

COMMON_HELPER_DIR = "/online1/ycsc_chenkh/hitici_11/SearchSkill/qwen3_8b_hotpotqa_eval_20260323"
if COMMON_HELPER_DIR not in sys.path:
    sys.path.insert(0, COMMON_HELPER_DIR)

from hotpotqa_eval_common import (  # noqa: E402
    build_chat_prompt,
    build_summary,
    clean_prediction,
    dump_json,
    exact_match_multi,
    flush_file,
    generate_text,
    load_jsonl,
    load_model_and_tokenizer,
    setup_logger,
)


DEFAULT_MODEL_PATH = "/path/to/hf_models/Qwen2.5-7B-Instruct"
DEFAULT_SKILL_BANK_PATH = "/online1/ycsc_chenkh/hitici_11/HJCproject/SearchSkill Code/skill_bank/round_4_musique/outputs/final_skill_bank.md"

SYSTEM_PROMPT_WITH_SKILLS = (
    "You are participating in a search-and-answer evaluation with access to SkillBank hints. "
    "Every assistant turn must emit exactly one action tag, either <search>query</search> "
    "or <answer>span</answer>, and stop immediately after the closing tag. "
    "Use the provided SkillBank hints to plan better searches, but do not output <skill> tags. "
    "Do not output explanations, markdown, or any tag other than <search> or <answer>. "
    "Do not output <information> by yourself. "
    "Answer as soon as the evidence is sufficient."
)

USER_PROMPT_TEMPLATE_WITH_SKILLS = (
    "Question: {question}\n"
    "Relevant the final SkillBank hints:\n{skill_hints}\n"
    "Suggested search budget: {search_budget}\n"
    "Easy questions usually finish in 2-3 searches; harder chain or comparison questions may need 4-5.\n"
    "Do not repeat the same entity-attribute pair.\n"
    "If the answer span is already explicit in the evidence, answer immediately.\n"
    "Do not use <skill> tags.\n"
    "Output exactly one <search>...</search> or <answer>...</answer>."
)

FOLLOWUP_USER_TEMPLATE_WITH_SKILLS = (
    "<information>{search_results}</information>\n\n"
    "Continue the same question.\n"
    "Remember these SkillBank hints: {recommended_skills}\n"
    "Searches used: {searches_used}/{search_budget}\n"
    "Recent searches: {recent_searches}\n"
    "If the answer is now supported, answer immediately.\n"
    "Otherwise make one targeted search for the remaining missing entity or attribute only.\n"
    "Do not use <skill> tags.\n"
    "Output exactly one <search>...</search> or <answer>...</answer>.\n"
    "Stop immediately after the closing tag."
)

SYSTEM_PROMPT_WITHOUT_SKILLS = (
    "You are participating in a search-and-answer evaluation. "
    "Every assistant turn must emit exactly one action tag, either <search>query</search> "
    "or <answer>span</answer>, and stop immediately after the closing tag. "
    "Do not output explanations, markdown, or any tag other than <search> or <answer>. "
    "Do not output <information> by yourself. "
    "Answer as soon as the evidence is sufficient."
)

USER_PROMPT_TEMPLATE_WITHOUT_SKILLS = (
    "Question: {question}\n"
    "Suggested search budget: {search_budget}\n"
    "Easy questions usually finish in 2-3 searches; harder chain or comparison questions may need 4-5.\n"
    "Do not repeat the same entity-attribute pair.\n"
    "If the answer span is already explicit in the evidence, answer immediately.\n"
    "Do not use <skill> tags.\n"
    "Output exactly one <search>...</search> or <answer>...</answer>."
)

FOLLOWUP_USER_TEMPLATE_WITHOUT_SKILLS = (
    "<information>{search_results}</information>\n\n"
    "Continue the same question.\n"
    "Searches used: {searches_used}/{search_budget}\n"
    "Recent searches: {recent_searches}\n"
    "If the answer is now supported, answer immediately.\n"
    "Otherwise make one targeted search for the remaining missing entity or attribute only.\n"
    "Do not use <skill> tags.\n"
    "Output exactly one <search>...</search> or <answer>...</answer>.\n"
    "Stop immediately after the closing tag."
)

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
SKILL_ID_RE = re.compile(r"`([a-z0-9][a-z0-9\-]*)`")

STOP_STRINGS = (
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
    "</answer>",
    " </answer>",
    "</answer>\n",
    " </answer>\n",
    "</answer>\n\n",
    " </answer>\n\n",
    "<|im_end|>",
)

COMPARISON_CUE_RE = re.compile(r"\b(compare|same|both|older|younger|earlier|later|more|less|higher|lower|born first|released first|came out first)\b")
KINSHIP_CUE_RE = re.compile(r"\b(mother|father|spouse|wife|husband|daughter|son|grandfather|grandmother|maternal|paternal|in-law)\b")
RELATION_OF_CUE_RE = re.compile(r"\b(director|author|founder|creator|composer|performer|writer|actor|actress|producer|father|mother|spouse|wife|husband|place of birth|date of birth|nationality|alma mater|school|university|burial|headquarters|country|city|state|county)\b.*\bof\b")
RELATION_CONNECTOR_RE = re.compile(r"\b(of|whose|where|that|after|before|during|while)\b")
TEMPORAL_ANCHOR_CUE_RE = re.compile(r"\b(before|after|during|when|year|date|season|last|next|former|current)\b")


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase, sequences: Sequence[str]):
        super().__init__()
        encoded = [tokenizer.encode(seq, add_special_tokens=False) for seq in sequences]
        self.target_ids = [torch.as_tensor(item, dtype=torch.long) for item in encoded if item]
        self.target_lengths = [len(item) for item in self.target_ids]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        if not self.target_ids:
            return False
        current = input_ids[0]
        if current.shape[0] < min(self.target_lengths):
            return False
        for target, target_length in zip(self.target_ids, self.target_lengths):
            target = target.to(current.device)
            if torch.equal(current[-target_length:], target):
                return True
        return False


class RetrieverClient:
    def __init__(self, retriever_host: str, retriever_port: int, topk: int, timeout: int):
        self.retriever_url = f"http://{retriever_host}:{retriever_port}/retrieve"
        self.topk = topk
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def search(self, query: str) -> str:
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True,
        }
        response = self.session.post(self.retriever_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        results = response.json()["result"][0]
        passages: List[str] = []
        for idx, doc_item in enumerate(results, start=1):
            content = doc_item["document"]["contents"]
            parts = content.split("\n")
            title = parts[0]
            text = "\n".join(parts[1:])
            passages.append(f"Doc {idx}(Title: {title}) {text}")
        return "\n".join(passages)


def dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_skill_ids(skill_bank_text: str) -> List[str]:
    return dedupe_keep_order(SKILL_ID_RE.findall(skill_bank_text))


def parse_skill_entries(skill_bank_text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    lines = [line.rstrip() for line in skill_bank_text.splitlines()]
    for idx, line in enumerate(lines):
        match = SKILL_ID_RE.fullmatch(line.strip())
        if not match:
            continue
        skill_id = match.group(1)
        description = ""
        next_idx = idx + 1
        while next_idx < len(lines):
            candidate = lines[next_idx].strip()
            if not candidate:
                next_idx += 1
                continue
            if SKILL_ID_RE.fullmatch(candidate):
                break
            description = candidate
            break
        entries[skill_id] = description
    return entries


def load_skill_bank_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def get_gold_answers(example: Dict[str, Any]) -> List[str]:
    raw = example.get("golden_answers")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if values:
            return values
    raw = example.get("answer", "")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if values:
            return values
    value = str(raw).strip()
    return [value] if value else []


def relation_connector_count(question: str) -> int:
    return len(RELATION_CONNECTOR_RE.findall(question.lower()))


def question_complexity(question: str) -> int:
    q = question.lower()
    score = 0
    if COMPARISON_CUE_RE.search(q):
        score += 1
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ")):
        score += 1
    if KINSHIP_CUE_RE.search(q):
        score += 1
    if RELATION_OF_CUE_RE.search(q):
        score += 1
    if TEMPORAL_ANCHOR_CUE_RE.search(q):
        score += 1
    if " or " in q:
        score += 1
    if relation_connector_count(q) >= 3:
        score += 1
    if len(q.split()) >= 18:
        score += 1
    return score


def estimate_search_budget(question: str, dataset_tag: str) -> int:
    score = question_complexity(question)
    if dataset_tag in {"nq", "triviaqa"}:
        return 2 if score == 0 else 3
    if score <= 1:
        return 3
    if score <= 3:
        return 4
    return 5


def select_recommended_skills(question: str, available_skill_ids: Sequence[str]) -> List[str]:
    q = question.lower()
    available = set(available_skill_ids)
    selected: List[str] = []

    def maybe_add(skill_id: str) -> None:
        if skill_id in available:
            selected.append(skill_id)

    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ")):
        maybe_add("multihop-yes-no-verification")
    if COMPARISON_CUE_RE.search(q):
        maybe_add("bridge-comparison-planning")
        maybe_add("parallel-attribute-compare")
    if KINSHIP_CUE_RE.search(q):
        maybe_add("derived-kinship-inference-join")
    if RELATION_OF_CUE_RE.search(q):
        maybe_add("bridge-entity-search")
        maybe_add("relation-chain-decomposition")
    elif relation_connector_count(q) >= 2:
        maybe_add("bridge-entity-search")
    if TEMPORAL_ANCHOR_CUE_RE.search(q):
        maybe_add("temporal-range-extract")
    if " or " in q and not COMPARISON_CUE_RE.search(q):
        maybe_add("forced-choice-option-resolution")
    if question_complexity(question) >= 3:
        maybe_add("sequential-hop-checkpointing")
        maybe_add("multi-constraint-query-anchoring")
    if not selected:
        maybe_add("single-entity-relation-lookup")
    maybe_add("answer-grounding-check")
    maybe_add("verbatim-evidence-span")
    return dedupe_keep_order(selected)[:5]


def normalize_query(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return " ".join(value.split())


def render_skill_hints(recommended_skills: Sequence[str], skill_entries: Dict[str, str]) -> str:
    lines: List[str] = []
    for skill_id in recommended_skills:
        description = skill_entries.get(skill_id, "").strip()
        if description:
            lines.append(f"- {skill_id}: {description}")
        else:
            lines.append(f"- {skill_id}")
    return "\n".join(lines) if lines else "(none)"


def parse_generation_turn(text: str, prompt_mode: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    skill_match = SKILL_RE.search(stripped)
    search_match = SEARCH_RE.search(stripped)
    answer_match = ANSWER_RE.search(stripped)

    selected_skills = []
    if skill_match is not None:
        selected_skills = dedupe_keep_order(part.strip() for part in skill_match.group(1).split("|"))

    action_match = None
    action_type = "invalid"
    if search_match and answer_match:
        if search_match.start() < answer_match.start():
            action_match = search_match
            action_type = "search"
        else:
            action_match = answer_match
            action_type = "final"
    elif search_match:
        action_match = search_match
        action_type = "search"
    elif answer_match:
        action_match = answer_match
        action_type = "final"

    if action_match is None:
        return {"action": "invalid", "format_ok": False, "selected_skills": selected_skills, "visible_output": stripped, "query": "", "answer": "", "reason": "missing_action_tag"}

    visible_output = stripped[: action_match.end()].strip()
    if action_type == "search":
        return {"action": "search", "format_ok": True, "selected_skills": selected_skills, "visible_output": visible_output, "query": action_match.group(1).strip(), "answer": "", "reason": ""}
    return {"action": "final", "format_ok": True, "selected_skills": selected_skills, "visible_output": visible_output, "query": "", "answer": action_match.group(1).strip(), "reason": ""}


def build_initial_prompt(
    question: str,
    recommended_skills: Sequence[str],
    skill_hints: str,
    search_budget: int,
    prompt_mode: str,
) -> str:
    value = question.strip()
    if value and not value.endswith("?"):
        value += "?"
    if prompt_mode == "with_skills":
        return USER_PROMPT_TEMPLATE_WITH_SKILLS.format(
            question=value,
            recommended_skills=", ".join(recommended_skills) or "(none)",
            skill_hints=skill_hints,
            search_budget=search_budget,
        )
    return USER_PROMPT_TEMPLATE_WITHOUT_SKILLS.format(question=value, search_budget=search_budget)


def build_followup_prompt(
    search_results: str,
    *,
    searches_used: int,
    search_budget: int,
    recent_searches: Sequence[str],
    recommended_skills: Sequence[str],
    prompt_mode: str,
) -> str:
    common = {
        "search_results": search_results,
        "searches_used": searches_used,
        "search_budget": search_budget,
        "recent_searches": " | ".join(recent_searches[-3:]) or "(none)",
        "recommended_skills": ", ".join(recommended_skills) or "(none)",
    }
    if prompt_mode == "with_skills":
        return FOLLOWUP_USER_TEMPLATE_WITH_SKILLS.format(**common)
    return FOLLOWUP_USER_TEMPLATE_WITHOUT_SKILLS.format(**common)


def load_eval_model_and_tokenizer(*, model_path: str, adapter_path: Optional[str], dtype_name: str, trust_remote_code: bool, logger) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    tokenizer, model = load_model_and_tokenizer(model_path, dtype_name=dtype_name, trust_remote_code=trust_remote_code, logger=logger)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a LoRA adapter during evaluation.") from exc

        adapter_tokenizer_path = Path(adapter_path)
        if (adapter_tokenizer_path / "tokenizer_config.json").exists():
            logger.info("Reloading tokenizer from adapter path %s", adapter_path)
            tokenizer = transformers.AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def evaluate_example(*, question: str, dataset_tag: str, recommended_skills: Sequence[str], skill_hints: str, tokenizer: transformers.PreTrainedTokenizerBase, model: transformers.PreTrainedModel, retriever: RetrieverClient, stopping_criteria: transformers.StoppingCriteriaList, args: argparse.Namespace) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    search_budget = estimate_search_budget(question, dataset_tag)
    system_prompt = SYSTEM_PROMPT_WITH_SKILLS if args.prompt_mode == "with_skills" else SYSTEM_PROMPT_WITHOUT_SKILLS
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_initial_prompt(question, recommended_skills, skill_hints, search_budget, args.prompt_mode)},
    ]
    trace_steps: List[Dict[str, Any]] = []
    recent_queries: List[str] = []
    seen_queries = set()
    searches_used = 0
    final_prediction = ""
    final_reason = "no_answer"

    max_turns = search_budget + 1
    for step_idx in range(max_turns):
        prompt = build_chat_prompt(tokenizer, messages, enable_thinking=not args.disable_thinking)
        raw_output = generate_text(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, stopping_criteria=stopping_criteria)
        parsed = parse_generation_turn(raw_output, args.prompt_mode)
        record: Dict[str, Any] = {"step": step_idx, "generated": parsed["visible_output"], "action": parsed["action"], "selected_skills": parsed["selected_skills"], "format_ok": parsed["format_ok"], "query": None, "retrieved": None, "reason": parsed["reason"]}
        if raw_output.strip() != parsed["visible_output"].strip():
            record["raw_generated"] = raw_output

        if parsed["action"] == "invalid":
            trace_steps.append(record)
            final_reason = parsed["reason"] or "invalid_turn"
            break

        if parsed["action"] == "final":
            final_prediction = clean_prediction(parsed["answer"])
            record["draft_prediction"] = final_prediction
            trace_steps.append(record)
            final_reason = "answered"
            break

        query = parsed["query"].strip()
        record["query"] = query
        normalized_query = normalize_query(query)
        if not normalized_query:
            record["action"] = "invalid_search"
            trace_steps.append(record)
            final_reason = "empty_search"
            break
        if searches_used >= search_budget:
            record["action"] = "budget_exhausted_search"
            trace_steps.append(record)
            final_reason = "budget_exhausted_search"
            break

        if normalized_query in seen_queries:
            search_results = "Duplicate search target detected. Answer now or search for a different missing entity or attribute."
            record["duplicate_query"] = True
        else:
            try:
                search_results = retriever.search(query)
            except Exception as exc:
                record["action"] = "retriever_error"
                record["retrieved_error"] = str(exc)
                trace_steps.append(record)
                final_reason = "retriever_error"
                break
            seen_queries.add(normalized_query)

        searches_used += 1
        recent_queries.append(query)
        record["retrieved"] = search_results
        trace_steps.append(record)
        messages.append({"role": "assistant", "content": parsed["visible_output"]})
        messages.append(
            {
                "role": "user",
                "content": build_followup_prompt(
                    search_results,
                    searches_used=searches_used,
                    search_budget=search_budget,
                    recent_searches=recent_queries,
                    recommended_skills=recommended_skills,
                    prompt_mode=args.prompt_mode,
                ),
            }
        )

    extra = {"search_budget": search_budget, "searches_used": searches_used, "final_reason": final_reason}
    return final_prediction, trace_steps, extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate instruct models with or without explicit SkillBank prompting.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--skill-bank-path", type=str, default=DEFAULT_SKILL_BANK_PATH)
    parser.add_argument("--dataset-tag", type=str, required=True)
    parser.add_argument("--prompt-mode", choices=("with_skills", "without_skills"), default="with_skills")
    parser.add_argument("--retriever-host", type=str, default="127.0.0.1")
    parser.add_argument("--retriever-port", type=int, default=8000)
    parser.add_argument("--retriever-topk", type=int, default=3)
    parser.add_argument("--retriever-timeout", type=int, default=60)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, "qwen_prompt_ablation_eval")

    start_time = time.time()
    dataset = load_jsonl(args.data_path, max_samples=args.max_samples)
    logger.info("Loaded %d examples from %s", len(dataset), args.data_path)
    logger.info("Prompt mode: %s", args.prompt_mode)

    available_skill_ids: List[str] = []
    skill_entries: Dict[str, str] = {}
    effective_skill_bank_path: Optional[str] = None
    if args.prompt_mode == "with_skills":
        skill_bank_text = load_skill_bank_text(args.skill_bank_path)
        available_skill_ids = parse_skill_ids(skill_bank_text)
        skill_entries = parse_skill_entries(skill_bank_text)
        effective_skill_bank_path = args.skill_bank_path
        logger.info("Loaded %d skill ids from %s", len(available_skill_ids), args.skill_bank_path)
    else:
        logger.info("Running without explicit skill prompts.")

    tokenizer, model = load_eval_model_and_tokenizer(model_path=args.model_path, adapter_path=args.adapter_path, dtype_name=args.dtype, trust_remote_code=args.trust_remote_code, logger=logger)
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, STOP_STRINGS)])
    retriever = RetrieverClient(args.retriever_host, args.retriever_port, args.retriever_topk, args.retriever_timeout)

    records: List[Dict[str, Any]] = []
    n_correct = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            question = str(example.get("question", "")).strip()
            gold_answers = get_gold_answers(example)
            recommended_skills = select_recommended_skills(question, available_skill_ids) if args.prompt_mode == "with_skills" else []
            skill_hints = render_skill_hints(recommended_skills, skill_entries) if args.prompt_mode == "with_skills" else "(none)"
            prediction, trace_steps, extra = evaluate_example(question=question, dataset_tag=args.dataset_tag, recommended_skills=recommended_skills, skill_hints=skill_hints, tokenizer=tokenizer, model=model, retriever=retriever, stopping_criteria=stopping_criteria, args=args)
            em = exact_match_multi(prediction, gold_answers)
            n_correct += em

            record = {
                "idx": idx,
                "id": example.get("id"),
                "question": question,
                "gold": gold_answers,
                "prediction": prediction,
                "em": em,
                "steps": trace_steps,
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "skill_bank_path": effective_skill_bank_path,
                "dataset_tag": args.dataset_tag,
                "prompt_mode": args.prompt_mode,
                **extra,
            }
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            flush_file(fout)

            if (idx + 1) % args.print_every == 0:
                logger.info("Processed %d / %d | running EM = %.4f", idx + 1, len(dataset), n_correct / max(1, idx + 1))

    end_time = time.time()

    total_search_steps = sum(sum(1 for step in record["steps"] if step.get("action") == "search") for record in records)
    format_ok_turns = sum(1 for record in records for step in record["steps"] if step.get("format_ok") is True)
    total_turns = sum(len(record["steps"]) for record in records)
    final_action_rate = sum(1 for record in records if record["steps"] and record["steps"][-1].get("action") == "final") / max(1, len(records))
    duplicate_search_count = sum(1 for record in records for step in record["steps"] if step.get("duplicate_query"))
    empty_count = sum(1 for record in records if not str(record.get("prediction", "")).strip())

    dump_json(args.out_json, records)
    summary = build_summary(
        model_path=args.model_path,
        data_path=args.data_path,
        out_jsonl=args.out_jsonl,
        log_file=args.log_file,
        n_examples=len(records),
        n_correct=n_correct,
        start_time=start_time,
        end_time=end_time,
        extra={
            "summary_json": args.summary_json,
            "out_json": args.out_json,
            "adapter_path": args.adapter_path,
            "skill_bank_path": effective_skill_bank_path,
            "dataset_tag": args.dataset_tag,
            "prompt_mode": args.prompt_mode,
            "retriever_host": args.retriever_host,
            "retriever_port": args.retriever_port,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "disable_thinking": args.disable_thinking,
            "avg_search_steps": total_search_steps / max(1, len(records)),
            "format_ok_rate": format_ok_turns / max(1, total_turns),
            "final_action_rate": final_action_rate,
            "duplicate_search_count": duplicate_search_count,
            "empty_count": empty_count,
            "n_skills_in_bank": len(available_skill_ids),
        },
    )
    dump_json(args.summary_json, summary)
    logger.info("Finished evaluation. Final EM on %d examples: %.4f", len(records), summary["em"])


if __name__ == "__main__":
    main()
