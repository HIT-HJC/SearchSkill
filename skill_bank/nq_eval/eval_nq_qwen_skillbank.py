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

COMMON_HELPER_DIR = "outputs/qwen3_8b_hotpotqa_eval_20260323"
if COMMON_HELPER_DIR not in sys.path:
    sys.path.insert(0, COMMON_HELPER_DIR)

from hotpotqa_eval_common import (  # noqa: E402
    build_chat_prompt,
    build_summary,
    clean_prediction,
    dump_json,
    exact_match_multi,
    extract_answer,
    flush_file,
    generate_text,
    load_jsonl,
    load_model_and_tokenizer,
    setup_logger,
)


DEFAULT_MODEL_PATH = ""
DEFAULT_DATA_PATH = "eval/nq_b0_b1/data/nq_dev_sample100_seed42.jsonl"
DEFAULT_SKILL_BANK_PATH = "skill_bank/inputs/seed_skill_bank.md"

SYSTEM_PROMPT = (
    "You are participating in a retrieval tool-use evaluation. "
    "You do not have direct access to search results. "
    "Never fabricate or simulate an <information> block yourself. "
    "If you need retrieval, emit a <search>...</search> tag and stop immediately after the first </search>. "
    "Do not output more than one <search> tag in a single response. "
    "When you have enough evidence, emit the final answer inside <answer>...</answer>. "
    "The final answer must be the shortest exact answer span, not a sentence. "
    "You must emit <skill>...</skill> at the start of every turn to declare the skill you are using."
)

USER_PROMPT_TEMPLATE = (
    "Answer the given question. "
    "If you need retrieval, output exactly one <search>...</search> tag and stop immediately after </search>. "
    "The search results will be returned between <information> and </information>. "
    "If you have enough evidence, output exactly one <answer>...</answer> tag and stop immediately after </answer>. "
    "Do not describe searching in natural language. "
    "Do not output an <information> block by yourself. "
    "After each <information> block, start a new turn with <skill>...</skill> followed by exactly one <search>...</search> or <answer>...</answer>. "
    "Never output a standalone closing tag such as </search> or </answer>. "
    "The final answer must be the shortest exact answer span, not a full sentence. "
    "For yes/no questions, output exactly yes or no. "
    "For person/place/work names, output only the name. "
    "For years or numbers, output only the value. "
    "Do not add explanation, punctuation, markdown, or prefixes such as 'The answer is'. "
    "Available skills:\n{skill_bank}\n"
    "Recommended skills for this question: {recommended_skills}\n"
    "At the start of every turn, emit <skill>chosen-skill-1|chosen-skill-2</skill> before your <search> or <answer>. "
    "Question: {question}\n"
)

FOLLOWUP_USER_TEMPLATE = (
    "<information>{search_results}</information>\n\n"
    "Continue the same question. "
    "Start a new turn with <skill>...</skill>. "
    "If you still need outside evidence, output exactly one <search>...</search>. "
    "Otherwise, output the final answer inside <answer>...</answer> with no extra explanation. "
    "Never output a standalone closing tag."
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
DERIVED_KINSHIP_CUE_RE = re.compile(r"\b(maternal|paternal|grandfather|grandmother|great grandfather|great grandmother|mother-in-law|father-in-law|in-law)\b")
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


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_skill_bank_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def parse_skill_ids(skill_bank_text: str) -> List[str]:
    return dedupe_keep_order(SKILL_ID_RE.findall(skill_bank_text))


def get_gold_answers(example: Dict[str, Any]) -> List[str]:
    raw = example.get("golden_answers")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if values:
            return values
    raw = example.get("answer", "")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    value = str(raw).strip()
    return [value] if value else []


def extract_skills(text: str) -> List[str]:
    skills: List[str] = []
    for raw in SKILL_RE.findall(text):
        for part in raw.split("|"):
            skill = part.strip()
            if skill:
                skills.append(skill)
    return dedupe_keep_order(skills)


def question_type(question: str) -> str:
    q = question.lower()
    if q.startswith(("are ", "is ", "was ", "were ", "do ", "does ", "did ", "has ", "have ", "had ", "can ", "could ")):
        return "yesno"
    if COMPARISON_CUE_RE.search(q):
        return "compare"
    if " or " in q:
        return "forced_choice"
    if re.search(r"\b(first|largest|smallest|highest|lowest|oldest|youngest|top)\b", q):
        return "ranking"
    if re.search(r"\b(real name|full name|nickname|alternate name|also known as|formerly|stage name)\b", q):
        return "alias"
    if re.search(r"\b(when|what year|date|how many|how much|population|duration|height|age)\b", q):
        return "time_num"
    if len(q.split()) >= 12 or re.search(r"\b(which|what|who)\b.*\b(of|in|with|from|for)\b", q):
        return "constraint_heavy"
    return "fact"


def relation_connector_count(question: str) -> int:
    return len(RELATION_CONNECTOR_RE.findall(question.lower()))


def has_explicit_relation_of(question: str) -> bool:
    return bool(RELATION_OF_CUE_RE.search(question.lower()))


def is_long_hop_question(question: str) -> bool:
    q = question.lower()
    connectors = relation_connector_count(q)
    relation_mentions = len(
        re.findall(
            r"\b(director|author|founder|creator|composer|performer|writer|actor|actress|producer|father|mother|spouse|wife|husband|country|city|state|county|school|university|burial|death|birth)\b",
            q,
        )
    )
    return connectors >= 4 or relation_mentions >= 3 or (len(q.split()) >= 18 and connectors >= 3)


def finalize_recommendations(selected: List[str], limit: int = 5) -> List[str]:
    return dedupe_keep_order(selected)[:limit]


def select_initial_skills(question: str, available_skill_ids: Sequence[str]) -> List[str]:
    q = question.lower()
    available = set(available_skill_ids)
    selected: List[str] = []

    def maybe_add(skill_id: str) -> None:
        if skill_id in available:
            selected.append(skill_id)

    qtype = question_type(question)
    long_hop = is_long_hop_question(question)
    explicit_relation = has_explicit_relation_of(question)
    has_comparison = bool(COMPARISON_CUE_RE.search(q))
    has_kinship = bool(KINSHIP_CUE_RE.search(q))
    has_derived_kinship = bool(DERIVED_KINSHIP_CUE_RE.search(q))
    has_temporal_anchor = bool(TEMPORAL_ANCHOR_CUE_RE.search(q))
    bridge_like = explicit_relation or bool(re.search(r"\b(which|who|what)\b.*\b(of|whose|from|in|with)\b", q))

    if qtype == "yesno":
        maybe_add("multihop-yes-no-verification")
    if has_comparison:
        if bridge_like:
            maybe_add("bridge-comparison-planning")
        maybe_add("parallel-attribute-compare")
    if has_derived_kinship:
        maybe_add("derived-kinship-inference-join")
    elif has_kinship and explicit_relation and long_hop:
        maybe_add("derived-kinship-inference-join")

    if explicit_relation:
        maybe_add("bridge-entity-search")
        maybe_add("relation-chain-decomposition")
    elif bridge_like:
        maybe_add("bridge-entity-search")

    if long_hop:
        maybe_add("sequential-hop-checkpointing")
        if has_temporal_anchor:
            maybe_add("temporal-anchor-carry-forward")
        maybe_add("re-anchored-long-hop-decomposition")
        maybe_add("reconstructed-chain-verification")

    if qtype == "alias":
        maybe_add("surface-name-resolution")
    if qtype == "ranking":
        maybe_add("superlative-ranking-match")
    if qtype in {"time_num", "ranking"} or has_temporal_anchor:
        maybe_add("temporal-range-extract")
    if qtype == "forced_choice" and not has_comparison:
        maybe_add("forced-choice-option-resolution")
    if qtype == "constraint_heavy":
        maybe_add("multi-constraint-query-anchoring")
    if not selected:
        maybe_add("single-entity-relation-lookup")
    if re.search(r"\b(company|organization|network|team|university|corporation)\b", q):
        maybe_add("conflict-check")

    maybe_add("answer-grounding-check")
    return finalize_recommendations(selected)


def extract_answer_from_output(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    _source, extracted = extract_answer(text)
    return extracted


def parse_generation_turn(text: str) -> Tuple[str, Optional[str], str]:
    stripped = text.strip()
    search_match = SEARCH_RE.search(stripped)
    answer_match = ANSWER_RE.search(stripped)
    if search_match and (not answer_match or search_match.start() < answer_match.start()):
        return "search", search_match.group(1).strip(), stripped[: search_match.end()].strip()
    if answer_match:
        return "final", None, stripped[: answer_match.end()].strip()
    return "final", None, stripped


def extract_prediction_from_trace(full_transcript: str, final_turn_output: str) -> str:
    matches = ANSWER_RE.findall(full_transcript)
    if matches:
        return matches[-1].strip()
    final_lower = final_turn_output.strip().lower()
    if (
        (SEARCH_RE.search(final_turn_output) or "<search" in final_lower or "</search>" in final_lower or final_lower in {"/search", "search"})
        and not ANSWER_RE.search(final_turn_output)
    ):
        return ""
    return extract_answer_from_output(final_turn_output)


def build_initial_prompt(
    question: str,
    skill_bank_text: str,
    available_skill_ids: Sequence[str],
    *,
    recommend_skills: bool,
) -> str:
    value = question.strip()
    if value and not value.endswith("?"):
        value += "?"
    recommended_skills = "(none)"
    if recommend_skills:
        recommended_skills = ", ".join(select_initial_skills(value, available_skill_ids)) or "(none)"
    return USER_PROMPT_TEMPLATE.format(
        question=value,
        skill_bank=skill_bank_text,
        recommended_skills=recommended_skills,
    )


def load_eval_model_and_tokenizer(
    *,
    model_path: str,
    adapter_path: Optional[str],
    dtype_name: str,
    trust_remote_code: bool,
    logger,
) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    tokenizer, model = load_model_and_tokenizer(
        model_path,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
        logger=logger,
    )
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a LoRA adapter during evaluation.") from exc

        adapter_tokenizer_path = Path(adapter_path)
        if (adapter_tokenizer_path / "tokenizer_config.json").exists():
            logger.info("Reloading tokenizer from adapter path %s", adapter_path)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                adapter_path,
                trust_remote_code=trust_remote_code,
            )

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def evaluate_example(
    *,
    question: str,
    skill_bank_text: str,
    available_skill_ids: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    retriever: RetrieverClient,
    stopping_criteria: transformers.StoppingCriteriaList,
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, Any]]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_prompt(
                question,
                skill_bank_text,
                available_skill_ids,
                recommend_skills=args.recommend_skills,
            ),
        },
    ]
    trace_steps: List[Dict[str, Any]] = []
    assistant_transcript = ""
    final_turn_output = ""

    for step_idx in range(args.max_steps):
        prompt = build_chat_prompt(
            tokenizer,
            messages,
            enable_thinking=not args.disable_thinking,
        )
        raw_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stopping_criteria=stopping_criteria,
        )
        action, content, visible_output = parse_generation_turn(raw_output)
        selected_skills = extract_skills(raw_output)

        record: Dict[str, Any] = {
            "step": step_idx,
            "generated": visible_output,
            "query": None,
            "retrieved": None,
            "action": action,
            "selected_skills": selected_skills,
        }
        if raw_output.strip() != visible_output.strip():
            record["raw_generated"] = raw_output

        if action == "search":
            search_query = (content or "").strip()
            record["query"] = search_query
            if not search_query:
                record["action"] = "invalid_search"
                trace_steps.append(record)
                final_turn_output = visible_output
                assistant_transcript += visible_output
                break

            try:
                search_results = retriever.search(search_query)
                record["retrieved"] = search_results
            except Exception as exc:
                record["retrieved_error"] = str(exc)
                record["retrieved"] = ""
                trace_steps.append(record)
                final_turn_output = visible_output
                assistant_transcript += visible_output
                break

            trace_steps.append(record)
            messages.append({"role": "assistant", "content": visible_output})
            messages.append(
                {
                    "role": "user",
                    "content": FOLLOWUP_USER_TEMPLATE.format(search_results=record["retrieved"]),
                }
            )
            assistant_transcript += visible_output
            continue

        final_turn_output = visible_output
        record["draft_prediction"] = clean_prediction(extract_answer_from_output(visible_output))
        trace_steps.append(record)
        assistant_transcript += visible_output
        break
    draft_prediction = extract_prediction_from_trace(assistant_transcript, final_turn_output)
    prediction = clean_prediction(draft_prediction)
    return prediction, trace_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NQ with Qwen2.5-7B-Instruct under a specified skill bank.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--skill-bank-path", type=str, default=DEFAULT_SKILL_BANK_PATH)
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
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--recommend-skills",
        action="store_true",
        help="Enable rule-based recommended skill hints in the prompt. Leave disabled for clean SkillBank evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, "nq_qwen_skillbank_eval")

    start_time = time.time()
    dataset = load_jsonl(args.data_path, max_samples=args.max_samples)
    logger.info("Loaded %d examples from %s", len(dataset), args.data_path)

    skill_bank_text = load_skill_bank_text(args.skill_bank_path)
    available_skill_ids = parse_skill_ids(skill_bank_text)
    logger.info("Loaded %d skills from %s", len(available_skill_ids), args.skill_bank_path)

    tokenizer, model = load_eval_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        logger=logger,
    )
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, STOP_STRINGS)])
    retriever = RetrieverClient(args.retriever_host, args.retriever_port, args.retriever_topk, args.retriever_timeout)

    records: List[Dict[str, Any]] = []
    n_correct = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            question = str(example.get("question", "")).strip()
            gold_answers = get_gold_answers(example)

            logger.info("%s", "=" * 80)
            logger.info("[Example %d] Q: %s", idx, question)

            prediction, trace_steps = evaluate_example(
                question=question,
                skill_bank_text=skill_bank_text,
                available_skill_ids=available_skill_ids,
                tokenizer=tokenizer,
                model=model,
                retriever=retriever,
                stopping_criteria=stopping_criteria,
                args=args,
            )
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
                "skill_bank_path": args.skill_bank_path,
                "retriever_host": args.retriever_host,
                "retriever_port": args.retriever_port,
                "recommend_skills": args.recommend_skills,
            }
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            flush_file(fout)

            search_steps = sum(1 for step in trace_steps if step.get("action") == "search")
            logger.info("Prediction: %s | Gold: %s | EM=%d | search_steps=%d", prediction, gold_answers, em, search_steps)
            if (idx + 1) % args.print_every == 0:
                logger.info("Processed %d / %d | running EM = %.4f", idx + 1, len(dataset), n_correct / max(1, idx + 1))

    end_time = time.time()

    avg_search_steps = sum(sum(1 for step in record["steps"] if step.get("action") == "search") for record in records) / max(1, len(records))
    avg_total_steps = sum(len(record["steps"]) for record in records) / max(1, len(records))
    avg_selected_skills = sum(
        len({skill for step in record["steps"] for skill in step.get("selected_skills", [])})
        for record in records
    ) / max(1, len(records))
    tagged_final_turn_rate = sum(
        1
        for record in records
        if record["steps"] and "<answer>" in str(record["steps"][-1].get("generated", "")).lower()
    ) / max(1, len(records))
    total_search_turns = sum(1 for record in records for step in record["steps"] if step.get("action") == "search")
    tagged_search_turn_rate = sum(
        1
        for record in records
        for step in record["steps"]
        if step.get("action") == "search" and "<search>" in str(step.get("generated", "")).lower()
    ) / max(1, total_search_turns)

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
            "skill_bank_path": args.skill_bank_path,
            "retriever_host": args.retriever_host,
            "retriever_port": args.retriever_port,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_steps": args.max_steps,
            "dtype": args.dtype,
            "disable_thinking": args.disable_thinking,
            "avg_search_steps": avg_search_steps,
            "avg_total_steps": avg_total_steps,
            "avg_selected_skills": avg_selected_skills,
            "tagged_final_turn_rate": tagged_final_turn_rate,
            "tagged_search_turn_rate": tagged_search_turn_rate,
            "n_skills_in_bank": len(available_skill_ids),
        },
    )
    dump_json(args.summary_json, summary)
    logger.info("Finished evaluation. Final EM on %d examples: %.4f", len(records), summary["em"])


if __name__ == "__main__":
    main()
