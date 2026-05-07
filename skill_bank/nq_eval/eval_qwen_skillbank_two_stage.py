#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval_qwen_skillbank_v3 import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_SKILL_BANK_PATH,
    RetrieverClient,
    StopOnSequence,
    build_chat_prompt,
    build_summary,
    clean_prediction,
    dedupe_keep_order,
    dump_json,
    estimate_search_budget,
    exact_match_multi,
    exact_match_multi_robust,
    flush_file,
    format_skill_cards as _legacy_format_skill_cards,
    generate_text,
    get_gold_answers,
    load_eval_model_and_tokenizer,
    load_jsonl,
    normalize_query,
    parse_generation_turn,
    parse_skill_bank_entries,
    setup_logger,
)


SELECT_SKILL_RE = re.compile(r"<select_skill>(.*?)</select_skill>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a SearchSkill policy model using the final SkillBank in two phases. "
    "In the skill-selection phase, output only <select_skill>skill-id</select_skill> "
    "or <select_skill>skill-id|skill-id</select_skill>, then stop. "
    "After the selected skill cards are provided, output exactly "
    "<skill>the-same-skill-ids</skill> followed by exactly one action tag, either "
    "<search>query</search> or <answer>span</answer>, and stop immediately after "
    "the closing action tag. Do not output explanations, markdown, natural-language "
    "tool descriptions, or <information> by yourself. Answer as soon as the evidence "
    "is sufficient."
)

SELECTION_INSTRUCTION = (
    "Selection phase: choose 1 to 3 skill ids from the the final SkillBank index that should "
    "govern the next action. If the evidence is sufficient, choose a closure skill "
    "such as verbatim-evidence-span and/or answer-grounding-check. Output only "
    "<select_skill>skill-id</select_skill> or <select_skill>skill-id|skill-id</select_skill>. "
    "Do not search or answer in this turn."
)

ACTION_INSTRUCTION_TEMPLATE = (
    "Action phase. Read the selected the final SkillBank card(s) and follow them for the next action.\n"
    "Selected skill ids: {skill_ids}\n"
    "Now output exactly <skill>{skill_ids}</skill> followed by exactly one "
    "<search>...</search> or <answer>...</answer>. Use the same skill ids in the "
    "<skill> tag. Do not output <select_skill> in this turn. Stop immediately after "
    "the closing action tag."
)

INITIAL_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Suggested search budget: {search_budget}\n"
    "Easy questions usually finish in 2-3 searches; harder chain or comparison questions may need 4-5.\n"
    "For A-or-B comparison questions, keep the explicit options as anchors; the final answer should be one of those options.\n"
    "For bridge questions, do not answer with an intermediate entity copied from the question unless it is the requested final attribute.\n"
    "Do not repeat the same entity-attribute pair.\n"
    "If the answer span is already explicit in the evidence, select a closure skill and answer immediately.\n\n"
    "{skill_index}\n\n"
    "{selection_instruction}"
)

FOLLOWUP_PROMPT_TEMPLATE = (
    "<information>{search_results}</information>\n\n"
    "Continue the same question.\n"
    "Searches used: {searches_used}/{search_budget}\n"
    "Recent searches: {recent_searches}\n"
    "If the answer is now supported, select a closure skill and answer immediately.\n"
    "For A-or-B comparisons, keep checking the explicit options and do not introduce a third option as the answer.\n"
    "Otherwise make one targeted search for the remaining missing entity or attribute only.\n\n"
    "{skill_index}\n\n"
    "{selection_instruction}"
)

SELECT_STOP_STRINGS = (
    "</select_skill>",
    " </select_skill>",
    "</select_skill>\n",
    " </select_skill>\n",
    "<|im_end|>",
)

ACTION_STOP_STRINGS = (
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</answer>",
    " </answer>",
    "</answer>\n",
    " </answer>\n",
    "<|im_end|>",
)


def compact_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars].rstrip() + "..."
    return value


def format_skill_index(skill_bank_entries: Dict[str, str], *, max_desc_chars: int) -> str:
    lines = ["Available the final SkillBank index:"]
    for skill_id, description in skill_bank_entries.items():
        lines.append(f"- {skill_id}: {compact_text(description, max_desc_chars)}")
    return "\n".join(lines)


def format_selected_skill_cards(
    selected_skills: Sequence[str],
    skill_bank_entries: Dict[str, str],
    *,
    max_chars_per_skill: int,
) -> str:
    cards = _legacy_format_skill_cards(
        selected_skills,
        skill_bank_entries,
        max_chars_per_skill=max_chars_per_skill,
    )
    if not cards.strip():
        return "Selected the final SkillBank card(s):\n(none)"
    return "Selected the final SkillBank card(s):\n" + "\n".join(
        f'<skill_card id="{line.split(":", 1)[0][2:]}">\n{line.split(":", 1)[1].strip()}\n</skill_card>'
        for line in cards.splitlines()
        if line.startswith("- ") and ":" in line
    )


def parse_skill_selection_turn(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    match = SELECT_SKILL_RE.search(stripped)
    if match is None:
        return {
            "format_ok": False,
            "selected_skills": [],
            "visible_output": stripped,
            "reason": "missing_select_skill_tag",
        }
    selected_skills = dedupe_keep_order(part.strip() for part in match.group(1).split("|"))
    return {
        "format_ok": True,
        "selected_skills": selected_skills,
        "visible_output": stripped[: match.end()].strip(),
        "reason": "",
    }


def valid_selected_skills(selected_skills: Sequence[str], skill_bank_entries: Dict[str, str]) -> List[str]:
    available = set(skill_bank_entries)
    return [skill_id for skill_id in dedupe_keep_order(selected_skills) if skill_id in available][:3]


def build_initial_prompt(question: str, search_budget: int, skill_index: str) -> str:
    value = question.strip()
    if value and not value.endswith("?"):
        value += "?"
    return INITIAL_PROMPT_TEMPLATE.format(
        question=value,
        search_budget=search_budget,
        skill_index=skill_index,
        selection_instruction=SELECTION_INSTRUCTION,
    )


def build_followup_prompt(
    search_results: str,
    *,
    searches_used: int,
    search_budget: int,
    recent_searches: Sequence[str],
    skill_index: str,
) -> str:
    return FOLLOWUP_PROMPT_TEMPLATE.format(
        search_results=search_results,
        searches_used=searches_used,
        search_budget=search_budget,
        recent_searches=" | ".join(recent_searches[-3:]) or "(none)",
        skill_index=skill_index,
        selection_instruction=SELECTION_INSTRUCTION,
    )


def build_action_prompt(
    selected_skills: Sequence[str],
    skill_bank_entries: Dict[str, str],
    *,
    max_skill_card_chars: int,
) -> str:
    skill_ids_text = "|".join(selected_skills)
    return (
        f"{format_selected_skill_cards(selected_skills, skill_bank_entries, max_chars_per_skill=max_skill_card_chars)}\n\n"
        f"{ACTION_INSTRUCTION_TEMPLATE.format(skill_ids=skill_ids_text)}"
    )


def evaluate_example(
    *,
    question: str,
    dataset_tag: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    retriever: RetrieverClient,
    action_stopping_criteria: transformers.StoppingCriteriaList,
    select_stopping_criteria: transformers.StoppingCriteriaList,
    skill_bank_entries: Dict[str, str],
    skill_index: str,
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    search_budget = estimate_search_budget(question, dataset_tag)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(question, search_budget, skill_index)},
    ]
    trace_steps: List[Dict[str, Any]] = []
    recent_queries: List[str] = []
    seen_queries = set()
    searches_used = 0
    final_prediction = ""
    final_reason = "no_answer"

    max_turns = search_budget + 1
    for step_idx in range(max_turns):
        select_prompt = build_chat_prompt(
            tokenizer,
            messages,
            enable_thinking=not args.disable_thinking,
        )
        raw_select_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=select_prompt,
            max_new_tokens=args.max_select_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stopping_criteria=select_stopping_criteria,
        )
        parsed_select = parse_skill_selection_turn(raw_select_output)
        selected_for_context = valid_selected_skills(parsed_select["selected_skills"], skill_bank_entries)

        record: Dict[str, Any] = {
            "step": step_idx,
            "skill_selection": {
                "generated": parsed_select["visible_output"],
                "selected_skills": parsed_select["selected_skills"],
                "selected_skills_for_context": selected_for_context,
                "format_ok": parsed_select["format_ok"],
                "reason": parsed_select["reason"],
            },
            "action": None,
            "selected_skills": [],
            "format_ok": None,
            "query": None,
            "retrieved": None,
            "reason": "",
        }
        if raw_select_output.strip() != parsed_select["visible_output"].strip():
            record["skill_selection"]["raw_generated"] = raw_select_output

        if not parsed_select["format_ok"] or not selected_for_context:
            record["action"] = "invalid_skill_selection"
            record["reason"] = parsed_select["reason"] or "no_valid_skill_selected"
            trace_steps.append(record)
            final_reason = record["reason"]
            break

        select_message = {"role": "assistant", "content": parsed_select["visible_output"]}
        action_user_message = {
            "role": "user",
            "content": build_action_prompt(
                selected_for_context,
                skill_bank_entries,
                max_skill_card_chars=args.max_skill_card_chars,
            ),
        }
        action_messages = [*messages, select_message, action_user_message]
        action_prompt = build_chat_prompt(
            tokenizer,
            action_messages,
            enable_thinking=not args.disable_thinking,
        )
        raw_action_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=action_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stopping_criteria=action_stopping_criteria,
        )
        parsed_action = parse_generation_turn(raw_action_output)
        record.update(
            {
                "generated": parsed_action["visible_output"],
                "action": parsed_action["action"],
                "selected_skills": parsed_action["selected_skills"],
                "format_ok": parsed_action["format_ok"],
                "reason": parsed_action["reason"],
                "skill_context_ids": selected_for_context,
            }
        )
        if raw_action_output.strip() != parsed_action["visible_output"].strip():
            record["raw_generated"] = raw_action_output

        if parsed_action["action"] == "invalid":
            trace_steps.append(record)
            final_reason = parsed_action["reason"] or "invalid_action_turn"
            break

        messages.extend([select_message, action_user_message, {"role": "assistant", "content": parsed_action["visible_output"]}])

        if parsed_action["action"] == "final":
            final_prediction = clean_prediction(parsed_action["answer"])
            record["draft_prediction"] = final_prediction
            trace_steps.append(record)
            final_reason = "answered"
            break

        query = parsed_action["query"].strip()
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
            record["duplicate_query"] = True
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
        messages.append(
            {
                "role": "user",
                "content": build_followup_prompt(
                    search_results,
                    searches_used=searches_used,
                    search_budget=search_budget,
                    recent_searches=recent_queries,
                    skill_index=skill_index,
                ),
            }
        )

    extra = {
        "search_budget": search_budget,
        "searches_used": searches_used,
        "final_reason": final_reason,
        "skill_context_mode": "two_stage_skillbank",
    }
    return final_prediction, trace_steps, extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SearchSkill two-stage policy with explicit the final SkillBank two-stage protocol.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--skill-bank-path", type=str, default=DEFAULT_SKILL_BANK_PATH)
    parser.add_argument("--dataset-tag", type=str, required=True)
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
    parser.add_argument("--max-select-tokens", type=int, default=64)
    parser.add_argument("--max-skill-card-chars", type=int, default=900)
    parser.add_argument("--max-index-desc-chars", type=int, default=180)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--strict-em-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, "qwen_skillbank_two_stage_eval")

    start_time = time.time()
    dataset = load_jsonl(args.data_path, max_samples=args.max_samples)
    logger.info("Loaded %d examples from %s", len(dataset), args.data_path)

    skill_bank_text = Path(args.skill_bank_path).read_text(encoding="utf-8").strip()
    skill_bank_entries = parse_skill_bank_entries(skill_bank_text)
    if not skill_bank_entries:
        raise RuntimeError(f"No skill entries parsed from {args.skill_bank_path}")
    skill_index = format_skill_index(skill_bank_entries, max_desc_chars=args.max_index_desc_chars)
    logger.info("Loaded %d skill entries from %s", len(skill_bank_entries), args.skill_bank_path)

    tokenizer, model = load_eval_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        logger=logger,
    )
    action_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, ACTION_STOP_STRINGS)])
    select_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, SELECT_STOP_STRINGS)])
    retriever = RetrieverClient(args.retriever_host, args.retriever_port, args.retriever_topk, args.retriever_timeout)

    records: List[Dict[str, Any]] = []
    n_correct = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            question = str(example.get("question", "")).strip()
            gold_answers = get_gold_answers(example)
            prediction, trace_steps, extra = evaluate_example(
                question=question,
                dataset_tag=args.dataset_tag,
                tokenizer=tokenizer,
                model=model,
                retriever=retriever,
                action_stopping_criteria=action_stopping_criteria,
                select_stopping_criteria=select_stopping_criteria,
                skill_bank_entries=skill_bank_entries,
                skill_index=skill_index,
                args=args,
            )
            em = exact_match_multi(prediction, gold_answers) if args.strict_em_only else exact_match_multi_robust(prediction, gold_answers)
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
                "dataset_tag": args.dataset_tag,
                **extra,
            }
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            flush_file(fout)

            if (idx + 1) % args.print_every == 0:
                logger.info("Processed %d / %d | running EM = %.4f", idx + 1, len(dataset), n_correct / max(1, idx + 1))

    end_time = time.time()

    total_search_steps = sum(sum(1 for step in record["steps"] if step.get("action") == "search") for record in records)
    total_turns = sum(len(record["steps"]) for record in records)
    action_format_ok_turns = sum(1 for record in records for step in record["steps"] if step.get("format_ok") is True)
    selection_ok_turns = sum(
        1
        for record in records
        for step in record["steps"]
        if (step.get("skill_selection") or {}).get("format_ok") is True
    )
    final_action_rate = sum(
        1
        for record in records
        if record["steps"] and record["steps"][-1].get("action") == "final"
    ) / max(1, len(records))
    duplicate_search_count = sum(
        1
        for record in records
        for step in record["steps"]
        if step.get("duplicate_query")
    )
    empty_count = sum(1 for record in records if not str(record.get("prediction", "")).strip())
    invalid_selection_count = sum(
        1
        for record in records
        for step in record["steps"]
        if step.get("action") == "invalid_skill_selection"
    )

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
            "dataset_tag": args.dataset_tag,
            "retriever_host": args.retriever_host,
            "retriever_port": args.retriever_port,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "disable_thinking": args.disable_thinking,
            "avg_search_steps": total_search_steps / max(1, len(records)),
            "action_format_ok_rate": action_format_ok_turns / max(1, total_turns),
            "skill_selection_ok_rate": selection_ok_turns / max(1, total_turns),
            "final_action_rate": final_action_rate,
            "duplicate_search_count": duplicate_search_count,
            "empty_count": empty_count,
            "invalid_skill_selection_count": invalid_selection_count,
            "n_skills_in_bank": len(skill_bank_entries),
            "answer_guard_enabled": False,
            "strict_em_only": args.strict_em_only,
            "skill_context_mode": "two_stage_skillbank",
            "max_select_tokens": args.max_select_tokens,
            "max_skill_card_chars": args.max_skill_card_chars,
            "max_index_desc_chars": args.max_index_desc_chars,
        },
    )
    dump_json(args.summary_json, summary)
    logger.info("Finished evaluation. Final EM on %d examples: %.4f", len(records), summary["em"])


if __name__ == "__main__":
    main()
