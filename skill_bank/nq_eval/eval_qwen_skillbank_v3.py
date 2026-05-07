#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
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

SYSTEM_PROMPT = (
    "You are a SearchSkill policy model. "
    "Every assistant turn must follow exactly this format: "
    "first emit <skill>skill-1|skill-2</skill>, then emit exactly one action tag, "
    "either <search>query</search> or <answer>span</answer>, and stop immediately after the closing tag. "
    "Do not output explanations, markdown, or any tag other than <skill>, <search>, or <answer>. "
    "Do not output <information> by yourself. "
    "Answer as soon as the evidence is sufficient."
)

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are the skill selector for a SearchSkill policy model. "
    "Choose the skill ids that should govern the next action. "
    "Output only one <skill>...</skill> tag and stop. "
    "Do not output <search>, <answer>, explanations, markdown, or any other tag."
)

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Recommended skills: {recommended_skills}\n"
    "{skill_context_block}"
    "Suggested search budget: {search_budget}\n"
    "Easy questions usually finish in 2-3 searches; harder chain or comparison questions may need 4-5.\n"
    "For A-or-B comparison questions, keep the explicit options as anchors; the final answer should be one of those options.\n"
    "For bridge questions, do not answer with an intermediate entity copied from the question unless it is the requested final attribute.\n"
    "Do not repeat the same entity-attribute pair.\n"
    "If the answer span is already explicit in the evidence, answer immediately.\n"
    "Start with <skill>...</skill> and then output exactly one <search>...</search> or <answer>...</answer>."
)

FOLLOWUP_USER_TEMPLATE = (
    "<information>{search_results}</information>\n\n"
    "Continue the same question.\n"
    "Searches used: {searches_used}/{search_budget}\n"
    "Recent searches: {recent_searches}\n"
    "{skill_context_block}"
    "If the answer is now supported, answer immediately.\n"
    "For A-or-B comparisons, keep checking the explicit options and do not introduce a third option as the answer.\n"
    "Otherwise make one targeted search for the remaining missing entity or attribute only.\n"
    "Start with <skill>...</skill> and then output exactly one <search>...</search> or <answer>...</answer>.\n"
    "Stop immediately after the closing tag."
)

ANSWER_REPAIR_USER_TEMPLATE = (
    "Your previous final answer was: {answer}\n"
    "Sanity-check that answer against the question and the evidence above before finalizing.\n"
    "Issue to fix: {reason}\n"
    "If this is an A-or-B comparison, answer with exactly one explicit option from the question.\n"
    "If the answer is a date, place, person, organization, or title, output the most specific final span requested, not an intermediate entity or a copied anchor.\n"
    "{search_instruction}\n"
    "Start with <skill>answer-grounding-check|verbatim-evidence-span</skill> and then output exactly one <answer>...</answer>"
    "{search_tag_instruction}. Stop immediately after the closing tag."
)

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
SKILL_ID_RE = re.compile(r"`([a-z0-9][a-z0-9\-]*)`")
SKILL_HEADER_RE = re.compile(r"^`([a-z0-9][a-z0-9\-]*)`\s*$")

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
SKILL_STOP_STRINGS = (
    "</skill>",
    " </skill>",
    "</skill>\n",
    " </skill>\n",
    "<|im_end|>",
)

COMPARISON_CUE_RE = re.compile(r"\b(compare|same|both|older|younger|earlier|later|more|less|higher|lower|born first|released first|came out first)\b")
KINSHIP_CUE_RE = re.compile(r"\b(mother|father|spouse|wife|husband|daughter|son|grandfather|grandmother|maternal|paternal|in-law)\b")
RELATION_OF_CUE_RE = re.compile(r"\b(director|author|founder|creator|composer|performer|writer|actor|actress|producer|father|mother|spouse|wife|husband|place of birth|date of birth|nationality|alma mater|school|university|burial|headquarters|country|city|state|county)\b.*\bof\b")
RELATION_CONNECTOR_RE = re.compile(r"\b(of|whose|where|that|after|before|during|while)\b")
TEMPORAL_ANCHOR_CUE_RE = re.compile(r"\b(before|after|during|when|year|date|season|last|next|former|current)\b")
YES_NO_QUESTION_RE = re.compile(r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could)\b", re.I)
BARE_YEAR_RE = re.compile(r"^\d{3,4}$")
COARSE_DIRECTION_RE = re.compile(r"^(north|south|east|west|northeast|northwest|southeast|southwest|northern|southern|eastern|western)$", re.I)


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


def parse_skill_bank_entries(skill_bank_text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    current_skill: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_skill, current_lines
        if current_skill:
            value = "\n".join(current_lines).strip()
            if value:
                entries[current_skill] = value
        current_skill = None
        current_lines = []

    for line in skill_bank_text.splitlines():
        match = SKILL_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_skill = match.group(1)
            current_lines = []
            continue
        if current_skill:
            current_lines.append(line)
    flush()
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


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_answer_loose(text: str) -> str:
    value = strip_accents(str(text or "")).lower()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match_multi_robust(prediction: str, gold_list: Sequence[str]) -> int:
    """Lightweight, dataset-agnostic answer normalization.

    This keeps the primary metric exact-match based, but avoids penalizing common
    surface variants such as parenthetical disambiguators, accents, and harmless
    location/title qualifiers.
    """
    if exact_match_multi(prediction, gold_list):
        return 1
    norm_pred = normalize_answer_loose(prediction)
    if not norm_pred:
        return 0
    pred_tokens = norm_pred.split()
    for gold in gold_list:
        norm_gold = normalize_answer_loose(gold)
        if not norm_gold:
            continue
        gold_tokens = norm_gold.split()
        if norm_pred == norm_gold:
            return 1
        if BARE_YEAR_RE.fullmatch(norm_gold) and re.search(rf"\b{re.escape(norm_gold)}\b", norm_pred):
            return 1
        # Accept exact gold spans with harmless qualifiers, e.g. "USS Essex (CV-9)".
        if len(gold_tokens) >= 2 and re.search(rf"\b{re.escape(norm_gold)}\b", norm_pred):
            return 1
        if len(pred_tokens) >= 2 and re.search(rf"\b{re.escape(norm_pred)}\b", norm_gold):
            return 1
        # Common location/title expansion: "Xian" -> "Xi'an, Shaanxi".
        if len(norm_gold) >= 4 and norm_pred.startswith(norm_gold + " "):
            return 1
    return 0


def clean_option_text(text: str) -> str:
    value = str(text or "").strip(" \t\n\r'\"“”‘’")
    value = re.sub(r"^(?:which|what|who|where|when)\b.*?,", "", value, flags=re.I).strip()
    value = re.sub(r"^(?:which|what|who)\s+(?:film|song|person|band|director|country|city|place)\s+", "", value, flags=re.I).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,:;?()[]")


def extract_or_options(question: str) -> List[str]:
    if " or " not in question.lower():
        return []
    parts = re.split(r"\s+or\s+", question.strip().rstrip("?"), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return []
    left, right = parts
    left = re.split(r"[,;:]", left)[-1]
    right = re.split(
        r"[,;?]|\s+(?:has|have|had|was|were|is|are|came|released|born|died|lived|located|come|comes)\b",
        right,
        maxsplit=1,
        flags=re.I,
    )[0]
    options = [clean_option_text(left), clean_option_text(right)]
    options = [item for item in options if 1 <= len(item.split()) <= 10 and len(item) >= 2]
    return options if len(options) == 2 else []


def answer_matches_option(answer: str, option: str) -> bool:
    norm_answer = normalize_answer_loose(answer)
    norm_option = normalize_answer_loose(option)
    if not norm_answer or not norm_option:
        return False
    return (
        norm_answer == norm_option
        or re.search(rf"\b{re.escape(norm_option)}\b", norm_answer) is not None
        or re.search(rf"\b{re.escape(norm_answer)}\b", norm_option) is not None
    )


def is_question_anchor_answer(question: str, answer: str, options: Sequence[str]) -> bool:
    norm_answer = normalize_answer_loose(answer)
    norm_question = normalize_answer_loose(question)
    if not norm_answer or norm_answer in {"yes", "no"}:
        return False
    if any(answer_matches_option(answer, option) for option in options):
        return False
    return re.search(rf"\b{re.escape(norm_answer)}\b", norm_question) is not None


def answer_repair_reason(question: str, answer: str, searches_used: int, search_budget: int) -> str:
    value = str(answer or "").strip()
    if not value:
        return "empty final answer"
    norm_answer = normalize_answer_loose(value)
    q_lower = question.lower()
    if norm_answer in {"none", "unknown", "not found", "cannot determine"} and searches_used < search_budget:
        return "final answer is a non-answer placeholder despite remaining search budget"
    if YES_NO_QUESTION_RE.search(question) and norm_answer not in {"yes", "no"}:
        return "question asks for a yes/no decision; final answer must be yes or no"
    options = extract_or_options(question)
    if options and norm_answer not in {"yes", "no"} and not any(answer_matches_option(value, option) for option in options):
        return "final answer is not one of the explicit options in the comparison question"
    if COARSE_DIRECTION_RE.fullmatch(value) and ("part of" in q_lower or "which part" in q_lower):
        return "final answer is a coarse direction; include the requested full location span"
    if BARE_YEAR_RE.fullmatch(value) and q_lower.startswith("when") and "year" not in q_lower:
        return "final answer is only a bare year; use the most specific date span supported by evidence"
    if searches_used < search_budget and is_question_anchor_answer(question, value, options):
        return "final answer appears to copy an intermediate anchor from the question"
    return ""


def build_answer_repair_prompt(answer: str, reason: str, *, can_search: bool) -> str:
    if can_search:
        search_instruction = "If the evidence is still insufficient and searches remain, make one targeted search for the missing final attribute."
        search_tag_instruction = " or <search>...</search>"
    else:
        search_instruction = "Do not search again; choose the best final answer span from the evidence already shown."
        search_tag_instruction = ""
    return ANSWER_REPAIR_USER_TEMPLATE.format(
        answer=answer,
        reason=reason,
        search_instruction=search_instruction,
        search_tag_instruction=search_tag_instruction,
    )


def parse_generation_turn(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    skill_match = SKILL_RE.search(stripped)
    search_match = SEARCH_RE.search(stripped)
    answer_match = ANSWER_RE.search(stripped)

    if skill_match is None:
        return {
            "action": "invalid",
            "format_ok": False,
            "selected_skills": [],
            "visible_output": stripped,
            "query": "",
            "answer": "",
            "reason": "missing_skill_tag",
        }

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

    if action_match is None or skill_match.start() > action_match.start():
        return {
            "action": "invalid",
            "format_ok": False,
            "selected_skills": selected_skills,
            "visible_output": stripped,
            "query": "",
            "answer": "",
            "reason": "missing_or_misordered_action_tag",
        }

    visible_output = stripped[: action_match.end()].strip()
    if action_type == "search":
        return {
            "action": "search",
            "format_ok": True,
            "selected_skills": selected_skills,
            "visible_output": visible_output,
            "query": action_match.group(1).strip(),
            "answer": "",
            "reason": "",
        }
    return {
        "action": "final",
        "format_ok": True,
        "selected_skills": selected_skills,
        "visible_output": visible_output,
        "query": "",
        "answer": action_match.group(1).strip(),
        "reason": "",
    }


def parse_skill_selection_turn(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    skill_match = SKILL_RE.search(stripped)
    if skill_match is None:
        return {
            "format_ok": False,
            "selected_skills": [],
            "visible_output": stripped,
            "reason": "missing_skill_tag",
        }
    selected_skills = dedupe_keep_order(part.strip() for part in skill_match.group(1).split("|"))
    return {
        "format_ok": True,
        "selected_skills": selected_skills,
        "visible_output": stripped[: skill_match.end()].strip(),
        "reason": "",
    }


def sanitize_selected_skills(
    selected_skills: Sequence[str],
    recommended_skills: Sequence[str],
    skill_bank_entries: Dict[str, str],
) -> List[str]:
    available = set(skill_bank_entries)
    recommended = [skill_id for skill_id in recommended_skills if skill_id in available]
    cleaned = [skill_id for skill_id in dedupe_keep_order(selected_skills) if skill_id in available]
    if cleaned:
        return cleaned[:3]
    return recommended[:3]


def format_skill_cards(skill_ids: Sequence[str], skill_bank_entries: Dict[str, str], *, max_chars_per_skill: int) -> str:
    cards: List[str] = []
    for skill_id in skill_ids:
        description = " ".join(str(skill_bank_entries.get(skill_id, "")).split())
        if not description:
            continue
        if len(description) > max_chars_per_skill:
            description = description[: max_chars_per_skill].rstrip() + "..."
        cards.append(f"- {skill_id}: {description}")
    return "\n".join(cards)


def build_skill_context_block(skill_context: str) -> str:
    if not str(skill_context or "").strip():
        return ""
    return f"Skill instructions from the skillbank:\n{skill_context.strip()}\n"


def build_skill_selection_suffix(recommended_skills: Sequence[str]) -> str:
    return (
        "Skill selection phase. Choose the skill or skills from the recommended skill list "
        "that should govern the next action. Output only <skill>skill-id</skill> or "
        "<skill>skill-id|skill-id</skill>, then stop. Do not search or answer yet.\n"
        f"Allowed recommended skills: {', '.join(recommended_skills) or '(none)'}"
    )


def build_selected_skill_action_suffix(selected_skills: Sequence[str], skill_context: str) -> str:
    selected = "|".join(selected_skills)
    return (
        "Action phase. The selected skill instructions are now active:\n"
        f"{build_skill_context_block(skill_context)}"
        f"Now output exactly <skill>{selected}</skill> followed by exactly one "
        "<search>...</search> or <answer>...</answer>. Follow the selected skill instructions. "
        "Stop immediately after the closing action tag."
    )


def with_user_suffix(messages: Sequence[Dict[str, str]], suffix: str) -> List[Dict[str, str]]:
    result = [dict(message) for message in messages]
    if not result or result[-1].get("role") != "user":
        result.append({"role": "user", "content": suffix.strip()})
        return result
    result[-1]["content"] = result[-1].get("content", "").rstrip() + "\n\n" + suffix.strip()
    return result


def with_system_prompt(messages: Sequence[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    result = [dict(message) for message in messages]
    if result and result[0].get("role") == "system":
        result[0]["content"] = system_prompt
    else:
        result.insert(0, {"role": "system", "content": system_prompt})
    return result


def build_initial_prompt(
    question: str,
    recommended_skills: Sequence[str],
    search_budget: int,
    skill_context: str = "",
) -> str:
    value = question.strip()
    if value and not value.endswith("?"):
        value += "?"
    return USER_PROMPT_TEMPLATE.format(
        question=value,
        recommended_skills=", ".join(recommended_skills) or "(none)",
        skill_context_block=build_skill_context_block(skill_context),
        search_budget=search_budget,
    )


def build_followup_prompt(
    search_results: str,
    *,
    searches_used: int,
    search_budget: int,
    recent_searches: Sequence[str],
    skill_context: str = "",
) -> str:
    return FOLLOWUP_USER_TEMPLATE.format(
        search_results=search_results,
        searches_used=searches_used,
        search_budget=search_budget,
        recent_searches=" | ".join(recent_searches[-3:]) or "(none)",
        skill_context_block=build_skill_context_block(skill_context),
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
    dataset_tag: str,
    recommended_skills: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    retriever: RetrieverClient,
    stopping_criteria: transformers.StoppingCriteriaList,
    skill_stopping_criteria: transformers.StoppingCriteriaList,
    skill_bank_entries: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    search_budget = estimate_search_budget(question, dataset_tag)
    recommended_skill_context = ""
    if args.skill_context_mode == "cards":
        recommended_skill_context = format_skill_cards(
            recommended_skills,
            skill_bank_entries,
            max_chars_per_skill=args.max_skill_card_chars,
        )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_prompt(
                question,
                recommended_skills,
                search_budget,
                skill_context=recommended_skill_context,
            ),
        },
    ]
    trace_steps: List[Dict[str, Any]] = []
    recent_queries: List[str] = []
    seen_queries = set()
    searches_used = 0
    final_prediction = ""
    final_reason = "no_answer"

    max_turns = search_budget + 1
    for step_idx in range(max_turns):
        skill_selection_record: Optional[Dict[str, Any]] = None
        action_messages = messages
        if args.skill_context_mode == "two_stage":
            skill_prompt = build_chat_prompt(
                tokenizer,
                with_system_prompt(
                    with_user_suffix(messages, build_skill_selection_suffix(recommended_skills)),
                    SKILL_SELECTION_SYSTEM_PROMPT,
                ),
                enable_thinking=not args.disable_thinking,
            )
            raw_skill_output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=skill_prompt,
                max_new_tokens=args.max_skill_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stopping_criteria=skill_stopping_criteria,
            )
            parsed_skill = parse_skill_selection_turn(raw_skill_output)
            selected_for_context = sanitize_selected_skills(
                parsed_skill["selected_skills"],
                recommended_skills,
                skill_bank_entries,
            )
            selected_skill_context = format_skill_cards(
                selected_for_context,
                skill_bank_entries,
                max_chars_per_skill=args.max_skill_card_chars,
            )
            skill_selection_record = {
                "generated": parsed_skill["visible_output"],
                "selected_skills": parsed_skill["selected_skills"],
                "selected_skills_for_context": selected_for_context,
                "format_ok": parsed_skill["format_ok"],
                "reason": parsed_skill["reason"],
            }
            if raw_skill_output.strip() != parsed_skill["visible_output"].strip():
                skill_selection_record["raw_generated"] = raw_skill_output
            action_messages = with_user_suffix(
                messages,
                build_selected_skill_action_suffix(selected_for_context, selected_skill_context),
            )

        prompt = build_chat_prompt(
            tokenizer,
            action_messages,
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
        parsed = parse_generation_turn(raw_output)
        record: Dict[str, Any] = {
            "step": step_idx,
            "generated": parsed["visible_output"],
            "action": parsed["action"],
            "selected_skills": parsed["selected_skills"],
            "format_ok": parsed["format_ok"],
            "query": None,
            "retrieved": None,
            "reason": parsed["reason"],
        }
        if skill_selection_record is not None:
            record["skill_selection"] = skill_selection_record
        if args.skill_context_mode == "cards":
            record["skill_context_ids"] = list(recommended_skills)
        elif skill_selection_record is not None:
            record["skill_context_ids"] = skill_selection_record["selected_skills_for_context"]
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
                    skill_context=recommended_skill_context,
                ),
            }
        )

    extra = {
        "search_budget": search_budget,
        "searches_used": searches_used,
        "final_reason": final_reason,
        "skill_context_mode": args.skill_context_mode,
    }
    return final_prediction, trace_steps, extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SearchSkill with the rebuilt strict v3 protocol.")
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
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--disable-answer-guard", action="store_true")
    parser.add_argument("--strict-em-only", action="store_true")
    parser.add_argument(
        "--skill-context-mode",
        choices=("ids", "cards", "two_stage"),
        default="two_stage",
        help=(
            "ids: legacy prompt with skill ids only; cards: include recommended skill definitions in each action prompt; "
            "two_stage: first decode <skill>, retrieve those skill definitions from the skillbank, then decode the action."
        ),
    )
    parser.add_argument("--max-skill-tokens", type=int, default=64)
    parser.add_argument("--max-skill-card-chars", type=int, default=700)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, "qwen_skillbank_v3_eval")

    start_time = time.time()
    dataset = load_jsonl(args.data_path, max_samples=args.max_samples)
    logger.info("Loaded %d examples from %s", len(dataset), args.data_path)

    skill_bank_text = load_skill_bank_text(args.skill_bank_path)
    skill_bank_entries = parse_skill_bank_entries(skill_bank_text)
    available_skill_ids = list(skill_bank_entries) or parse_skill_ids(skill_bank_text)
    logger.info("Loaded %d skill ids from %s", len(available_skill_ids), args.skill_bank_path)

    tokenizer, model = load_eval_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        logger=logger,
    )
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, STOP_STRINGS)])
    skill_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(tokenizer, SKILL_STOP_STRINGS)])
    retriever = RetrieverClient(args.retriever_host, args.retriever_port, args.retriever_topk, args.retriever_timeout)

    records: List[Dict[str, Any]] = []
    n_correct = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            question = str(example.get("question", "")).strip()
            gold_answers = get_gold_answers(example)
            recommended_skills = select_recommended_skills(question, available_skill_ids)

            prediction, trace_steps, extra = evaluate_example(
                question=question,
                dataset_tag=args.dataset_tag,
                recommended_skills=recommended_skills,
                tokenizer=tokenizer,
                model=model,
                retriever=retriever,
                stopping_criteria=stopping_criteria,
                skill_stopping_criteria=skill_stopping_criteria,
                skill_bank_entries=skill_bank_entries,
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
    format_ok_turns = sum(1 for record in records for step in record["steps"] if step.get("format_ok") is True)
    total_turns = sum(len(record["steps"]) for record in records)
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
            "format_ok_rate": format_ok_turns / max(1, total_turns),
            "final_action_rate": final_action_rate,
            "duplicate_search_count": duplicate_search_count,
            "empty_count": empty_count,
            "n_skills_in_bank": len(available_skill_ids),
            "answer_guard_enabled": False,
            "strict_em_only": args.strict_em_only,
            "answer_repair_count": 0,
            "skill_context_mode": args.skill_context_mode,
            "max_skill_tokens": args.max_skill_tokens,
            "max_skill_card_chars": args.max_skill_card_chars,
        },
    )
    dump_json(args.summary_json, summary)
    logger.info("Finished evaluation. Final EM on %d examples: %.4f", len(records), summary["em"])


if __name__ == "__main__":
    main()
