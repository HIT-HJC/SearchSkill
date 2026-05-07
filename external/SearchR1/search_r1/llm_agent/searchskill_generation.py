from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import requests
import torch
from verl import DataProto

from search_r1.llm_agent.generation import LLMGenerationManager


SELECT_RE = re.compile(r"<select_skill>(.*?)</select_skill>", re.IGNORECASE | re.DOTALL)
SKILL_HEADER_RE = re.compile(r"^`([a-z0-9][a-z0-9\-]*)`\s*$")

SELECTION_INSTRUCTION = (
    "Selection phase: choose 1 to 3 skill ids from the the final SkillBank index that should govern the next action. "
    "If the evidence is sufficient, choose a closure skill such as verbatim-evidence-span and/or "
    "answer-grounding-check. Output only <select_skill>skill-id</select_skill> or "
    "<select_skill>skill-id|skill-id</select_skill>. Do not search or answer in this turn."
)

ACTION_INSTRUCTION_TEMPLATE = (
    "Action phase. Read the selected the final SkillBank card(s) and follow them for the next action.\n"
    "Selected skill ids: {skill_ids}\n"
    "Now output exactly <skill>{skill_ids}</skill> followed by exactly one <search>...</search> or "
    "<answer>...</answer>. Use the same skill ids in the <skill> tag. Do not output <select_skill> "
    "in this turn. Stop immediately after the closing action tag."
)

FOLLOWUP_TEMPLATE = (
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

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.IGNORECASE | re.DOTALL)


def compact_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars].rstrip() + "..."
    return value


def parse_skill_bank() -> dict[str, str]:
    raw_path = os.environ.get("SEARCHSKILL_SKILL_BANK_PATH", "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_file():
        return {}
    entries: dict[str, str] = {}
    current = ""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = SKILL_HEADER_RE.match(raw.strip())
        if match:
            if current:
                entries[current] = " ".join(item.strip() for item in lines if item.strip())
            current = match.group(1)
            lines = []
        elif current:
            lines.append(raw)
    if current:
        entries[current] = " ".join(item.strip() for item in lines if item.strip())
    return entries


SKILL_BANK = parse_skill_bank()


def format_skill_index() -> str:
    lines = ["Available the final SkillBank index:"]
    for skill_id, desc in SKILL_BANK.items():
        lines.append(f"- {skill_id}: {compact_text(desc, 180)}")
    return "\n".join(lines)


def dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_skill_ids(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]


def parse_selected(text: str) -> list[str]:
    match = SELECT_RE.search(text or "")
    if not match:
        return []
    legal = set(SKILL_BANK)
    return [skill for skill in dedupe_keep_order(match.group(1).split("|")) if skill in legal][:3]


def normalize_query(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return " ".join(value.split())


def parse_action_prediction(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    skill_match = SKILL_RE.search(raw)
    action_matches = list(ACTION_RE.finditer(raw))
    if skill_match is None:
        return {"action": "invalid", "skills": [], "content": "", "reason": "missing_skill_tag"}
    if not action_matches:
        return {"action": "invalid", "skills": parse_skill_ids(skill_match.group(1)), "content": "", "reason": "missing_action_tag"}
    action_match = action_matches[0]
    if skill_match.start() > action_match.start():
        return {
            "action": "invalid",
            "skills": parse_skill_ids(skill_match.group(1)),
            "content": "",
            "reason": "misordered_skill_action_tags",
        }
    skills = parse_skill_ids(skill_match.group(1))
    if not skills or any(skill not in SKILL_BANK for skill in skills):
        return {"action": "invalid", "skills": skills, "content": "", "reason": "invalid_skill_id"}
    return {
        "action": action_match.group(1).lower(),
        "skills": skills,
        "content": action_match.group(2).strip(),
        "reason": "",
    }


def selected_cards(skill_ids: Sequence[str]) -> str:
    lines = ["Selected the final SkillBank card(s):"]
    for skill_id in skill_ids:
        lines.append(f'<skill_card id="{skill_id}">\n{compact_text(SKILL_BANK.get(skill_id, ""), 900)}\n</skill_card>')
    return "\n".join(lines)


def build_action_prompt(skill_ids: Sequence[str]) -> str:
    skill_text = "|".join(skill_ids)
    return f"{selected_cards(skill_ids)}\n\n{ACTION_INSTRUCTION_TEMPLATE.format(skill_ids=skill_text)}"


def search_budgets_from_batch(gen_batch: DataProto, batch_size: int) -> list[int]:
    budgets = [5] * batch_size
    extra = getattr(gen_batch, "non_tensor_batch", {}).get("extra_info") if getattr(gen_batch, "non_tensor_batch", None) else None
    if extra is None:
        return budgets
    for idx, item in enumerate(extra):
        try:
            budget = int((item or {}).get("search_budget") or 5)
        except Exception:
            budget = 5
        budgets[idx] = min(max(1, budget), 5)
    return budgets


def trim_to_tag(text: str, tag: str) -> str:
    close = f"</{tag}>"
    idx = text.lower().find(close)
    if idx >= 0:
        return text[: idx + len(close)].strip()
    return text.strip()


class SearchSkillGenerationManager(LLMGenerationManager):
    def _batch_search(self, queries):
        payload = {"queries": queries, "topk": self.config.topk, "return_scores": True}
        session = requests.Session()
        session.trust_env = False
        last_error = None
        for attempt in range(4):
            try:
                response = session.post(self.config.search_url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"SearchSkill retriever failed after retries: {last_error}")

    def batch_search(self, queries):
        if not queries:
            return []
        outputs = []
        chunk_size = int(os.environ.get("SEARCHSKILL_RETRIEVER_CHUNK", "4"))
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start : start + chunk_size]
            results = self._batch_search(chunk)["result"]
            outputs.extend(self._passages2string(result) for result in results)
        return outputs

    def _postprocess_select(self, responses: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        strings = [trim_to_tag(item, "select_skill") for item in self.tokenizer.batch_decode(responses, skip_special_tokens=True)]
        return self._batch_tokenize(strings), strings

    def _postprocess_action(self, responses: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        strings = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        strings = [
            item.split("</search>")[0] + "</search>"
            if "</search>" in item
            else item.split("</answer>")[0] + "</answer>"
            if "</answer>" in item
            else item.strip()
            for item in strings
        ]
        return self._batch_tokenize(strings), strings

    def _append_to_rolling(self, rollings: DataProto, response_ids: torch.Tensor, obs_texts: list[str]) -> DataProto:
        obs_ids = self._process_next_obs(obs_texts)
        return self._update_rolling_state(rollings, response_ids, obs_ids)

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor):
        original_left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {"responses": initial_input_ids[:, []], "responses_with_info_mask": initial_input_ids[:, []]}
        batch_size = gen_batch.batch["input_ids"].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        search_budgets = search_budgets_from_batch(gen_batch, batch_size)
        searches_used = [0] * batch_size
        recent_queries = [[] for _ in range(batch_size)]
        seen_queries = [set() for _ in range(batch_size)]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=["input_ids", "attention_mask", "position_ids"])
            active_batch = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            select_output = self._generate_with_gpu_padding(active_batch)
            select_ids, select_str = self._postprocess_select(select_output.batch["responses"])
            select_ids, select_str = self.tensor_fn._example_level_pad(select_ids, select_str, active_mask)
            selected = [parse_selected(text) if active else [] for text, active in zip(select_str, active_mask)]
            selection_valid = torch.tensor([bool(items) for items in selected], dtype=torch.bool)
            action_prompts = [build_action_prompt(items) if valid else "" for items, valid in zip(selected, selection_valid)]

            rollings = self._append_to_rolling(rollings, select_ids, action_prompts)
            original_right_side = self._update_right_side(original_right_side, select_ids, self._process_next_obs(action_prompts))
            active_mask = active_mask & selection_valid
            if not active_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=["input_ids", "attention_mask", "position_ids"])
            active_batch = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            action_output = self._generate_with_gpu_padding(active_batch)
            action_ids, action_str = self._postprocess_action(action_output.batch["responses"])
            action_ids, action_str = self.tensor_fn._example_level_pad(action_ids, action_str, active_mask)
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                action_str,
                self.tokenizer.pad_token,
                active_mask,
                selected_context=selected,
                search_budgets=search_budgets,
                searches_used=searches_used,
                recent_queries=recent_queries,
                seen_queries=seen_queries,
            )
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            rollings = self._append_to_rolling(rollings, action_ids, next_obs)
            original_right_side = self._update_right_side(original_right_side, action_ids, self._process_next_obs(next_obs))

        meta_info = {"turns_stats": turns_stats.tolist(), "active_mask": active_mask.tolist(), "valid_action_stats": valid_action_stats.tolist(), "valid_search_stats": valid_search_stats.tolist()}
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def execute_predictions(
        self,
        predictions: list[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
        selected_context: list[list[str]] | None = None,
        search_budgets: list[int] | None = None,
        searches_used: list[int] | None = None,
        recent_queries: list[list[str]] | None = None,
        seen_queries: list[set[str]] | None = None,
    ):
        parsed = [parse_action_prediction(item) for item in predictions]
        next_obs, dones, valid_action, is_search = [], [], [], []
        search_jobs: list[tuple[int, str]] = []
        for idx, (item, active) in enumerate(zip(parsed, active_mask)):
            if not active or item["action"] != "search":
                continue
            expected = (selected_context or [[]])[idx]
            if set(item["skills"]) != set(expected):
                continue
            query = str(item["content"]).strip()
            norm = normalize_query(query)
            budget = (search_budgets or [5] * len(parsed))[idx]
            used = (searches_used or [0] * len(parsed))[idx]
            if not norm or used >= budget:
                continue
            if seen_queries is not None and norm in seen_queries[idx]:
                continue
            search_jobs.append((idx, query))
        raw_search_results = self.batch_search([query for _, query in search_jobs]) if do_search else [""] * len(search_jobs)
        search_result_by_idx = {idx: result for (idx, _), result in zip(search_jobs, raw_search_results)}
        skill_index = format_skill_index()
        for idx, (item, active) in enumerate(zip(parsed, active_mask)):
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            expected = (selected_context or [[]])[idx]
            if set(item["skills"]) != set(expected):
                next_obs.append("\nMy previous action used skill ids that do not match the selected skills. I should select valid skills, then use the same ids in <skill> before one action.\n")
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
            elif item["action"] == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
            elif item["action"] == "search":
                query = str(item["content"]).strip()
                norm = normalize_query(query)
                budget = (search_budgets or [5] * len(parsed))[idx]
                used = (searches_used or [0] * len(parsed))[idx]
                if not norm:
                    next_obs.append("\nMy previous search query was empty. I should select valid skills, then output one targeted non-empty search or answer.\n")
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    continue
                if used >= budget:
                    next_obs.append("")
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                    continue
                if seen_queries is not None and norm in seen_queries[idx]:
                    result = "Duplicate search target detected. Answer now or search for a different missing entity or attribute."
                else:
                    result = str(search_result_by_idx.get(idx, "")).strip()
                    if seen_queries is not None:
                        seen_queries[idx].add(norm)
                if searches_used is not None:
                    searches_used[idx] += 1
                    used = searches_used[idx]
                if recent_queries is not None:
                    recent_queries[idx].append(query)
                    recent = " | ".join(recent_queries[idx][-3:]) or "(none)"
                else:
                    recent = query or "(none)"
                next_obs.append(
                    FOLLOWUP_TEMPLATE.format(
                        search_results=result,
                        searches_used=used,
                        search_budget=budget,
                        recent_searches=recent,
                        skill_index=skill_index,
                        selection_instruction=SELECTION_INSTRUCTION,
                    )
                )
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
            else:
                next_obs.append("\nMy previous action is invalid. I should first select valid skills, then output one search or answer action.\n")
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
        return next_obs, dones, valid_action, is_search
