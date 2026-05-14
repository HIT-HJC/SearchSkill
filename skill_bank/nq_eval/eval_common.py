from __future__ import annotations

import json
import logging
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def dump_json(path: str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flush_file(handle) -> None:
    handle.flush()


def setup_logger(log_file: str, name: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def build_chat_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        template_code = getattr(tokenizer.apply_chat_template, "__code__", None)
        if template_code is not None and "enable_thinking" in template_code.co_varnames:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    parts = []
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.bfloat16)


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype_name: str,
    trust_remote_code: bool,
    logger: logging.Logger,
) -> Tuple[transformers.PreTrainedTokenizerBase, transformers.PreTrainedModel]:
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    logger.info("Loading model from %s", model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(dtype_name),
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_text(
    *,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stopping_criteria: transformers.StoppingCriteriaList,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generation_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "stopping_criteria": stopping_criteria,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^['\"]|['\"]$", "", text)
    return text.strip()


def exact_match_multi(prediction: str, gold_answers: List[str]) -> int:
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(gold) for gold in gold_answers))


def extract_answer(text: str) -> Tuple[str, str]:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return "answer_tag", match.group(1).strip()
    return "raw", text.strip()


def build_summary(
    *,
    model_path: str,
    data_path: str,
    out_jsonl: str,
    log_file: str,
    n_examples: int,
    n_correct: int,
    start_time: float,
    end_time: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "model_path": model_path,
        "data_path": data_path,
        "out_jsonl": out_jsonl,
        "log_file": log_file,
        "n_examples": n_examples,
        "n_correct": n_correct,
        "em": n_correct / max(1, n_examples),
        "elapsed_wall_seconds": end_time - start_time,
    }
    if extra:
        summary.update(extra)
    return summary
