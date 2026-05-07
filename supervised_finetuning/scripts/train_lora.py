from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT model on SearchSkill trajectories.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--init-adapter-path", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--truncate-side", choices=("left", "right"), default="left")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--search-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_chat_template(tokenizer: AutoTokenizer, messages: Sequence[Dict[str, str]]) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
    )
    return list(encoded)


def assistant_turn_weight(content: str, *, answer_loss_weight: float, search_loss_weight: float) -> float:
    lowered = str(content or "").lower()
    if "<answer>" in lowered:
        return float(answer_loss_weight)
    if "<search>" in lowered:
        return float(search_loss_weight)
    return 1.0


def build_example(
    tokenizer: AutoTokenizer,
    row: Dict[str, Any],
    *,
    max_length: int,
    truncate_side: str,
    answer_loss_weight: float,
    search_loss_weight: float,
) -> Dict[str, List[int]]:
    messages = row["messages"]
    input_ids: List[int] = []
    labels: List[int] = []
    loss_weights: List[float] = []
    previous_ids: List[int] = []
    supervision_mode = str(row.get("supervision_mode", "all_assistant")).strip().lower()
    assistant_indices = [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]
    last_assistant_idx = assistant_indices[-1] if assistant_indices else -1

    for idx, message in enumerate(messages):
        current_ids = apply_chat_template(tokenizer, messages[: idx + 1])
        delta = current_ids[len(previous_ids) :]
        if not delta:
            previous_ids = current_ids
            continue
        if message.get("role") == "assistant":
            supervise_turn = supervision_mode != "final_only" or idx == last_assistant_idx
            if supervise_turn:
                delta_labels = list(delta)
                weight = assistant_turn_weight(
                    message.get("content", ""),
                    answer_loss_weight=answer_loss_weight,
                    search_loss_weight=search_loss_weight,
                )
                delta_loss_weights = [weight] * len(delta)
            else:
                delta_labels = [-100] * len(delta)
                delta_loss_weights = [0.0] * len(delta)
        else:
            delta_labels = [-100] * len(delta)
            delta_loss_weights = [0.0] * len(delta)
        input_ids.extend(delta)
        labels.extend(delta_labels)
        loss_weights.extend(delta_loss_weights)
        previous_ids = current_ids

    if len(input_ids) > max_length:
        if truncate_side == "left":
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
            loss_weights = loss_weights[-max_length:]
        else:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            loss_weights = loss_weights[:max_length]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights,
    }


def build_dataset(
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    *,
    max_length: int,
    truncate_side: str,
    answer_loss_weight: float,
    search_loss_weight: float,
) -> Dataset:
    packed_rows: List[Dict[str, Any]] = []
    for row in rows:
        example = build_example(
            tokenizer,
            row,
            max_length=max_length,
            truncate_side=truncate_side,
            answer_loss_weight=answer_loss_weight,
            search_loss_weight=search_loss_weight,
        )
        if any(label != -100 for label in example["labels"]):
            packed_rows.append(example)
    return Dataset.from_list(packed_rows)


class WeightedSFTDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_loss_weights: List[List[float]] = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            batch_labels.append(feature["labels"] + [-100] * pad_length)
            batch_loss_weights.append(feature["loss_weights"] + [0.0] * pad_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "loss_weights": torch.tensor(batch_loss_weights, dtype=torch.float32),
        }


class WeightedSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        flat_weights = shift_weights.view(-1)
        weighted_loss = token_loss * flat_weights
        denom = flat_weights.sum().clamp_min(1.0)
        loss = weighted_loss.sum() / denom
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("CUDA is required for SearchSkill SFT training.")
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        json.dumps(
            {
                "cuda_available": use_cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "use_bf16": use_bf16,
                "use_fp16": use_fp16,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tokenizer_source = str(args.init_adapter_path) if args.init_adapter_path and (args.init_adapter_path / "tokenizer_config.json").exists() else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = args.truncate_side

    train_rows = load_jsonl(args.train_path)
    eval_rows = load_jsonl(args.eval_path)
    train_dataset = build_dataset(
        tokenizer,
        train_rows,
        max_length=args.max_length,
        truncate_side=args.truncate_side,
        answer_loss_weight=args.answer_loss_weight,
        search_loss_weight=args.search_loss_weight,
    )
    eval_dataset = build_dataset(
        tokenizer,
        eval_rows,
        max_length=args.max_length,
        truncate_side=args.truncate_side,
        answer_loss_weight=args.answer_loss_weight,
        search_loss_weight=args.search_loss_weight,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()

    if args.init_adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(args.init_adapter_path),
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        bf16=use_bf16,
        fp16=use_fp16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=True,
        report_to=[],
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
    )

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WeightedSFTDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
