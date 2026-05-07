from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a dense Hugging Face checkpoint.")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_base_model_path(adapter_path: Path, explicit_base_model_path: str | None) -> str:
    if explicit_base_model_path:
        return explicit_base_model_path
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_model_path = str(payload.get("base_model_name_or_path") or "").strip()
    if not base_model_path:
        raise ValueError("base_model_name_or_path is missing from adapter_config.json")
    return base_model_path


def resolve_tokenizer_source(adapter_path: Path, base_model_path: str) -> str:
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return base_model_path


def maybe_clear_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output dir already exists and is not empty: {output_dir}")
    shutil.rmtree(output_dir)


def main() -> None:
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    adapter_path = args.adapter_path.resolve()
    output_dir = args.output_dir.resolve()
    base_model_path = resolve_base_model_path(adapter_path, args.base_model_path)
    tokenizer_source = resolve_tokenizer_source(adapter_path, base_model_path)

    maybe_clear_output_dir(output_dir, overwrite=args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_map = None
    low_cpu_mem_usage = True
    if args.device == "cuda":
        device_map = {"": 0}

    print(
        json.dumps(
            {
                "stage": "merge_start",
                "adapter_path": str(adapter_path),
                "base_model_path": base_model_path,
                "tokenizer_source": tokenizer_source,
                "output_dir": str(output_dir),
                "dtype": args.dtype,
                "device": args.device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "adapter_path": str(adapter_path),
        "base_model_path": base_model_path,
        "tokenizer_source": tokenizer_source,
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "device": args.device,
        "max_shard_size": args.max_shard_size,
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"stage": "merge_done", "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
