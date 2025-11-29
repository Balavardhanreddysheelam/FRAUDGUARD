#!/usr/bin/env python3
"""
Merge the fine-tuned LoRA adapters with the base Llama-3.1-8B model
and produce a fully merged model ready for vLLM inference.
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_lora_path = project_root / "training" / "lora_model"
    default_output_path = project_root / "inference" / "model" / "fraudguard-8b-merged"

    parser = argparse.ArgumentParser(description="Merge LoRA adapters into full 8B model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=default_lora_path,
        help="Path to the trained LoRA adapters",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path,
        help="Where to save the merged full model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Torch dtype for merging",
    )
    return parser.parse_args()


def merge_lora(base_model: str, lora_path: Path, output_path: Path, dtype: str) -> None:
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    print("🔄 Loading base model:", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("🧩 Loading LoRA adapters from:", lora_path)
    lora_model = PeftModel.from_pretrained(base, str(lora_path))

    print("🔗 Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()
    merged_model.config.use_cache = True

    print("✂️  Removing existing output directory (if any)...")
    if output_path.exists():
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    print("💾 Saving merged model to:", output_path)
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    print("🪄 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("\n✅ Merge complete!")
    print(f"   Merged model path: {output_path}")
    print("   Next step: point vLLM to this directory.")


def main() -> None:
    args = parse_args()
    merge_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()

