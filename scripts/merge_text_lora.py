"""Merge a text-only Qwen3.5 LoRA adapter into a standalone checkpoint."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.model_utils import load_text_causal_lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = load_text_causal_lm(args.base, dtype=torch.bfloat16, device="cpu")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(output)
    print(f"Merged {args.adapter} into {output}")


if __name__ == "__main__":
    main()
