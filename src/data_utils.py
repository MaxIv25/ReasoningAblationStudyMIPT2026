"""
Data loading and preparation utilities for SFT Ablation Study.

Handles:
- OpenR1-Math-220K loading and filtering
- Chat template formatting for Qwen3.5
- Curriculum sorting
- Train/eval splitting
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, save_results

logger = setup_logging("data_utils")


# ──────────────────────────────────────────────────────────────
# OpenR1-Math-220K loading & filtering
# ──────────────────────────────────────────────────────────────

def load_openr1_math(
    split: str = "default",
    num_samples: int = None,
    seed: int = 42,
    filter_config: dict = None,
) -> Dataset:
    """
    Load and filter OpenR1-Math-220K dataset.
    
    Args:
        split: Dataset split ("default" or "extended")
        num_samples: Number of samples to keep (None = all)
        seed: Random seed for sampling
        filter_config: Dict with filtering parameters
    
    Returns:
        Filtered HuggingFace Dataset
    """
    logger.info(f"Loading open-r1/OpenR1-Math-220k split={split}...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split=split)
    logger.info(f"Raw dataset size: {len(ds)}")

    # Apply filtering
    if filter_config is None:
        filter_config = {
            "min_think_words": 20,
            "max_think_words": 2000,
            "require_think_tags": True,
            "require_boxed": True,
        }

    def filter_fn(example):
        solution = example.get("solution", "") or ""

        # Check for <think> tags
        if filter_config.get("require_think_tags", True):
            if "<think>" not in solution or "</think>" not in solution:
                return False

        # Check for \boxed{}
        if filter_config.get("require_boxed", True):
            if "\\boxed{" not in solution:
                return False

        # Check thinking length
        think_match = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
            word_count = len(think_text.split())
            min_words = filter_config.get("min_think_words", 20)
            max_words = filter_config.get("max_think_words", 2000)
            if word_count < min_words or word_count > max_words:
                return False
        elif filter_config.get("require_think_tags", True):
            return False

        return True

    filtered = ds.filter(filter_fn, num_proc=4)
    logger.info(f"After filtering: {len(filtered)} ({len(filtered)/len(ds)*100:.1f}%)")

    # Sample if needed
    if num_samples and num_samples < len(filtered):
        filtered = filtered.shuffle(seed=seed).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples")

    return filtered


def get_solution_token_length(example: dict, tokenizer=None) -> int:
    """
    Get approximate length of solution in tokens.
    
    Uses word count as proxy if tokenizer is not provided.
    """
    solution = example.get("solution", "")
    if tokenizer:
        return len(tokenizer(solution)["input_ids"])
    return len(solution.split())


# ──────────────────────────────────────────────────────────────
# Chat template formatting
# ──────────────────────────────────────────────────────────────

def format_for_sft(
    dataset: Dataset,
    tokenizer=None,
    system_prompt: str = None,
) -> Dataset:
    """
    Format dataset into chat messages for SFT with Qwen3.5.
    
    The format follows Qwen's chat template:
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    {solution}<|im_end|>
    
    Args:
        dataset: Filtered OpenR1-Math dataset
        tokenizer: Tokenizer (optional, for applying chat template)
        system_prompt: Optional system prompt
    
    Returns:
        Dataset with "messages" or "text" column
    """

    def format_fn(example):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": example["problem"]})
        messages.append({"role": "assistant", "content": example["solution"]})

        result = {"messages": messages}

        # Add solution length for curriculum sorting
        result["solution_length"] = len(example.get("solution", "").split())

        return result

    formatted = dataset.map(format_fn, num_proc=4)
    logger.info(f"Formatted {len(formatted)} examples into chat messages")
    return formatted


# ──────────────────────────────────────────────────────────────
# Curriculum sorting
# ──────────────────────────────────────────────────────────────

def apply_curriculum(
    dataset: Dataset,
    order: str = "easy_to_hard",
) -> Dataset:
    """
    Sort dataset by difficulty (solution length as proxy).
    
    Args:
        dataset: Dataset with "solution_length" column
        order: "easy_to_hard", "hard_to_easy", or "random"
    
    Returns:
        Sorted dataset
    """
    if order == "random":
        return dataset.shuffle(seed=42)
    elif order == "easy_to_hard":
        return dataset.sort("solution_length")
    elif order == "hard_to_easy":
        return dataset.sort("solution_length", reverse=True)
    else:
        raise ValueError(f"Unknown curriculum order: {order}")


# ──────────────────────────────────────────────────────────────
# Train/eval split
# ──────────────────────────────────────────────────────────────

def split_train_eval(
    dataset: Dataset,
    eval_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """Split dataset into train and eval."""
    ds_split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    logger.info(
        f"Split: train={len(ds_split['train'])}, eval={len(ds_split['test'])}"
    )
    return {"train": ds_split["train"], "eval": ds_split["test"]}


# ──────────────────────────────────────────────────────────────
# Data preparation pipeline
# ──────────────────────────────────────────────────────────────

def prepare_sft_data(
    config: dict,
    output_dir: str = None,
    curriculum: str = "random",
) -> dict:
    """
    Full data preparation pipeline.
    
    Args:
        config: Configuration dict
        output_dir: Where to save prepared data (optional)
        curriculum: "random", "easy_to_hard", "hard_to_easy"
    
    Returns:
        Dict with "train" and "eval" datasets
    """
    data_config = config.get("data", {})
    filter_config = data_config.get("filter", {})

    # Load and filter
    ds = load_openr1_math(
        split=data_config.get("split", "default"),
        num_samples=data_config.get("num_samples", 20000),
        seed=data_config.get("seed", 42),
        filter_config=filter_config,
    )

    # Format for SFT
    formatted = format_for_sft(ds)

    # Apply curriculum
    formatted = apply_curriculum(formatted, order=curriculum)

    # Split
    splits = split_train_eval(
        formatted,
        eval_fraction=data_config.get("eval_fraction", 0.05),
        seed=data_config.get("seed", 42),
    )

    # Save if output dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        splits["train"].save_to_disk(os.path.join(output_dir, "train"))
        splits["eval"].save_to_disk(os.path.join(output_dir, "eval"))
        logger.info(f"Saved prepared data to {output_dir}")

        # Save stats
        stats = {
            "raw_size": len(ds),
            "train_size": len(splits["train"]),
            "eval_size": len(splits["eval"]),
            "curriculum": curriculum,
            "filter_config": filter_config,
        }
        save_results(stats, os.path.join(output_dir, "data_stats.json"))

    return splits


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--curriculum", type=str, default="random",
        choices=["random", "easy_to_hard", "hard_to_easy"],
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of samples",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_samples:
        config.setdefault("data", {})["num_samples"] = args.num_samples

    prepare_sft_data(config, output_dir=args.output, curriculum=args.curriculum)


if __name__ == "__main__":
    main()
