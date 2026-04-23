"""
Data loading and preparation utilities for SFT Ablation Study.

OpenR1-Math-220K structure:
    - problem: str — math problem text
    - generations: list[str] — 2-4 reasoning traces from DeepSeek-R1
    - correctness_math_verify: list[bool] — correctness for each trace
    - correctness_llama: list[bool] — correctness verified by Llama-3.3-70B
    - messages: list[dict] — pre-formatted chat messages (from first correct trace)
    - solution: str — original NuminaMath solution (NOT R1 trace)

For SFT we use `generations` (R1 traces), picking the first correct one per problem.

Handles:
- OpenR1-Math-220K loading and filtering (correct R1 traces only)
- Chat template formatting for Qwen3.5 with <think> prefix
- Curriculum sorting by trace length
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
    
    For each problem, selects the FIRST CORRECT trace from `generations`
    (verified by math_verify). Filters by trace quality.
    
    Args:
        split: Dataset split ("default" or "extended")
        num_samples: Number of samples to keep (None = all)
        seed: Random seed for sampling
        filter_config: Dict with filtering parameters
    
    Returns:
        Filtered HuggingFace Dataset with columns:
        - problem: str
        - trace: str (selected correct R1 reasoning trace)
        - trace_length: int (word count of trace)
        - difficulty: float (1 - pass_rate, higher = harder)
    """
    logger.info(f"Loading open-r1/OpenR1-Math-220k split={split}...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split=split)
    logger.info(f"Raw dataset size: {len(ds)}")

    if filter_config is None:
        filter_config = {
            "min_trace_words": 20,
            "require_think_tags": True,
            "require_boxed": True,
        }

    min_words = filter_config.get("min_trace_words", 20)
    require_think = filter_config.get("require_think_tags", True)
    require_boxed = filter_config.get("require_boxed", True)

    # Process each example: pick first correct trace
    processed = []
    skipped_no_correct = 0
    skipped_filter = 0

    for example in tqdm(ds, desc="Processing traces"):
        generations = example.get("generations", [])
        correctness = example.get("correctness_math_verify", [])

        # Find the first correct generation
        trace = None
        for gen, is_correct in zip(generations, correctness):
            if is_correct:
                trace = gen
                break

        if trace is None:
            skipped_no_correct += 1
            continue

        # Apply filters
        if require_think and ("<think>" not in trace or "</think>" not in trace):
            skipped_filter += 1
            continue

        if require_boxed and "\\boxed{" not in trace:
            skipped_filter += 1
            continue

        word_count = len(trace.split())
        if word_count < min_words:
            skipped_filter += 1
            continue

        # Compute difficulty from correctness ratio:
        # pass_rate = fraction of correct generations
        # difficulty = 1 - pass_rate (higher = harder)
        # E.g.: 1/4 correct → difficulty=0.75, 4/4 correct → difficulty=0.0
        num_correct = sum(1 for c in correctness if c)
        num_total = len(correctness)
        pass_rate = num_correct / num_total if num_total > 0 else 0
        difficulty = 1.0 - pass_rate

        processed.append({
            "problem": example["problem"],
            "trace": trace,
            "trace_length": word_count,
            "difficulty": round(difficulty, 4),
        })

    logger.info(
        f"Processed: {len(processed)} examples "
        f"(skipped {skipped_no_correct} no correct trace, "
        f"{skipped_filter} filtered out)"
    )

    # Convert to HuggingFace Dataset
    result_ds = Dataset.from_list(processed)

    # Log difficulty distribution
    diffs = result_ds["difficulty"]
    logger.info(
        f"Difficulty: min={min(diffs):.2f}, max={max(diffs):.2f}, "
        f"mean={sum(diffs)/len(diffs):.2f}"
    )

    # Sample if needed
    if num_samples and num_samples < len(result_ds):
        result_ds = result_ds.shuffle(seed=seed).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples")

    return result_ds


# ──────────────────────────────────────────────────────────────
# Chat template formatting
# ──────────────────────────────────────────────────────────────

def format_for_sft(
    dataset: Dataset,
    add_think_prefix: bool = True,
) -> Dataset:
    """
    Format dataset into chat messages for SFT with Qwen3.5.
    
    Qwen3 uses the following chat template:
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        <think>
        {reasoning}
        </think>
        {final_answer}<|im_end|>
    
    Since we're training a Base model, we need to explicitly include
    <think> at the start of the assistant response so the model learns
    to always begin with reasoning.
    
    If the trace already starts with <think>, we use it as-is.
    If not, we prepend <think> to teach the thinking format.
    
    Args:
        dataset: Dataset with "problem" and "trace" columns
        add_think_prefix: If True, ensure trace starts with <think>
    
    Returns:
        Dataset with "messages" column for TRL SFTTrainer
    """

    def format_fn(example):
        trace = example["trace"]

        # Ensure the trace starts with <think> tag
        if add_think_prefix and not trace.strip().startswith("<think>"):
            trace = "<think>\n" + trace

        messages = [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": trace},
        ]

        return {
            "messages": messages,
            "solution_length": example.get("trace_length", len(trace.split())),
            "difficulty": example.get("difficulty", 0.0),
        }

    formatted = dataset.map(format_fn, num_proc=4)
    logger.info(f"Formatted {len(formatted)} examples into chat messages")

    # Log stats
    lengths = formatted["solution_length"]
    logger.info(
        f"Trace lengths: min={min(lengths)}, max={max(lengths)}, "
        f"mean={sum(lengths)/len(lengths):.0f}, "
        f"median={sorted(lengths)[len(lengths)//2]}"
    )

    return formatted


# ──────────────────────────────────────────────────────────────
# Curriculum sorting
# ──────────────────────────────────────────────────────────────

def apply_curriculum(
    dataset: Dataset,
    order: str = "easy_to_hard",
) -> Dataset:
    """
    Sort dataset by difficulty for curriculum learning.
    
    Difficulty is computed from the correctness ratio in OpenR1-Math:
    - difficulty = 1 - (num_correct / num_total_generations)
    - E.g.: 1/4 correct → difficulty=0.75 (hard)
    - E.g.: 4/4 correct → difficulty=0.0 (easy)
    
    This is a much better proxy than trace length, because:
    - Length ≠ difficulty (verbose ≠ hard, short ≠ easy)
    - Length-based sorting creates bias toward longer outputs
    - Correctness ratio directly measures how hard R1 found the problem
    
    Args:
        dataset: Dataset with "difficulty" column
        order: "easy_to_hard", "hard_to_easy", or "random"
    
    Returns:
        Sorted dataset
    """
    if order == "random":
        return dataset.shuffle(seed=42)
    elif order == "easy_to_hard":
        return dataset.sort("difficulty")
    elif order == "hard_to_easy":
        return dataset.sort("difficulty", reverse=True)
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
            "raw_dataset_size": "open-r1/OpenR1-Math-220k",
            "processed_size": len(ds),
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
