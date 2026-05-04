"""
Prepare dataset for GRPO training from OpenR1-Math-220K.

GRPOTrainer needs:
- `prompt`: list of chat messages (conversational format)
- `solution`: ground truth answer string (passed to reward function via **kwargs)

We extract problems and their verified answers from OpenR1-Math-220K.
The trainer handles generation internally — we only provide prompts + answers.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, extract_boxed_answer

logger = setup_logging("prepare_grpo_data")


def prepare_grpo_dataset(
    split: str = "train",
    num_samples: int = 20000,
    seed: int = 42,
    output_dir: str = None,
) -> Dataset:
    """
    Load OpenR1-Math-220K and prepare for GRPO.

    For each problem:
    1. Find the first correct trace (verified by math_verify)
    2. Extract the ground truth answer from \\boxed{} in that trace
    3. Build conversational prompt (user message with the problem)

    Args:
        split: Dataset split ("train" or "default")
        num_samples: Number of samples to keep
        seed: Random seed for sampling
        output_dir: Where to save the prepared dataset

    Returns:
        HuggingFace Dataset with columns: prompt, solution
    """
    logger.info(f"Loading open-r1/OpenR1-Math-220k split={split}...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split=split)
    logger.info(f"Raw dataset size: {len(ds)}")

    processed = []
    skipped_no_correct = 0
    skipped_no_answer = 0

    for example in tqdm(ds, desc="Extracting problems + answers"):
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

        # Extract ground truth answer from \boxed{} in the correct trace
        answer = extract_boxed_answer(trace)
        if answer is None:
            skipped_no_answer += 1
            continue

        # Build conversational prompt
        # System message tells model to reason step-by-step
        prompt = [
            {
                "role": "user",
                "content": (
                    f"{example['problem']}\n\n"
                    "Please reason step by step, and put your final answer "
                    "within \\boxed{}."
                ),
            }
        ]

        processed.append({
            "prompt": prompt,
            "solution": answer,
        })

    logger.info(
        f"Processed: {len(processed)} problems "
        f"(skipped {skipped_no_correct} no correct trace, "
        f"{skipped_no_answer} no \\boxed answer)"
    )

    result_ds = Dataset.from_list(processed)

    # Sample if needed
    if num_samples and num_samples < len(result_ds):
        result_ds = result_ds.shuffle(seed=seed).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples")

    logger.info(f"Final dataset size: {len(result_ds)}")

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_ds.save_to_disk(output_dir)
        logger.info(f"Saved to {output_dir}")

    return result_ds


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO dataset")
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20000,
        help="Number of samples to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()

    prepare_grpo_dataset(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
