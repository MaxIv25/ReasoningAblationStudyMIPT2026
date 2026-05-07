"""
Prepare a difficulty-calibrated GRPO dataset from GSM8K + MATH.

For GRPO to work, the model needs tasks it solves 20-80% of the time.
OpenR1 problems are too hard for 0.8B (~10% accuracy), so we use easier sources:

- MATH train Level 3:  ~1500 tasks (pass@8 ~35-50% for 0.8B)
- MATH train Level 2:  ~750 tasks  (pass@8 ~50-65%)
- MATH train Level 4:  ~500 tasks  (pass@8 ~20-35%)
- GSM8K train:         ~250 tasks  (pass@8 ~75-85%, stability anchors)

Output format matches prepare_grpo_data.py:
- `prompt`: list of chat messages (conversational format)
- `solution`: ground truth answer string
"""

import argparse
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, extract_boxed_answer

logger = setup_logging("prepare_grpo_easy")


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer after #### in GSM8K format."""
    match = re.search(r"####\s*(.+)$", answer_text, re.MULTILINE)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def extract_math_answer(solution: str) -> str:
    """Extract answer from \\boxed{} in MATH dataset solutions."""
    return extract_boxed_answer(solution)


def build_prompt(problem: str) -> list:
    """Build conversational prompt matching the existing GRPO format."""
    return [
        {
            "role": "user",
            "content": (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        }
    ]


def load_gsm8k(num_samples: int = 250, seed: int = 42) -> list:
    """Load GSM8K train problems."""
    logger.info("Loading GSM8K train...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    logger.info(f"  GSM8K raw: {len(ds)} examples")

    processed = []
    for example in tqdm(ds, desc="GSM8K"):
        answer = extract_gsm8k_answer(example["answer"])
        if answer is None:
            continue
        processed.append({
            "prompt": build_prompt(example["question"]),
            "solution": answer,
            "source": "gsm8k",
        })

    # Shuffle and sample
    import random
    random.seed(seed)
    random.shuffle(processed)
    result = processed[:num_samples]
    logger.info(f"  GSM8K: {len(result)} samples selected")
    return result


def load_math_by_level(
    levels: dict,  # {level: num_samples}
    seed: int = 42,
) -> list:
    """
    Load MATH train problems filtered by difficulty level.

    Args:
        levels: dict mapping level (int) to desired sample count
        seed: random seed
    """
    logger.info("Loading MATH train (hendrycks/competition_math)...")
    try:
        ds = load_dataset("hendrycks/competition_math", split="train")
    except Exception:
        # Alternative name
        logger.info("Trying alternative dataset name: lighteval/MATH...")
        ds = load_dataset("lighteval/MATH", "all", split="train")

    logger.info(f"  MATH raw: {len(ds)} examples")

    # Group by level
    by_level = {}
    for example in ds:
        level_str = example.get("level", "")
        # Parse "Level X" format
        match = re.search(r"(\d+)", str(level_str))
        if match:
            level = int(match.group(1))
        else:
            continue
        by_level.setdefault(level, []).append(example)

    logger.info(f"  MATH by level: " + ", ".join(
        f"L{k}: {len(v)}" for k, v in sorted(by_level.items())
    ))

    import random
    random.seed(seed)
    processed = []

    for level, count in sorted(levels.items()):
        available = by_level.get(level, [])
        if not available:
            logger.warning(f"  No MATH Level {level} examples found!")
            continue

        level_processed = []
        for example in available:
            answer = extract_math_answer(example.get("solution", ""))
            if answer is None:
                continue
            level_processed.append({
                "prompt": build_prompt(example["problem"]),
                "solution": answer,
                "source": f"math_L{level}",
            })

        random.shuffle(level_processed)
        selected = level_processed[:count]
        processed.extend(selected)
        logger.info(f"  MATH Level {level}: {len(selected)}/{count} "
                     f"(available: {len(level_processed)})")

    return processed


def prepare_grpo_easy(
    total_samples: int = 3000,
    seed: int = 42,
    output_dir: str = None,
) -> Dataset:
    """
    Build a difficulty-calibrated GRPO dataset.

    Distribution (targeting 3000 total):
    - MATH L3: 50% (1500) — sweet spot for 0.8B model
    - MATH L2: 25% (750) — moderately easy
    - MATH L4: 17% (500) — challenging
    - GSM8K:    8% (250) — easy anchors
    """
    # Calculate split sizes
    n_math3 = int(total_samples * 0.50)  # 1500
    n_math2 = int(total_samples * 0.25)  # 750
    n_math4 = int(total_samples * 0.17)  # 510
    n_gsm8k = total_samples - n_math3 - n_math2 - n_math4  # 240

    logger.info(f"Target: {total_samples} samples")
    logger.info(f"  MATH L3: {n_math3}, L2: {n_math2}, L4: {n_math4}, GSM8K: {n_gsm8k}")

    # Load sources
    gsm8k_data = load_gsm8k(num_samples=n_gsm8k, seed=seed)
    math_data = load_math_by_level(
        levels={2: n_math2, 3: n_math3, 4: n_math4},
        seed=seed,
    )

    all_data = gsm8k_data + math_data
    logger.info(f"Total collected: {len(all_data)}")

    # Shuffle
    import random
    random.seed(seed)
    random.shuffle(all_data)

    # Remove source column (not needed by trainer)
    # Log source distribution first
    from collections import Counter
    sources = Counter(d["source"] for d in all_data)
    logger.info(f"Source distribution: {dict(sources)}")

    final_data = [
        {"prompt": d["prompt"], "solution": d["solution"]}
        for d in all_data
    ]

    result_ds = Dataset.from_list(final_data)
    logger.info(f"Final dataset size: {len(result_ds)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_ds.save_to_disk(output_dir)
        logger.info(f"Saved to {output_dir}")

    return result_ds


def main():
    parser = argparse.ArgumentParser(
        description="Prepare difficulty-calibrated GRPO dataset (GSM8K + MATH L2-4)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3000,
        help="Total number of samples (default: 3000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()

    prepare_grpo_easy(
        total_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
