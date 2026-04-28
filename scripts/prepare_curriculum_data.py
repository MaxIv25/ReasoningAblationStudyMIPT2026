"""
Prepare datasets for curriculum learning experiment (Exp 3).

Creates 3 versions of training data with different orderings,
all sharing the SAME train/eval split (same examples, same eval set).

Critical: we split FIRST, then sort. Otherwise train_test_split
shuffles and destroys the curriculum ordering.

Usage:
    python scripts/prepare_curriculum_data.py [--num-samples 20000]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    load_openr1_math,
    format_for_sft,
    split_train_eval,
    apply_curriculum,
)
from src.utils import load_config, setup_logging, save_results

logger = setup_logging("prepare_curriculum")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare curriculum datasets for Exp 3"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of samples (default: from base.yaml = 20000)",
    )
    parser.add_argument(
        "--output-base", type=str, default="data/curriculum",
        help="Base output directory",
    )
    args = parser.parse_args()

    config = load_config(None)  # loads configs/base.yaml
    data_config = config.get("data", {})

    if args.num_samples:
        data_config["num_samples"] = args.num_samples

    # ── Step 1: Load and filter ──────────────────────────────
    logger.info("Step 1: Loading and filtering OpenR1-Math-220K...")
    ds = load_openr1_math(
        split=data_config.get("split", "train"),
        num_samples=data_config.get("num_samples", 20000),
        seed=data_config.get("seed", 42),
        filter_config=data_config.get("filter", {}),
    )
    logger.info(f"Loaded {len(ds)} examples with difficulty column")

    # ── Step 2: Format for SFT ───────────────────────────────
    logger.info("Step 2: Formatting for SFT (chat messages + <think> prefix)...")
    formatted = format_for_sft(ds)

    # ── Step 3: Split FIRST (before curriculum!) ─────────────
    # This ensures all 3 curriculum variants use the exact same
    # train/eval examples — only the ordering of train differs.
    logger.info("Step 3: Splitting into train/eval...")
    splits = split_train_eval(
        formatted,
        eval_fraction=data_config.get("eval_fraction", 0.05),
        seed=data_config.get("seed", 42),
    )

    train_ds = splits["train"]
    eval_ds = splits["eval"]
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ── Step 4: Create 3 curriculum variants ─────────────────
    orders = ["easy_to_hard", "random", "hard_to_easy"]
    logger.info(f"Step 4: Creating {len(orders)} curriculum variants...")

    for order in orders:
        logger.info(f"\n{'='*50}")
        logger.info(f"Creating variant: {order}")

        ordered_train = apply_curriculum(train_ds, order=order)

        out_dir = os.path.join(args.output_base, order)
        os.makedirs(out_dir, exist_ok=True)

        # Save train + eval
        ordered_train.save_to_disk(os.path.join(out_dir, "train"))
        eval_ds.save_to_disk(os.path.join(out_dir, "eval"))

        # Verify ordering
        diffs = ordered_train["difficulty"]
        logger.info(
            f"  First 5 difficulties: {diffs[:5]}"
        )
        logger.info(
            f"  Last  5 difficulties: {diffs[-5:]}"
        )

        # Save stats
        save_results(
            {
                "order": order,
                "train_size": len(ordered_train),
                "eval_size": len(eval_ds),
                "first_10_difficulties": diffs[:10],
                "last_10_difficulties": diffs[-10:],
                "mean_difficulty": sum(diffs) / len(diffs),
            },
            os.path.join(out_dir, "data_stats.json"),
        )

        logger.info(f"  Saved to {out_dir}/")

    logger.info(f"\n{'='*50}")
    logger.info(f"Done! Created curriculum datasets in {args.output_base}/")
    logger.info(f"  easy_to_hard/ — ascending difficulty (0.0 → 1.0)")
    logger.info(f"  random/       — shuffled (control)")
    logger.info(f"  hard_to_easy/ — descending difficulty (1.0 → 0.0)")


if __name__ == "__main__":
    main()
