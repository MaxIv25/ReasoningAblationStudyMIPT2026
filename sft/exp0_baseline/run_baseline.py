#!/usr/bin/env python3
"""
Phase 0: Baseline evaluation.

Evaluates raw Qwen3.5-4B-Base (without any fine-tuning) on:
- GSM8K test (1.3K examples)
- MATH-500

Expected results:
- GSM8K: ~20-40% (base model, no instruct)
- MATH-500: ~5-15%

Usage:
    python sft/exp0_baseline/run_baseline.py
    python sft/exp0_baseline/run_baseline.py --model /path/to/local/model
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model
from src.utils import save_results, setup_logging, get_gpu_memory_info

logger = setup_logging("baseline")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Baseline evaluation")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-4B-Base",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "sft" / "exp0_baseline" / "results"),
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.9)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 0: BASELINE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU info: {get_gpu_memory_info()}")

    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        benchmarks=["gsm8k", "math500"],
        max_new_tokens=4096,
        temperature=0.6,      # Qwen3 recommended for thinking mode
        top_p=0.95,
        top_k=20,
        num_samples=1,        # pass@1
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
    )

    # Add experiment metadata
    results["experiment"] = {
        "phase": 0,
        "name": "baseline",
        "description": "Raw Qwen3.5-4B-Base without fine-tuning",
        "model": args.model,
    }

    # Save
    output_path = os.path.join(args.output_dir, "baseline_results.json")
    save_results(results, output_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    for bench in ["gsm8k", "math500"]:
        if bench in results and isinstance(results[bench], dict):
            r = results[bench]
            logger.info(
                f"  {bench}: {r['accuracy']:.2f}% "
                f"({r['correct']}/{r['total']}) "
                f"[format: {r['format_compliance']:.1f}%]"
            )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

