"""
Calibrate GRPO dataset for PRIME by probing per-prompt accuracy.

PRIME's online filter discards prompts where all G completions are
correct or all are wrong. With bimodal per-prompt difficulty (most
prompts are either trivially easy or impossible), 90%+ get filtered.

This script:
1. Loads the FULL source pool (MATH L1-L5 + GSM8K train ≈ 12,500 problems)
   OR an existing dataset from --data-dir
2. Generates N completions per prompt using the SFT model (vLLM)
3. Computes empirical pass rate per prompt
4. Filters to keep only prompts with pass rate in [min_rate, max_rate]
5. Selects --num-samples prompts from the survivors
6. Saves the calibrated dataset

Usage (recommended — probe full pool):
    CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_prime_data.py \
        --model outputs/exp2_1_full_ft \
        --source full \
        --output data/grpo_prime_calibrated \
        --num-samples 1500

Usage (probe existing dataset):
    CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_prime_data.py \
        --model outputs/exp2_1_full_ft \
        --data-dir data/grpo_easy_1500 \
        --output data/grpo_prime_calibrated
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, extract_boxed_answer, verify_answer

logger = setup_logging("calibrate_prime_data")


# ──────────────────────────────────────────────────────────────
# Dataset loading (same logic as prepare_grpo_easy.py)
# ──────────────────────────────────────────────────────────────

def build_prompt(problem: str) -> list:
    """Build conversational prompt matching the GRPO format."""
    return [
        {
            "role": "user",
            "content": (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        },
        {
            "role": "assistant",
            "content": "<think>\n",
        },
    ]


def load_full_pool() -> list[dict]:
    """Load ALL available problems: MATH L1-L5 + GSM8K train."""
    all_data = []

    # GSM8K train
    logger.info("Loading GSM8K train...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for example in tqdm(ds, desc="GSM8K"):
        match = re.search(r"####\s*(.+)$", example["answer"], re.MULTILINE)
        if match:
            answer = match.group(1).strip().replace(",", "")
            all_data.append({
                "prompt": build_prompt(example["question"]),
                "solution": answer,
                "source": "gsm8k",
            })
    logger.info(f"  GSM8K: {sum(1 for d in all_data if d['source']=='gsm8k')} problems")

    # MATH train (all levels, all subjects)
    logger.info("Loading MATH train (all levels)...")
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra",
        "precalculus",
    ]
    for subj in subjects:
        try:
            ds_subj = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            for example in ds_subj:
                answer = extract_boxed_answer(example.get("solution", ""))
                if answer is None:
                    continue
                level_str = example.get("level", "")
                level_match = re.search(r"(\d+)", str(level_str))
                level = int(level_match.group(1)) if level_match else 0
                all_data.append({
                    "prompt": build_prompt(example["problem"]),
                    "solution": answer,
                    "source": f"math_L{level}",
                })
            logger.info(f"  {subj}: {len(ds_subj)} loaded")
        except Exception as e:
            logger.warning(f"  {subj}: failed — {e}")

    # Log distribution
    from collections import Counter
    sources = Counter(d["source"] for d in all_data)
    logger.info(f"Total pool: {len(all_data)} problems")
    for src, cnt in sorted(sources.items()):
        logger.info(f"  {src}: {cnt}")

    return all_data


# ──────────────────────────────────────────────────────────────
# Probing
# ──────────────────────────────────────────────────────────────

def probe_dataset(
    model_path: str,
    prompts: list[str],
    solutions: list[str],
    num_probes: int = 16,
    max_new_tokens: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    gpu_memory_utilization: float = 0.9,
) -> list[float]:
    """
    Generate num_probes completions per prompt and compute pass rate.
    """
    from vllm import LLM, SamplingParams

    logger.info(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_new_tokens + 4096,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        n=num_probes,
    )

    logger.info(f"Probing {len(prompts)} prompts × {num_probes} samples = "
                f"{len(prompts) * num_probes} completions...")

    # Generate (vLLM handles batching internally)
    outputs = llm.generate(prompts, sampling_params)

    # Compute pass rates
    pass_rates = []
    for solution, output in zip(solutions, outputs):
        correct = 0
        total = len(output.outputs)
        for out in output.outputs:
            predicted = extract_boxed_answer(out.text)
            if predicted is not None and verify_answer(predicted, solution):
                correct += 1
        rate = correct / total if total > 0 else 0.0
        pass_rates.append(rate)

    # Cleanup
    del llm
    import torch
    torch.cuda.empty_cache()

    return pass_rates


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate GRPO dataset for PRIME (per-prompt accuracy probing)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="SFT model path (e.g. outputs/exp2_1_full_ft)",
    )

    # Data source: either --source full (load everything) or --data-dir (existing dataset)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source", type=str, choices=["full"],
        help="Load full pool: MATH L1-L5 + GSM8K train (~12,500 problems)",
    )
    source_group.add_argument(
        "--data-dir", type=str,
        help="Input dataset directory (e.g. data/grpo_easy_1500)",
    )

    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for calibrated dataset",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1500,
        help="Number of prompts to keep after calibration (default: 1500)",
    )
    parser.add_argument(
        "--num-probes", type=int, default=16,
        help="Completions per prompt for probing (default: 16)",
    )
    parser.add_argument(
        "--min-rate", type=float, default=0.15,
        help="Minimum pass rate to keep (default: 0.15)",
    )
    parser.add_argument(
        "--max-rate", type=float, default=0.85,
        help="Maximum pass rate to keep (default: 0.85)",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16384,
        help="Max tokens per completion (default: 16384)",
    )

    args = parser.parse_args()

    # ── Load data ──
    if args.source == "full":
        raw_data = load_full_pool()
    else:
        logger.info(f"Loading dataset from {args.data_dir}")
        dataset = load_from_disk(args.data_dir)
        raw_data = [
            {
                "prompt": example["prompt"],
                "solution": example["solution"],
                "source": "existing",
            }
            for example in dataset
        ]
    logger.info(f"Loaded {len(raw_data)} problems for probing")

    # ── Build vLLM prompts ──
    prompts = []
    solutions = []
    for item in raw_data:
        messages = item["prompt"]
        user_content = messages[0]["content"]
        prompt = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        prompts.append(prompt)
        solutions.append(item["solution"])

    # ── Probe ──
    pass_rates = probe_dataset(
        model_path=args.model,
        prompts=prompts,
        solutions=solutions,
        num_probes=args.num_probes,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_mem,
    )

    # ── Statistics ──
    import numpy as np
    rates = np.array(pass_rates)
    logger.info(f"\n{'='*60}")
    logger.info(f"Per-prompt pass rate statistics ({len(rates)} prompts):")
    logger.info(f"  Mean:   {rates.mean():.3f}")
    logger.info(f"  Median: {np.median(rates):.3f}")
    logger.info(f"  Std:    {rates.std():.3f}")

    # Distribution histogram
    buckets = [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.01]
    logger.info(f"\n  Pass rate distribution:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        count = int(np.sum((rates >= lo) & (rates < hi)))
        pct = count / len(rates) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"    [{lo:.2f}, {hi:.2f})  {count:5d} ({pct:5.1f}%) {bar}")

    # Estimated PRIME filter rate with G=8
    p_uniform = rates**8 + (1 - rates)**8
    logger.info(f"\n  With G=8 generations:")
    logger.info(f"    Estimated P(uniform group): {p_uniform.mean():.3f}")
    logger.info(f"    Prompts with P(uniform)>0.5: {int(np.sum(p_uniform > 0.5))} / {len(rates)}")

    # ── Filter ──
    mask = (rates >= args.min_rate) & (rates <= args.max_rate)
    surviving_indices = np.where(mask)[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"Calibration filter [{args.min_rate}, {args.max_rate}]:")
    logger.info(f"  Survived: {len(surviving_indices)} / {len(rates)} "
                f"({len(surviving_indices)/len(rates)*100:.1f}%)")

    if len(surviving_indices) == 0:
        logger.error("No prompts survived! Widen the filter range.")
        return

    # Sort by proximity to 0.5 (ideal difficulty) and take top N
    surviving_rates = rates[surviving_indices]
    difficulty_score = np.abs(surviving_rates - 0.5)  # 0 = ideal
    sorted_order = np.argsort(difficulty_score)

    n_select = min(args.num_samples, len(surviving_indices))
    selected_order = sorted_order[:n_select]
    selected_indices = surviving_indices[selected_order].tolist()
    selected_rates = rates[selected_indices]

    logger.info(f"  Selected: {n_select} prompts (closest to p=0.5)")
    logger.info(f"  Selected pass rate: mean={selected_rates.mean():.3f}, "
                f"std={selected_rates.std():.3f}")

    # Estimated PRIME filter rate after calibration
    p_uniform_selected = selected_rates**8 + (1 - selected_rates)**8
    logger.info(f"  Expected PRIME filter rate: ~{p_uniform_selected.mean()*100:.1f}% "
                f"(was ~{p_uniform.mean()*100:.1f}% before)")

    # Source distribution of selected
    from collections import Counter
    selected_sources = Counter(raw_data[i]["source"] for i in selected_indices)
    logger.info(f"  Source distribution: {dict(selected_sources)}")

    # ── Save ──
    final_data = [
        {"prompt": raw_data[i]["prompt"], "solution": raw_data[i]["solution"]}
        for i in selected_indices
    ]

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(final_data)

    result_ds = Dataset.from_list(final_data)
    os.makedirs(args.output, exist_ok=True)
    result_ds.save_to_disk(args.output)
    logger.info(f"\nSaved calibrated dataset ({len(result_ds)} prompts) to {args.output}")

    # Save probe results for analysis / poster
    probe_results = {
        "model": args.model,
        "source": args.source or args.data_dir,
        "num_probes": args.num_probes,
        "total_probed": len(rates),
        "survived_filter": int(len(surviving_indices)),
        "selected": n_select,
        "filter_range": [args.min_rate, args.max_rate],
        "stats": {
            "full_pool_mean": float(rates.mean()),
            "full_pool_std": float(rates.std()),
            "selected_mean": float(selected_rates.mean()),
            "selected_std": float(selected_rates.std()),
            "estimated_prime_filter_before": float(p_uniform.mean()),
            "estimated_prime_filter_after": float(p_uniform_selected.mean()),
        },
        "per_prompt_pass_rates": [float(r) for r in pass_rates],
    }
    probe_path = os.path.join(args.output, "probe_results.json")
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    logger.info(f"Saved probe results to {probe_path}")


if __name__ == "__main__":
    main()
