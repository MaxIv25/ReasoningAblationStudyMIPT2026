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

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_prime_data.py \
        --model outputs/exp2_1_full_ft \
        --source full \
        --output data/grpo_prime_calibrated \
        --num-samples 1500

Usage (multi-GPU — 4 GPUs, ~4x faster):
    # Launch shards in parallel:
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/calibrate_prime_data.py \
            --model outputs/exp2_1_full_ft \
            --source full \
            --output data/grpo_prime_calibrated \
            --shard-id $i --num-shards 4 &
    done
    wait
    # Merge shards:
    python scripts/calibrate_prime_data.py \
        --merge data/grpo_prime_calibrated \
        --num-samples 1500
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

def merge_shards(output_dir: str, num_samples: int, min_rate: float, max_rate: float):
    """Merge shard results into final calibrated dataset."""
    import glob
    import numpy as np

    shard_files = sorted(glob.glob(os.path.join(output_dir, "shard_*_of_*.json")))
    if not shard_files:
        logger.error(f"No shard files found in {output_dir}/shard_*_of_*.json")
        return

    logger.info(f"Found {len(shard_files)} shard files")

    # Combine all shard data
    all_raw_data = []
    all_pass_rates = []
    for sf in shard_files:
        with open(sf) as f:
            shard = json.load(f)
        all_raw_data.extend(shard["raw_data"])
        all_pass_rates.extend(shard["pass_rates"])
        logger.info(f"  {os.path.basename(sf)}: {len(shard['pass_rates'])} prompts")

    rates = np.array(all_pass_rates)
    logger.info(f"\nTotal: {len(rates)} prompts probed")

    # Statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Per-prompt pass rate statistics:")
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

    p_uniform = rates**8 + (1 - rates)**8
    logger.info(f"\n  With G=8 generations:")
    logger.info(f"    Estimated P(uniform group): {p_uniform.mean():.3f}")
    logger.info(f"    Prompts with P(uniform)>0.5: {int(np.sum(p_uniform > 0.5))} / {len(rates)}")

    # Filter
    mask = (rates >= min_rate) & (rates <= max_rate)
    surviving_indices = np.where(mask)[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"Calibration filter [{min_rate}, {max_rate}]:")
    logger.info(f"  Survived: {len(surviving_indices)} / {len(rates)} "
                f"({len(surviving_indices)/len(rates)*100:.1f}%)")

    if len(surviving_indices) == 0:
        logger.error("No prompts survived! Widen the filter range.")
        return

    surviving_rates = rates[surviving_indices]
    difficulty_score = np.abs(surviving_rates - 0.5)
    sorted_order = np.argsort(difficulty_score)

    n_select = min(num_samples, len(surviving_indices))
    selected_order = sorted_order[:n_select]
    selected_indices = surviving_indices[selected_order].tolist()
    selected_rates = rates[selected_indices]

    logger.info(f"  Selected: {n_select} prompts (closest to p=0.5)")
    logger.info(f"  Selected pass rate: mean={selected_rates.mean():.3f}, "
                f"std={selected_rates.std():.3f}")

    p_uniform_selected = selected_rates**8 + (1 - selected_rates)**8
    logger.info(f"  Expected PRIME filter rate: ~{p_uniform_selected.mean()*100:.1f}% "
                f"(was ~{p_uniform.mean()*100:.1f}% before)")

    from collections import Counter
    selected_sources = Counter(all_raw_data[i].get("source", "?") for i in selected_indices)
    logger.info(f"  Source distribution: {dict(selected_sources)}")

    # Save dataset
    final_data = [
        {"prompt": all_raw_data[i]["prompt"], "solution": all_raw_data[i]["solution"]}
        for i in selected_indices
    ]
    import random
    random.seed(42)
    random.shuffle(final_data)

    result_ds = Dataset.from_list(final_data)
    result_ds.save_to_disk(output_dir)
    logger.info(f"\nSaved calibrated dataset ({len(result_ds)} prompts) to {output_dir}")

    # Save probe results
    probe_results = {
        "num_shards": len(shard_files),
        "total_probed": len(rates),
        "survived_filter": int(len(surviving_indices)),
        "selected": n_select,
        "filter_range": [min_rate, max_rate],
        "stats": {
            "full_pool_mean": float(rates.mean()),
            "full_pool_std": float(rates.std()),
            "selected_mean": float(selected_rates.mean()),
            "selected_std": float(selected_rates.std()),
            "estimated_prime_filter_before": float(p_uniform.mean()),
            "estimated_prime_filter_after": float(p_uniform_selected.mean()),
        },
        "per_prompt_pass_rates": [float(r) for r in all_pass_rates],
    }
    probe_path = os.path.join(output_dir, "probe_results.json")
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    logger.info(f"Saved probe results to {probe_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate GRPO dataset for PRIME (per-prompt accuracy probing)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="SFT model path (e.g. outputs/exp2_1_full_ft)",
    )

    # Data source: either --source full (load everything) or --data-dir (existing dataset)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source", type=str, choices=["full"],
        help="Load full pool: MATH L1-L5 + GSM8K train (~12,500 problems)",
    )
    source_group.add_argument(
        "--data-dir", type=str,
        help="Input dataset directory (e.g. data/grpo_easy_1500)",
    )

    parser.add_argument(
        "--output", type=str, default=None,
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

    # Sharding for multi-GPU
    parser.add_argument(
        "--shard-id", type=int, default=None,
        help="Shard index (0-based) for multi-GPU parallelism",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards (= number of GPUs)",
    )

    # Merge mode
    parser.add_argument(
        "--merge", type=str, default=None,
        help="Merge shard results from this directory (no GPU needed)",
    )

    args = parser.parse_args()

    # ── Merge mode ──
    if args.merge:
        merge_shards(args.merge, args.num_samples, args.min_rate, args.max_rate)
        return

    if not args.model:
        parser.error("--model is required (unless using --merge)")
    if not args.source and not args.data_dir:
        parser.error("--source or --data-dir is required (unless using --merge)")
    if not args.output:
        parser.error("--output is required (unless using --merge)")

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

    # ── Sharding ──
    if args.shard_id is not None and args.num_shards is not None:
        shard_indices = list(range(args.shard_id, len(raw_data), args.num_shards))
        shard_data = [raw_data[i] for i in shard_indices]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: "
                    f"{len(shard_data)} prompts (of {len(raw_data)} total)")
    else:
        shard_data = raw_data
        shard_indices = list(range(len(raw_data)))

    # ── Build vLLM prompts ──
    prompts = []
    solutions = []
    for item in shard_data:
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

    # ── If sharding, save shard results and exit ──
    if args.shard_id is not None:
        os.makedirs(args.output, exist_ok=True)
        shard_result = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "indices": shard_indices,
            "pass_rates": pass_rates,
            "raw_data": [
                {"prompt": d["prompt"], "solution": d["solution"], "source": d.get("source", "?")}
                for d in shard_data
            ],
        }
        shard_path = os.path.join(args.output, f"shard_{args.shard_id}_of_{args.num_shards}.json")
        with open(shard_path, "w") as f:
            json.dump(shard_result, f)
        logger.info(f"Shard {args.shard_id} done. Saved to {shard_path}")
        logger.info(f"After all shards finish, run: python scripts/calibrate_prime_data.py "
                    f"--merge {args.output} --num-samples {args.num_samples}")
        return

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
