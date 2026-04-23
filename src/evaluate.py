"""
Evaluation script for GSM8K and MATH-500 benchmarks.

Uses vLLM for fast batch inference. Supports:
- pass@1 (greedy decoding)
- maj@K (majority voting with sampling)
- Format compliance (% of responses with \\boxed{})
- Average response length
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    extract_boxed_answer,
    load_config,
    save_results,
    setup_logging,
    Timer,
    get_gpu_memory_info,
)

logger = setup_logging("evaluate")


# ──────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────

def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test split."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for item in ds:
        # GSM8K answer format: "... #### 42"
        answer_text = item["answer"].split("####")[-1].strip()
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        answer_text = answer_text.replace(",", "")
        examples.append({
            "question": item["question"],
            "answer": answer_text,
            "source": "gsm8k",
        })
    return examples


def load_math500() -> list[dict]:
    """Load MATH-500 benchmark."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = []
    for item in ds:
        # MATH-500: answer is in 'answer' field
        examples.append({
            "question": item["problem"],
            "answer": item["answer"],
            "source": "math500",
        })
    return examples


# ──────────────────────────────────────────────────────────────
# Answer verification
# ──────────────────────────────────────────────────────────────

def verify_answer(predicted: str | None, ground_truth: str) -> bool:
    """
    Verify if predicted answer matches ground truth.
    
    Uses math_verify if available, otherwise falls back to string comparison.
    """
    if predicted is None:
        return False

    # Normalize
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Try math_verify first
    try:
        from math_verify import parse, verify
        return verify(parse(ground_truth), parse(predicted))
    except ImportError:
        pass
    except Exception:
        # math_verify can throw on unparseable expressions
        pass

    # Fallback: string-based comparison
    # Remove LaTeX formatting
    def normalize(s):
        s = s.replace("\\$", "").replace("$", "")
        s = s.replace("\\,", "").replace(",", "")
        s = s.replace(" ", "")
        s = s.strip()
        # Try to evaluate as number
        try:
            return float(s)
        except ValueError:
            return s.lower()

    return normalize(predicted) == normalize(ground_truth)


# ──────────────────────────────────────────────────────────────
# Evaluation with vLLM
# ──────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    benchmarks: list[str] = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    num_samples: int = 1,
    prompt_template: str = None,
    use_chat_template: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> dict:
    """
    Evaluate a model on GSM8K and/or MATH-500.
    
    Args:
        model_path: HuggingFace model ID or local path
        benchmarks: List of benchmarks ["gsm8k", "math500"]
        max_new_tokens: Max tokens to generate
        temperature: Qwen3 recommends 0.6 for thinking mode (DO NOT use greedy)
        top_p: Top-p sampling (Qwen3 default: 0.95)
        top_k: Top-k sampling (Qwen3 default: 20)
        num_samples: Number of samples per question (1 for pass@1, 8 for maj@8)
        prompt_template: Template for the prompt
        use_chat_template: If True, wrap prompt in Qwen3 chat template with
            <think> prefix. Use True for fine-tuned models, False for base model.
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
    
    Returns:
        Dict with results per benchmark
    """
    from vllm import LLM, SamplingParams

    if benchmarks is None:
        benchmarks = ["gsm8k", "math500"]

    if prompt_template is None:
        prompt_template = (
            "Please reason step by step, and put your final answer "
            "within \\boxed{}."
        )

    # Load datasets
    all_examples = {}
    if "gsm8k" in benchmarks:
        all_examples["gsm8k"] = load_gsm8k_test()
        logger.info(f"Loaded GSM8K test: {len(all_examples['gsm8k'])} examples")
    if "math500" in benchmarks:
        all_examples["math500"] = load_math500()
        logger.info(f"Loaded MATH-500: {len(all_examples['math500'])} examples")

    # Load model with vLLM
    logger.info(f"Loading model: {model_path}")
    with Timer("Model loading"):
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_new_tokens + 4096,  # input (up to 4K) + output
        )

    # Qwen3 best practices: t=0.6, top_p=0.95, top_k=20 for thinking mode.
    # DO NOT use greedy (t=0) — causes performance degradation and repetitions.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        n=num_samples,
    )

    logger.info(f"Chat template: {use_chat_template}")

    results = {}
    for bench_name, examples in all_examples.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {bench_name} ({len(examples)} examples)")
        logger.info(f"{'='*60}")

        # Build prompts
        prompts = []
        for ex in examples:
            question = f"{ex['question']}\n\n{prompt_template}"

            if use_chat_template:
                # For fine-tuned models: use chat template + <think> prefix
                # so model generates in the format it was trained on.
                # The model will continue from "<think>\n" and produce
                # reasoning + </think> + \boxed{answer}
                prompt = (
                    f"<|im_start|>user\n{question}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n"
                )
            else:
                # For base model: just raw text, no chat template
                prompt = question

            prompts.append(prompt)

        # Generate
        with Timer(f"{bench_name} generation"):
            outputs = llm.generate(prompts, sampling_params)

        # Evaluate
        correct = 0
        total = len(examples)
        format_ok = 0
        total_length = 0
        details = []

        for ex, output in zip(examples, outputs):
            if num_samples == 1:
                # pass@1: greedy
                generated = output.outputs[0].text
                predicted = extract_boxed_answer(generated)
                is_correct = verify_answer(predicted, ex["answer"])

                if is_correct:
                    correct += 1
                if predicted is not None:
                    format_ok += 1
                total_length += len(generated.split())

                details.append({
                    "question": ex["question"][:100] + "...",
                    "ground_truth": ex["answer"],
                    "predicted": predicted,
                    "correct": is_correct,
                    "response_length": len(generated.split()),
                })
            else:
                # maj@K: majority voting
                answers = []
                for out in output.outputs:
                    generated = out.text
                    predicted = extract_boxed_answer(generated)
                    if predicted is not None:
                        answers.append(predicted)
                        format_ok += 1
                    total_length += len(generated.split())

                if answers:
                    # Majority vote
                    counter = Counter(answers)
                    majority_answer = counter.most_common(1)[0][0]
                    is_correct = verify_answer(majority_answer, ex["answer"])
                else:
                    is_correct = False

                if is_correct:
                    correct += 1

                details.append({
                    "question": ex["question"][:100] + "...",
                    "ground_truth": ex["answer"],
                    "majority_answer": majority_answer if answers else None,
                    "num_valid": len(answers),
                    "correct": is_correct,
                })

        accuracy = correct / total if total > 0 else 0.0
        format_rate = format_ok / (total * num_samples) if total > 0 else 0.0
        avg_length = total_length / (total * num_samples) if total > 0 else 0.0

        metric_key = "pass@1" if num_samples == 1 else f"maj@{num_samples}"

        results[bench_name] = {
            "accuracy": round(accuracy * 100, 2),
            "metric": metric_key,
            "correct": correct,
            "total": total,
            "format_compliance": round(format_rate * 100, 2),
            "avg_response_length": round(avg_length, 1),
            "details": details[:20],  # Save first 20 for inspection
        }

        logger.info(f"\n{bench_name} Results:")
        logger.info(f"  {metric_key}: {accuracy*100:.2f}% ({correct}/{total})")
        logger.info(f"  Format compliance: {format_rate*100:.1f}%")
        logger.info(f"  Avg response length: {avg_length:.0f} words")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on math benchmarks")
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (Qwen3: 0.6 for thinking mode, DO NOT use 0.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling (Qwen3 default: 0.95)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Top-k sampling (Qwen3 default: 20)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Number of samples per question (1=pass@1, 8=maj@8)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this evaluation run (used in output filename)",
    )
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Use chat template with <think> prefix (for fine-tuned models). "
             "Don't use for base model baseline.",
    )

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        benchmarks=args.benchmarks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_samples=args.num_samples,
        use_chat_template=args.chat_template,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
    )

    # Add model info
    results["model"] = args.model
    results["gpu_info"] = get_gpu_memory_info()

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

