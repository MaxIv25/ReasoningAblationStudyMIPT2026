"""
Generate reasoning traces from a teacher model using vLLM.

For Exp 1.2: Generate traces from Qwen3.5-35B-A3B on the same prompts 
as OpenR1-Math to compare teacher models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, save_results, Timer

logger = setup_logging("generate_traces")


def generate_traces(
    teacher_model: str,
    prompts: list[str],
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 100,
) -> list[str]:
    """
    Generate reasoning traces from a teacher model.
    
    Args:
        teacher_model: Model path or HuggingFace ID
        prompts: List of math problem prompts
        max_new_tokens: Max tokens in generated trace
        temperature: Sampling temperature
        top_p: Top-p sampling
        tensor_parallel_size: TP for vLLM
        gpu_memory_utilization: GPU memory fraction
        batch_size: Batch size for generation
    
    Returns:
        List of generated traces
    """
    from vllm import LLM, SamplingParams

    logger.info(f"Loading teacher model: {teacher_model}")
    with Timer("Teacher model loading"):
        llm = LLM(
            model=teacher_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_new_tokens + 1024,
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    # Format prompts with thinking instruction
    system_prompt = (
        "You are a helpful math assistant. Think step by step inside "
        "<think>...</think> tags, then give your final answer using \\boxed{}."
    )

    formatted_prompts = []
    for prompt in prompts:
        formatted = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        formatted_prompts.append(formatted)

    # Generate in batches
    all_traces = []
    logger.info(f"Generating {len(prompts)} traces...")
    
    with Timer("Trace generation"):
        outputs = llm.generate(formatted_prompts, sampling_params)

    for output in outputs:
        trace = output.outputs[0].text
        all_traces.append(trace)

    logger.info(f"Generated {len(all_traces)} traces")
    logger.info(
        f"Average trace length: "
        f"{sum(len(t.split()) for t in all_traces) / len(all_traces):.0f} words"
    )

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return all_traces


def generate_and_save(
    teacher_model: str,
    output_path: str,
    num_samples: int = 20000,
    config: dict = None,
):
    """
    Full pipeline: load prompts from OpenR1-Math, generate traces, save.
    
    Args:
        teacher_model: Teacher model path
        output_path: Where to save the generated dataset
        num_samples: Number of prompts to use
        config: Optional config dict
    """
    # Load prompts from OpenR1-Math
    logger.info("Loading prompts from OpenR1-Math-220k...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="default")

    # Sample
    if num_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(num_samples))

    prompts = [ex["problem"] for ex in ds]
    logger.info(f"Using {len(prompts)} prompts")

    # Generate
    traces = generate_traces(
        teacher_model=teacher_model,
        prompts=prompts,
        max_new_tokens=config.get("model", {}).get("max_seq_len", 4096) if config else 4096,
    )

    # Build dataset
    data = []
    for prompt, trace in zip(prompts, traces):
        data.append({
            "problem": prompt,
            "solution": trace,
            "teacher": teacher_model,
        })

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(data)} generated traces to {output_path}")

    # Also save as HuggingFace Dataset
    hf_ds = Dataset.from_list(data)
    hf_output = output_path.replace(".json", "_hf")
    hf_ds.save_to_disk(hf_output)
    logger.info(f"Saved HF dataset to {hf_output}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces from teacher")
    parser.add_argument(
        "--teacher", type=str, required=True,
        help="Teacher model path or HuggingFace ID",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=20000,
        help="Number of prompts to generate traces for",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
    )

    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    generate_and_save(
        teacher_model=args.teacher,
        output_path=args.output,
        num_samples=args.num_samples,
        config=config,
    )


if __name__ == "__main__":
    main()
