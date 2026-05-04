"""
GRPO training script for reasoning ablation study.

Supports three loss variants via GRPOConfig.loss_type:
- "grpo"    — Vanilla GRPO (sequence-level normalization)
- "dapo"    — DAPO (token-level normalization + asymmetric clipping)
- "dr_grpo" — Dr. GRPO (constant normalization + no reward std scaling)

Reward functions:
- accuracy_reward: math_verify-based correctness checking (0.0 / 1.0)
- format_reward: checks for <think>...</think> + \\boxed{} format (0.0 / 0.5 / 1.0)

Uses vLLM colocate mode for fast generation on single GPU.
"""

import argparse
import os
import re
import sys

# MUST be set before importing trl/transformers which may import fla.
# TileLang backend crashes on backward pass for GDN layers.
os.environ.setdefault("FLA_BACKEND", "triton")

from pathlib import Path

import torch
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, get_gpu_memory_info, extract_boxed_answer

logger = setup_logging("train_grpo")


# ──────────────────────────────────────────────────────────────
# Reward functions
# ──────────────────────────────────────────────────────────────

def accuracy_reward(completions, solution, log_metric=None, **kwargs):
    """
    Check if the model's answer matches the ground truth.

    Uses math_verify for robust LaTeX comparison, with string fallback.
    Reward: 1.0 (correct) / 0.0 (incorrect).

    Args:
        completions: list of list of message dicts (conversational format)
        solution: list of ground truth answer strings
    """
    rewards = []
    num_correct = 0

    for completion, sol in zip(completions, solution):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        predicted = extract_boxed_answer(content)

        if predicted is not None and _verify_answer(predicted, sol):
            rewards.append(1.0)
            num_correct += 1
        else:
            rewards.append(0.0)

    # Log accuracy as a custom metric
    if log_metric and len(rewards) > 0:
        log_metric("accuracy", num_correct / len(rewards))

    return rewards


def format_reward(completions, log_metric=None, **kwargs):
    """
    Check if the completion follows the expected reasoning format.

    Checks for:
    - <think>...</think> block (reasoning trace)
    - \\boxed{} (final answer)

    Reward: 1.0 (both), 0.5 (one of two), 0.0 (neither).
    """
    rewards = []
    format_ok_count = 0

    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0

        has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))
        has_boxed = "\\boxed{" in content

        if has_think:
            score += 0.5
        if has_boxed:
            score += 0.5

        if score >= 1.0:
            format_ok_count += 1

        rewards.append(score)

    if log_metric and len(rewards) > 0:
        log_metric("format_compliance", format_ok_count / len(rewards))

    return rewards


def _verify_answer(predicted: str, ground_truth: str) -> bool:
    """Verify answer using math_verify with string fallback."""
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Try math_verify first
    try:
        from math_verify import parse, verify
        return verify(parse(ground_truth), parse(predicted))
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: normalized string comparison
    def normalize(s):
        s = s.replace("\\$", "").replace("$", "")
        s = s.replace("\\,", "").replace(",", "")
        s = s.replace(" ", "").strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()

    return normalize(predicted) == normalize(ground_truth)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(config: dict, data_dir: str = None, output_dir: str = None):
    """
    Run GRPO training.

    Args:
        config: Merged configuration dict (grpo_base.yaml + experiment yaml)
        data_dir: Path to prepared GRPO dataset
        output_dir: Output directory for checkpoints
    """
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3.5-0.8B-Base")
    run_name = config.get("run_name", "grpo")
    loss_type = grpo_cfg.get("loss_type", "grpo")

    if output_dir is None:
        output_dir = f"./outputs/{run_name}"

    logger.info(f"GRPO variant: {loss_type}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPU info: {get_gpu_memory_info()}")

    # TF32 for matmul on Hopper
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Force SDPA to use flash/mem_efficient kernels, NOT the math fallback.
    # Math fallback stores full attention matrices → OOM on long sequences.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    logger.info("SDPA config: flash=True, mem_efficient=True, math=DISABLED")

    # Load dataset
    if data_dir:
        train_dataset = load_from_disk(data_dir)
        logger.info(f"Loaded dataset: {len(train_dataset)} examples")
    else:
        raise ValueError("--data-dir is required for GRPO training")

    # ── Build GRPOConfig ──────────────────────────────────────
    # Common parameters
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,

        # Loss variant
        loss_type=loss_type,

        # Reward scaling
        # Vanilla GRPO: "group" (default) — normalize by group std
        # Dr. GRPO: "none" — no std scaling to avoid difficulty bias
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),

        # Clipping
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high", None),

        # KL divergence
        # β=0.0: modern practice for reasoning tasks (no KL penalty)
        beta=grpo_cfg.get("beta", 0.0),

        # Generation
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 8192),
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=grpo_cfg.get("top_p", 1.0),
        top_k=grpo_cfg.get("top_k", 0),

        # Mask completions that hit max_completion_length
        # (avoid noisy gradients from truncated reasoning)
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", True),

        # Training
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),

        # Precision
        bf16=True,
        gradient_checkpointing=True,

        # Saving
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),

        # Logging
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="tensorboard",

        # vLLM — colocate mode on single GPU
        # vLLM handles generation, training framework handles optimization
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.5),

        # Performance
        use_liger_kernel=True,

        # Reward weights: accuracy is primary, format is secondary
        reward_weights=[1.0, 0.3],

        # Model loading kwargs
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
    )

    # Log key config differences between variants
    logger.info(f"  loss_type={grpo_args.loss_type}")
    logger.info(f"  scale_rewards={grpo_args.scale_rewards}")
    logger.info(f"  epsilon={grpo_args.epsilon}, epsilon_high={grpo_args.epsilon_high}")
    logger.info(f"  beta={grpo_args.beta}")
    logger.info(f"  num_generations={grpo_args.num_generations}")
    logger.info(f"  max_completion_length={grpo_args.max_completion_length}")
    logger.info(f"  lr={grpo_args.learning_rate}")
    logger.info(f"  FLA_BACKEND={os.environ.get('FLA_BACKEND', 'default')}")
    logger.info(f"  attn_implementation=sdpa")

    # ── Create trainer ────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_name,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward, format_reward],
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Training complete! Metrics: {metrics}")
    return trainer


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to prepared GRPO dataset (from prepare_grpo_data.py)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Load config with grpo_base.yaml inheritance
    project_root = Path(__file__).parent.parent
    base_config_path = project_root / "configs" / "grpo_base.yaml"
    config = load_config(args.config, base_config_path=str(base_config_path))

    train(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
