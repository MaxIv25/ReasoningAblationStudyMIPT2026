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
import sys
from pathlib import Path

# Avoid a shared, permission-sensitive /tmp/tvm-debug-mode-tempdirs root.
# TileLang's persistent kernel cache is independent of these compiler temp files.
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

import torch
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config,
    setup_logging,
    get_gpu_memory_info,
    extract_boxed_answer,
    verify_answer,
)
from src.rl.dpo_z_trainer import (
    DPOZGRPOTrainer,
    MemoryBoundedDPOZGRPOTrainer,
)
from src.rl.dataset_contract import (
    validate_preformatted_math_rl_dataset,
    validate_prompt_token_lengths,
)
from src.rl.memory_bounded_grpo import MemoryBoundedGRPOTrainer
from src.rl.peft_utils import build_lora_config
from src.rl.model_utils import load_text_causal_lm, patch_vllm_language_model_only
from src.rl.vllm_sync_canary import install_vllm_sync_canary

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
        content = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        predicted = extract_boxed_answer(content)

        if predicted is not None and verify_answer(predicted, sol):
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
    - </think> closing tag (opening <think> is in prompt prefill)
    - \\boxed{} (final answer)

    Reward: 1.0 (both), 0.5 (one of two), 0.0 (neither).
    """
    rewards = []
    format_ok_count = 0

    for completion in completions:
        content = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        score = 0.0

        # Check for </think> closing tag
        # Note: <think> opening tag is in the prompt prefill (assistant message),
        # so the completion only contains the closing </think> tag.
        has_think = "</think>" in content
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
    validation_cfg = config.get("validation", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3.5-0.8B")
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
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

    # Enable all SDPA backends — flash and mem_efficient preferred,
    # math as fallback for shapes they can't handle.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    logger.info("SDPA config: flash=True, mem_efficient=True, math=True (fallback)")

    # Load dataset
    if data_dir:
        train_dataset = load_from_disk(data_dir)
        validate_preformatted_math_rl_dataset(train_dataset)
        logger.info(f"Loaded dataset: {len(train_dataset)} examples")

        # Optionally truncate dataset (grpo.num_samples)
        num_samples = grpo_cfg.get("num_samples", None)
        if num_samples and num_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(num_samples))
            logger.info(
                f"Truncated dataset to {num_samples} examples (grpo.num_samples)"
            )
    else:
        raise ValueError("--data-dir is required for GRPO training")

    eval_dataset = None
    eval_data_dir = validation_cfg.get("data_dir")
    if eval_data_dir:
        eval_dataset = load_from_disk(eval_data_dir)
        validate_preformatted_math_rl_dataset(eval_dataset)
        eval_samples = int(validation_cfg.get("num_samples", len(eval_dataset)))
        eval_dataset = eval_dataset.select(
            range(min(eval_samples, len(eval_dataset)))
        )
        logger.info("Loaded validation dataset: %d examples", len(eval_dataset))

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
        # Generation batching — larger batch = fewer vLLM calls = better throughput
        generation_batch_size=grpo_cfg.get("generation_batch_size", None),
        # Training
        num_iterations=train_cfg.get("num_iterations", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        # Data loading
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=train_cfg.get("seed", 42),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(validation_cfg.get("eval_steps", 20)),
        eval_on_start=bool(validation_cfg.get("eval_on_start", False)),
        per_device_eval_batch_size=int(
            validation_cfg.get("per_device_eval_batch_size", 8)
        ),
        num_generations_eval=int(validation_cfg.get("num_generations", 1)),
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        # Saving — allow saving optimizer states for resumption, limit to 2 checkpoints
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        save_only_model=train_cfg.get("save_only_model", False),
        optim=train_cfg.get("optim", "adamw_torch"),
        # Logging
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="tensorboard",
        log_completions=grpo_cfg.get("log_completions", True),
        num_completions_to_print=grpo_cfg.get("num_completions_to_print", 8),
        # vLLM — mode from config (server or colocate)
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_mode=grpo_cfg.get("vllm_mode", "colocate"),
        vllm_enable_sleep_mode=grpo_cfg.get("vllm_enable_sleep_mode", True),
        vllm_importance_sampling_correction=grpo_cfg.get(
            "vllm_importance_sampling_correction", False
        ),
        **(
            {
                "vllm_gpu_memory_utilization": grpo_cfg.get(
                    "vllm_gpu_memory_utilization", 0.3
                ),
                "vllm_max_model_length": grpo_cfg.get(
                    "max_model_len",
                    grpo_cfg.get("max_prompt_length", 1024)
                    + grpo_cfg.get("max_completion_length", 8192),
                ),
            }
            if grpo_cfg.get("vllm_mode", "colocate") == "colocate"
            else {
                "vllm_server_port": grpo_cfg.get("vllm_server_port", 8000),
                "vllm_group_port": grpo_cfg.get("vllm_group_port", 51216),
            }
        ),
        # Performance
        use_liger_kernel=grpo_cfg.get("use_liger_kernel", True),
        reward_weights=[1.0]
        + (
            [float(grpo_cfg["format_reward_weight"])]
            if float(grpo_cfg.get("format_reward_weight", 0.0)) > 0
            else []
        ),
        # Model loading kwargs
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": attn_implementation,
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
    logger.info(f"  generation_batch_size={grpo_args.generation_batch_size}")
    logger.info(f"  steps_per_generation={grpo_args.steps_per_generation}")
    logger.info(f"  dataloader_num_workers={grpo_args.dataloader_num_workers}")
    logger.info("  attn_implementation=%s", attn_implementation)

    patch_vllm_language_model_only(
        bool(model_cfg.get("language_model_only", True)), logger=logger
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_prompt_tokens = validate_prompt_token_lengths(
        train_dataset,
        tokenizer,
        int(grpo_cfg.get("max_prompt_length", 1024)),
    )
    logger.info("Prompt token audit: max=%d", max_prompt_tokens)
    if eval_dataset is not None:
        max_eval_prompt_tokens = validate_prompt_token_lengths(
            eval_dataset,
            tokenizer,
            int(grpo_cfg.get("max_prompt_length", 1024)),
        )
        logger.info(
            "Validation: n=%d max_prompt_tokens=%d eval_steps=%d "
            "num_generations=1 sampler=(temperature=%s, top_p=%s, top_k=%s)",
            len(eval_dataset),
            max_eval_prompt_tokens,
            int(validation_cfg.get("eval_steps", 20)),
            grpo_cfg.get("temperature", 1.0),
            grpo_cfg.get("top_p", 1.0),
            grpo_cfg.get("top_k", 0),
        )
    actor_model = load_text_causal_lm(
        model_name,
        device="cpu",
        attn_implementation=attn_implementation,
        logger=logger,
    )

    # ── Create trainer ────────────────────────────────────────
    baseline_type = grpo_cfg.get("advantage_baseline", "group_mean")
    memory_bounded_loss = bool(grpo_cfg.get("memory_bounded_policy_loss", False))
    trainer_kwargs = {}
    if baseline_type == "dpo_z":
        trainer_cls = (
            MemoryBoundedDPOZGRPOTrainer if memory_bounded_loss else DPOZGRPOTrainer
        )
        if memory_bounded_loss:
            trainer_kwargs["logprob_chunk_tokens"] = int(
                grpo_cfg.get("logprob_chunk_tokens", 256)
            )
        trainer_kwargs.update(
            dpo_z_beta=grpo_cfg.get("dpo_z_beta", 0.1),
            dpo_z_leave_one_out=grpo_cfg.get("dpo_z_leave_one_out", True),
        )
    elif baseline_type == "group_mean":
        trainer_cls = MemoryBoundedGRPOTrainer if memory_bounded_loss else GRPOTrainer
        if memory_bounded_loss:
            trainer_kwargs["logprob_chunk_tokens"] = int(
                grpo_cfg.get("logprob_chunk_tokens", 256)
            )
    else:
        raise ValueError(f"Unknown GRPO advantage_baseline={baseline_type!r}")

    peft_config = build_lora_config(config.get("peft", {}), enabled_key="actor")
    logger.info(f"Advantage baseline: {baseline_type}")
    logger.info(
        f"Actor tuning: {'LoRA' if peft_config is not None else 'full parameters'}"
    )
    logger.info(
        "Policy loss: %s",
        f"token-bounded ({trainer_kwargs.get('logprob_chunk_tokens')} tokens)"
        if memory_bounded_loss
        else "TRL default",
    )

    format_reward_weight = float(grpo_cfg.get("format_reward_weight", 0.0))
    reward_funcs = [accuracy_reward]
    if format_reward_weight > 0:
        reward_funcs.append(format_reward)
    logger.info(
        "Rewards: correctness=1.0, format=%s",
        format_reward_weight if format_reward_weight > 0 else "metric-only/off",
    )

    # TRL wraps the actor with PEFT before Trainer.__init__ applies args.seed.
    # Seed here so every baseline starts from the same random LoRA subspace.
    set_seed(int(grpo_args.seed))

    trainer = trainer_cls(
        model=actor_model,
        processing_class=tokenizer,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        **trainer_kwargs,
    )
    install_vllm_sync_canary(
        trainer,
        output_dir=output_dir,
        logger=logger,
        enabled=bool(grpo_cfg.get("verify_vllm_weight_sync", True)),
        require_change_after_step=bool(
            grpo_cfg.get("require_vllm_weight_change_after_step", True)
        ),
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
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to prepared GRPO dataset (from prepare_grpo_data.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
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
