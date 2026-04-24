"""
Universal SFT training script for ablation study.

Supports:
- Full Fine-Tuning
- LoRA
- DoRA (use_dora=True)
- PiSSA (init_lora_weights="pissa")

Uses TRL's SFTTrainer with PEFT integration.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, get_gpu_memory_info

logger = setup_logging("train_sft")


def get_peft_config(config: dict) -> LoraConfig | None:
    """
    Build PEFT config based on training method.
    
    Returns None for Full Fine-Tuning.
    """
    method = config.get("method", "full_ft")

    if method == "full_ft":
        return None

    lora_cfg = config.get("training", {}).get("lora", {})

    peft_kwargs = {
        "r": lora_cfg.get("r", 64),
        "lora_alpha": lora_cfg.get("lora_alpha", 128),
        "lora_dropout": lora_cfg.get("lora_dropout", 0.05),
        "target_modules": lora_cfg.get("target_modules", "all-linear"),
        "task_type": "CAUSAL_LM",
    }

    if method == "dora":
        peft_kwargs["use_dora"] = True
        logger.info("Using DoRA (Weight-Decomposed Low-Rank Adaptation)")

    if method == "pissa":
        peft_kwargs["init_lora_weights"] = "pissa"
        logger.info("Using PiSSA (SVD-initialized LoRA)")

    return LoraConfig(**peft_kwargs)


def train(config: dict, data_dir: str = None, output_dir: str = None):
    """
    Run SFT training.
    
    Args:
        config: Merged configuration dict
        data_dir: Path to prepared data (from data_utils.py)
        output_dir: Output directory for checkpoints
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    method = config.get("method", "full_ft")

    model_name = model_cfg.get("name", "Qwen/Qwen3.5-4B-Base")
    max_seq_len = model_cfg.get("max_seq_len", 4096)

    # Determine LR and epochs based on method
    if method == "full_ft":
        ft_cfg = train_cfg.get("full_ft", {})
        lr = ft_cfg.get("learning_rate", 1e-5)
        epochs = ft_cfg.get("num_train_epochs", 2)
    else:
        lora_cfg = train_cfg.get("lora", {})
        lr = lora_cfg.get("learning_rate", 2e-4)
        epochs = lora_cfg.get("num_train_epochs", 3)

    if output_dir is None:
        output_dir = f"./outputs/{method}"

    logger.info(f"Training method: {method}")
    logger.info(f"Model: {model_name}")
    logger.info(f"LR: {lr}, Epochs: {epochs}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"GPU info: {get_gpu_memory_info()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "torch_dtype": getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # For Full FT, don't use device_map (SFTTrainer handles it)
    if method == "full_ft":
        model_kwargs.pop("device_map")
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # PEFT config
    peft_config = get_peft_config(config)
    if peft_config:
        logger.info(f"PEFT config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # Load data
    if data_dir:
        train_dataset = load_from_disk(os.path.join(data_dir, "train"))
        eval_dataset = load_from_disk(os.path.join(data_dir, "eval"))
        logger.info(f"Data: train={len(train_dataset)}, eval={len(eval_dataset)}")
    else:
        # If no prepared data, use data_utils inline
        from src.data_utils import prepare_sft_data
        splits = prepare_sft_data(config)
        train_dataset = splits["train"]
        eval_dataset = splits["eval"]

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=1,  # Qwen3.5 vocab=248K → logits.float() is huge
        eval_accumulation_steps=8,     # Accumulate eval preds to avoid OOM
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        max_length=max_seq_len,  # TRL 1.x: was max_seq_length
        logging_steps=train_cfg.get("logging_steps", 50),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 500),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging: tensorboard (logs saved to output_dir/runs/)
        report_to="tensorboard",
        run_name=config.get("run_name", f"sft_{method}"),
        # Messages format auto-detected from "messages" column by TRL 1.x
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # TRL 1.x: was tokenizer
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Training complete! Metrics: {metrics}")
    return trainer


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to prepared data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["full_ft", "lora", "dora", "pissa"],
        help="Training method (overrides config)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.method:
        config["method"] = args.method

    train(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
