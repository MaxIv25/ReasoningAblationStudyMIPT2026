"""
PRIME (Process Reinforcement through Implicit Rewards) training script.

Subclasses TRL GRPOTrainer to add:
- Implicit PRM: token-level rewards via log-ratio of PRM/reference models
- Dense advantage: A_t = Return_process(t) + Return_outcome
- Online PRM update: BCE on outcome labels after each generation batch
- Configurable baselines: rloo, group_mean, truncated_mean, dpo_z

Reference: Yuan et al., 2025 — arxiv.org/abs/2502.01456
"""

import argparse
import os
import sys
from pathlib import Path

# ── FLA/TileLang workaround for H200 (Hopper) ────────────────

os.environ["FLA_TILELANG"] = "0"

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import fla.utils
fla.utils.IS_NVIDIA_HOPPER = False
import fla.ops.common.chunk_o as _chunk_o
_chunk_o.IS_NVIDIA_HOPPER = False

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, setup_logging, get_gpu_memory_info, extract_boxed_answer, verify_answer

logger = setup_logging("train_prime")


# ──────────────────────────────────────────────────────────────
# Baseline functions for advantage computation
# ──────────────────────────────────────────────────────────────

def compute_baseline_rloo(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Leave-One-Out baseline: b_i = mean(r_{j!=i}) for each sample i in group."""
    # rewards: (B*G,) -> reshape to (B, G)
    grouped = rewards.view(-1, num_generations)
    B, G = grouped.shape
    # LOO mean: (sum - r_i) / (G - 1)
    group_sum = grouped.sum(dim=1, keepdim=True)  # (B, 1)
    loo_baseline = (group_sum - grouped) / max(G - 1, 1)  # (B, G)
    return loo_baseline.reshape(-1)  # (B*G,)


def compute_baseline_group_mean(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Group mean baseline: b = mean(r_all) per group."""
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
    return mean.reshape(-1)


def compute_baseline_truncated_mean(rewards: torch.Tensor, num_generations: int,
                                     trim_frac: float = 0.125) -> torch.Tensor:
    """Truncated mean: remove top/bottom trim_frac of group, take mean of rest."""
    grouped = rewards.view(-1, num_generations)
    B, G = grouped.shape
    k = max(1, int(G * trim_frac))
    sorted_rewards, _ = grouped.sort(dim=1)
    trimmed = sorted_rewards[:, k:G-k]
    mean = trimmed.mean(dim=1, keepdim=True).expand_as(grouped)
    return mean.reshape(-1)


def compute_baseline_dpo_z(rewards: torch.Tensor, num_generations: int,
                            beta: float = 0.1) -> torch.Tensor:
    """DPO partition function estimate: b = log(mean(exp(β*r))) / β."""
    grouped = rewards.view(-1, num_generations)
    log_z = torch.logsumexp(beta * grouped, dim=1) - torch.log(
        torch.tensor(float(grouped.shape[1]), device=rewards.device)
    )
    baseline = (log_z / beta).unsqueeze(1).expand_as(grouped)
    return baseline.reshape(-1)


BASELINE_FUNCS = {
    "rloo": compute_baseline_rloo,
    "group_mean": compute_baseline_group_mean,
    "truncated_mean": compute_baseline_truncated_mean,
    "dpo_z": compute_baseline_dpo_z,
}


# ──────────────────────────────────────────────────────────────
# PrimeGRPOTrainer
# ──────────────────────────────────────────────────────────────

class PrimeGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with PRIME dense advantage estimation.

    Overrides _generate_and_score_completions to:
    1. Compute standard outcome rewards (inherited)
    2. Forward PRM + ref to get token-level process rewards
    3. Compute dense advantage: A_t = Return_process(t) + Return_outcome
    4. Update PRM online with BCE loss on outcome labels
    """

    def __init__(self, prime_cfg: dict, **kwargs):
        super().__init__(**kwargs)

        self.prime_beta = prime_cfg.get("beta", 0.1)
        self.prime_prm_lr = prime_cfg.get("prm_lr", 1e-6)
        self.prime_prm_update_epochs = prime_cfg.get("prm_update_epochs", 1)
        self.prime_baseline_type = prime_cfg.get("advantage_baseline", "rloo")
        self.prime_online_filter = prime_cfg.get("online_filter", True)
        self.prime_gamma = prime_cfg.get("gamma", 1.0)

        if self.prime_baseline_type not in BASELINE_FUNCS:
            raise ValueError(f"Unknown baseline: {self.prime_baseline_type}. "
                           f"Choose from {list(BASELINE_FUNCS.keys())}")

        self._gpu_device = self.accelerator.device

        # Load PRM — separate copy of SFT model, trainable
        # Stays on CPU (~1.6GB RAM for 0.8B), loaded to GPU on demand
        model_id = kwargs.get("model", None)
        if isinstance(model_id, str):
            prm_model_id = model_id
        else:
            from trl.trainer.utils import get_config_model_id
            prm_model_id = get_config_model_id(self.model.config)

        logger.info(f"Loading PRM from: {prm_model_id} (CPU offload)")
        self.prm_model = AutoModelForCausalLM.from_pretrained(
            prm_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )  # stays on CPU
        self.prm_model.gradient_checkpointing_enable()
        self.prm_model.train()

        # Ensure reference model exists (even with beta=0)
        # Stays on CPU (~1.6GB RAM), loaded to GPU on demand
        if self.ref_model is None:
            logger.info("Loading reference model for PRIME (CPU offload)")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                prm_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )  # stays on CPU
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            # Parent loaded ref to GPU — move to CPU for offload
            self.ref_model.to("cpu")

        # PRM optimizer (AdamW, per original PRIME/veRL)
        # States are lazily initialized on first step; managed via _move helpers
        self.prm_optimizer = torch.optim.AdamW(
            self.prm_model.parameters(),
            lr=self.prime_prm_lr,
            weight_decay=0.01,
        )

        logger.info(f"PRIME config: beta={self.prime_beta}, baseline={self.prime_baseline_type}, "
                    f"prm_lr={self.prime_prm_lr}, online_filter={self.prime_online_filter}")
        logger.info(f"PRIME memory: PRM+ref on CPU (~3.2GB RAM), GPU offload on demand")

    def _get_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        """Get per-token log-probs from a model (no grad)."""
        with torch.no_grad():
            logps, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=False,
            )
        return logps

    def _move_to_gpu(self, model):
        """Move model to GPU."""
        model.to(self._gpu_device)
        return model

    def _move_to_cpu(self, model):
        """Move model to CPU and free GPU cache."""
        model.to("cpu")
        torch.cuda.empty_cache()
        return model

    def _move_optimizer_states(self, device):
        """Move AdamW optimizer states to target device."""
        for state in self.prm_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _compute_process_rewards(self, prompt_completion_ids, attention_mask,
                                  completion_mask, logits_to_keep, batch_size):
        """
        Compute token-level process rewards: r_φ(y_t) = β * [log π_φ(y_t) - log π_ref(y_t)]
        PRM and ref offloaded from CPU → GPU → CPU sequentially (never both on GPU).
        """
        with torch.no_grad():
            # PRM forward
            self._move_to_gpu(self.prm_model)
            self.prm_model.eval()
            prm_logps = self._get_token_logps(
                self.prm_model, prompt_completion_ids, attention_mask,
                logits_to_keep, batch_size=batch_size,
            )
            self._move_to_cpu(self.prm_model)

            # Ref forward
            self._move_to_gpu(self.ref_model)
            ref_logps = self._get_token_logps(
                self.ref_model, prompt_completion_ids, attention_mask,
                logits_to_keep, batch_size=batch_size,
            )
            self._move_to_cpu(self.ref_model)

        # Token-level process rewards
        process_rewards = self.prime_beta * (prm_logps - ref_logps)
        process_rewards = process_rewards * completion_mask
        return process_rewards, prm_logps, ref_logps

    def _compute_process_returns(self, process_rewards, completion_mask):
        """
        Compute discounted returns: Return_process(t) = Σ_{s=t}^{T} γ^{s-t} * r_φ(y_s)

        For γ=1 this is just reverse cumsum.
        """
        if self.prime_gamma == 1.0:
            # Efficient reverse cumsum
            masked = process_rewards * completion_mask
            returns = torch.flip(
                torch.cumsum(torch.flip(masked, dims=[1]), dim=1),
                dims=[1]
            )
        else:
            # General case with discount
            B, T = process_rewards.shape
            returns = torch.zeros_like(process_rewards)
            running = torch.zeros(B, device=process_rewards.device)
            for t in range(T - 1, -1, -1):
                running = process_rewards[:, t] + self.prime_gamma * running
                running = running * completion_mask[:, t]
                returns[:, t] = running
        return returns

    def _update_prm(self, prompt_completion_ids, attention_mask,
                     completion_mask, logits_to_keep, outcome_rewards, batch_size):
        """
        Online PRM update with BCE loss on outcome labels.

        Memory strategy: all inputs arrive on CPU.
        1. Offload policy model to free ~8GB
        2. Load ref → no_grad forward all samples (batch=1) → cache ref_logps → offload ref
        3. Load PRM → per-sample forward WITH grads → compute loss → backward
           immediately → free graph. This avoids accumulating 64 graphs.
        4. Clip grads → optimizer step → offload PRM → restore policy

        Peak GPU during PRM forward: ~1.6GB (PRM) + ~3-5GB (one sample activations
        for 16K seq × 248K vocab with gradient checkpointing) = ~5-7GB.
        """
        import gc
        from trl.trainer.grpo_trainer import selective_log_softmax

        N = prompt_completion_ids.size(0)  # total samples (e.g. 64)

        # ── Step 0: Offload policy model to free GPU ──
        policy_was_training = self.model.training
        policy_device = next(self.model.parameters()).device
        self.model.to("cpu")
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    state[k] = v.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Move input tensors to GPU
        pci_gpu = prompt_completion_ids.to(self._gpu_device)
        am_gpu = attention_mask.to(self._gpu_device)
        cm_gpu = completion_mask.to(self._gpu_device)
        labels = (outcome_rewards > 0.5).float().to(self._gpu_device)

        for _ in range(self.prime_prm_update_epochs):
            # ── Ref forward: all samples, no_grad, batch=1 ──
            self._move_to_gpu(self.ref_model)
            all_ref_logps = []
            with torch.no_grad():
                for i in range(N):
                    inp = pci_gpu[i:i+1]
                    mask = am_gpu[i:i+1]
                    logits = self.ref_model(input_ids=inp, attention_mask=mask, use_cache=False).logits
                    logits = logits[:, :-1, :][:, -logits_to_keep:, :]
                    logits.div_(self.temperature)
                    comp_ids = inp[:, -logits_to_keep:]
                    logps = selective_log_softmax(logits, comp_ids)
                    all_ref_logps.append(logps)
                    del logits, logps
            ref_logps = torch.cat(all_ref_logps, dim=0).detach()
            del all_ref_logps
            self._move_to_cpu(self.ref_model)

            # ── PRM forward with per-sample gradient accumulation ──
            self._move_to_gpu(self.prm_model)
            self.prm_model.train()
            self._move_optimizer_states(self._gpu_device)
            self.prm_optimizer.zero_grad()

            total_loss = 0.0
            for i in range(N):
                inp = pci_gpu[i:i+1]
                mask = am_gpu[i:i+1]
                logits = self.prm_model(input_ids=inp, attention_mask=mask, use_cache=False).logits
                logits = logits[:, :-1, :][:, -logits_to_keep:, :]
                logits.div_(self.temperature)
                comp_ids = inp[:, -logits_to_keep:]
                prm_logps_i = selective_log_softmax(logits, comp_ids)

                # Per-sample loss
                log_ratio_i = (prm_logps_i - ref_logps[i:i+1]) * cm_gpu[i:i+1]
                seq_score_i = self.prime_beta * log_ratio_i.sum(dim=1)
                loss_i = F.binary_cross_entropy_with_logits(seq_score_i, labels[i:i+1])
                # Scale by 1/N for mean reduction across samples
                (loss_i / N).backward()
                total_loss += loss_i.item()

                # Free graph immediately
                del logits, prm_logps_i, log_ratio_i, seq_score_i, loss_i

            torch.nn.utils.clip_grad_norm_(self.prm_model.parameters(), 1.0)
            self.prm_optimizer.step()

            prm_loss_val = total_loss / N

            # Clean up
            del ref_logps
            self._move_optimizer_states("cpu")
            self._move_to_cpu(self.prm_model)

        # ── Clean up GPU tensors ──
        del pci_gpu, am_gpu, cm_gpu, labels

        # ── Restore policy model to GPU ──
        self.model.to(policy_device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(policy_device)
        if policy_was_training:
            self.model.train()

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["prime/prm_loss"].append(prm_loss_val)

    def _generate_and_score_completions(self, inputs):
        """
        Override to inject PRIME dense advantage computation.

        Flow:
        1. Call parent to generate completions and compute outcome rewards
        2. Offload ALL GPU tensors to CPU → free GPU
        3. Update PRM online (GPU is mostly empty, ~9GB policy+optimizer)
        4. Compute process rewards with UPDATED PRM (per PRIME paper)
        5. Restore tensors to GPU, compute dense advantage
        """
        import gc

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # ── Step 1: Parent generates completions and computes outcome rewards ──
        output = super()._generate_and_score_completions(inputs)

        # Extract outcome rewards from parent's advantages:
        # Parent computes: advantages = rewards - mean_grouped_rewards (then optionally / std)
        # We need binary correctness labels. Decode completions and check accuracy.
        completion_ids = output["completion_ids"]
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completions_for_reward = [
            [{"role": "assistant", "content": c}] for c in completions_text
        ]
        solutions_repeated = [
            inp["solution"] for inp in inputs
            for _ in range(num_generations)
        ]
        # outcome_rewards: binary 0/1 per completion (on CPU to avoid GPU pressure)
        outcome_rewards_cpu = torch.tensor(
            accuracy_reward(completions_for_reward, solutions_repeated),
            dtype=torch.float32,
        )
        del completions_text, completions_for_reward, solutions_repeated

        # Prepare PRM inputs on CPU (needed for both _update_prm and _compute_process_rewards)
        prompt_completion_ids_cpu = torch.cat(
            [output["prompt_ids"], output["completion_ids"]], dim=1
        ).cpu()
        attention_mask_cpu = torch.cat(
            [output["prompt_mask"], output["completion_mask"]], dim=1
        ).cpu()
        completion_mask_cpu = output["completion_mask"].cpu()
        logits_to_keep = output["completion_ids"].size(1)
        batch_size = self.args.per_device_train_batch_size

        # ── Step 2: Offload ALL GPU tensors to CPU before PRM update ──
        # Move output dict to CPU
        output_cpu = {}
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                output_cpu[k] = v.cpu()
            else:
                output_cpu[k] = v
        output.clear()

        # Force free all GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"GPU after offload: {get_gpu_memory_info()}")

        # ── Step 3: Online PRM update (GPU has only ~9GB: policy + optimizer) ──
        if mode == "train":
            self._update_prm(
                prompt_completion_ids_cpu, attention_mask_cpu, completion_mask_cpu,
                logits_to_keep, outcome_rewards_cpu, batch_size,
            )

        # ── Step 4: Compute process rewards with UPDATED PRM (per PRIME paper) ──
        # _compute_process_rewards loads PRM→GPU→forward→CPU, then ref→GPU→forward→CPU
        # We need tensors on GPU for this (model forward expects GPU tensors)
        prompt_completion_ids_gpu = prompt_completion_ids_cpu.to(device)
        attention_mask_gpu = attention_mask_cpu.to(device)
        completion_mask_gpu = completion_mask_cpu.to(device)

        with torch.no_grad():
            process_rewards, _, _ = self._compute_process_rewards(
                prompt_completion_ids_gpu, attention_mask_gpu, completion_mask_gpu,
                logits_to_keep, batch_size,
            )

        # Free the temporary GPU copies (we'll restore from output_cpu)
        del prompt_completion_ids_gpu, attention_mask_gpu
        del prompt_completion_ids_cpu, attention_mask_cpu, completion_mask_cpu

        # ── Step 5: Restore output dict to GPU ──
        for k, v in output_cpu.items():
            if isinstance(v, torch.Tensor):
                output[k] = v.to(device)
            else:
                output[k] = v
        del output_cpu

        # completion_mask is back on GPU via output
        completion_mask = output["completion_mask"]
        outcome_rewards = outcome_rewards_cpu.to(device)
        del outcome_rewards_cpu

        # ── Step 6: Compute PRIME advantage ──
        # Process returns: Return_process(t) = Σ_{s=t}^T r_φ(y_s)
        process_returns = self._compute_process_returns(process_rewards, completion_mask)

        # Get baseline function
        baseline_fn = BASELINE_FUNCS[self.prime_baseline_type]
        baseline_kwargs = {}
        if self.prime_baseline_type == "dpo_z":
            baseline_kwargs["beta"] = self.prime_beta

        # --- Outcome component ---
        outcome_baseline = baseline_fn(outcome_rewards, num_generations, **baseline_kwargs)
        outcome_component = outcome_rewards - outcome_baseline  # (local_B,)

        # Normalize outcome component
        outcome_std = outcome_component.std()
        if outcome_std > 1e-8:
            outcome_component = outcome_component / (outcome_std + 1e-4)

        # --- Process component ---
        # Per-sample total process reward for baseline
        total_process = (process_rewards * completion_mask).sum(dim=1)  # (local_B,)
        process_baseline = baseline_fn(total_process, num_generations, **baseline_kwargs)

        # Token-level: subtract per-sample baseline, then compute returns
        process_centered = process_rewards - (process_baseline.unsqueeze(1) / completion_mask.sum(dim=1, keepdim=True).clamp(min=1))
        process_returns_centered = self._compute_process_returns(process_centered, completion_mask)

        # Normalize process component
        proc_vals = process_returns_centered[completion_mask.bool()]
        proc_std = proc_vals.std() if proc_vals.numel() > 1 else torch.tensor(1.0, device=device)
        if proc_std > 1e-8:
            process_returns_centered = process_returns_centered / (proc_std + 1e-4)

        # --- Online prompt filter ---
        if self.prime_online_filter:
            grouped_outcomes = outcome_rewards.view(-1, num_generations)
            all_correct = grouped_outcomes.sum(dim=1) == num_generations
            all_wrong = grouped_outcomes.sum(dim=1) == 0
            skip_mask = (all_correct | all_wrong).repeat_interleave(num_generations)
            # Zero out advantages for filtered prompts
            outcome_component = outcome_component * (~skip_mask).float()
            process_returns_centered = process_returns_centered * (~skip_mask).float().unsqueeze(1)
            frac_filtered = skip_mask.float().mean().item()
            self._metrics[mode]["prime/filtered_prompts_frac"].append(frac_filtered)

        # --- Combine: A_t = Return_process(t) + Return_outcome ---
        # outcome_component is (B,), needs to be (B, 1) for broadcasting
        dense_advantages = process_returns_centered + outcome_component.unsqueeze(1)

        # Token-level dense advantages for PRIME policy gradient
        output["advantages"] = dense_advantages

        # ── Logging ──
        self._metrics[mode]["prime/process_reward_mean"].append(
            process_rewards[completion_mask.bool()].mean().item()
        )
        self._metrics[mode]["prime/process_reward_std"].append(
            process_rewards[completion_mask.bool()].std().item()
        )
        self._metrics[mode]["prime/outcome_component_mean"].append(
            outcome_component.mean().item()
        )
        proc_mean = process_returns_centered[completion_mask.bool()].mean().item()
        self._metrics[mode]["prime/process_component_mean"].append(proc_mean)
        self._metrics[mode]["prime/advantage_mean"].append(
            dense_advantages[completion_mask.bool()].mean().item()
        )

        return output


# ──────────────────────────────────────────────────────────────
# Reward functions (same as train_grpo.py)
# ──────────────────────────────────────────────────────────────

def accuracy_reward(completions, solution, log_metric=None, **kwargs):
    """Check if model answer matches ground truth. Reward: 1.0/0.0."""
    rewards = []
    num_correct = 0
    for completion, sol in zip(completions, solution):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        predicted = extract_boxed_answer(content)
        if predicted is not None and verify_answer(predicted, sol):
            rewards.append(1.0)
            num_correct += 1
        else:
            rewards.append(0.0)
    if log_metric and len(rewards) > 0:
        log_metric("accuracy", num_correct / len(rewards))
    return rewards


def format_reward(completions, log_metric=None, **kwargs):
    """Check for </think> and \\boxed{} format. Reward: 0.0/0.5/1.0."""
    rewards = []
    format_ok_count = 0
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        # <think> is in prompt prefill, completion only has </think>
        if "</think>" in content:
            score += 0.5
        if "\\boxed{" in content:
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
    """Run PRIME-GRPO training."""
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    prime_cfg = config.get("prime", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3.5-0.8B-Base")
    run_name = config.get("run_name", "prime_grpo")
    loss_type = grpo_cfg.get("loss_type", "grpo")

    if output_dir is None:
        output_dir = f"./outputs/{run_name}"

    logger.info(f"PRIME-GRPO | loss_type={loss_type} | model={model_name}")
    logger.info(f"PRIME config: {prime_cfg}")
    logger.info(f"GPU info: {get_gpu_memory_info()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    if data_dir:
        train_dataset = load_from_disk(data_dir)
        logger.info(f"Loaded dataset: {len(train_dataset)} examples")
        num_samples = grpo_cfg.get("num_samples", None)
        if num_samples and num_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(num_samples))
            logger.info(f"Truncated to {num_samples} examples")
    else:
        raise ValueError("--data-dir is required")

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        loss_type=loss_type,
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high", None),
        beta=grpo_cfg.get("beta", 0.0),
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 8192),
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=grpo_cfg.get("top_p", 1.0),
        top_k=grpo_cfg.get("top_k", 0),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", True),
        generation_batch_size=grpo_cfg.get("generation_batch_size", None),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        # PRIME requires batch_size=1: no Liger kernel → full 248K logits in memory
        # batch=1 → 1×T×248K logits ≈ 7.6GB; batch=2 → 15GB+ → OOM on backward
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,  # 64×1 = 64 completions = same effective batch
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        save_only_model=True,
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="tensorboard",
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_mode=grpo_cfg.get("vllm_mode", "colocate"),
        **(
            {"vllm_gpu_memory_utilization": grpo_cfg.get("vllm_gpu_memory_utilization", 0.3)}
            if grpo_cfg.get("vllm_mode", "colocate") == "colocate"
            else {
                "vllm_server_port": grpo_cfg.get("vllm_server_port", 8000),
                "vllm_group_port": grpo_cfg.get("vllm_group_port", 51216),
            }
        ),
        use_liger_kernel=False,  # PRIME needs (B,T) token-level advantages; Liger expects (B,)
        reward_weights=[1.0, 0.5],
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
    )

    trainer = PrimeGRPOTrainer(
        prime_cfg=prime_cfg,
        model=model_name,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward, format_reward],
    )

    logger.info("Starting PRIME-GRPO training...")
    train_result = trainer.train()

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training complete! Metrics: {metrics}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="PRIME-GRPO Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    base_config_path = project_root / "configs" / "grpo_base.yaml"
    config = load_config(args.config, base_config_path=str(base_config_path))
    train(config=config, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
