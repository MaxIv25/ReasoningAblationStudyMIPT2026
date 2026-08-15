"""Pure tensor implementation of PRIME rewards, baselines, and schedules.

The faithful path mirrors the public PRIME implementation:

* filter complete prompt groups by verifier accuracy;
* normalize implicit token rewards by the maximum absolute reverse return;
* apply RLOO independently to verifier and process rewards;
* reverse-cumulate both sources and whiten the final token advantages.

Experimental baselines are explicit alternatives and never silently alter the
faithful defaults.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrimeAdvantageResult:
    advantages: torch.Tensor
    keep_groups: torch.Tensor
    normalized_process_rewards: torch.Tensor
    process_normalization_factor: torch.Tensor
    process_weight: float


def _group(values: torch.Tensor, group_size: int) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape={tuple(values.shape)}")
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for group baselines")
    if values.numel() % group_size:
        raise ValueError(
            f"{values.numel()} values are not divisible by group_size={group_size}"
        )
    return values.reshape(-1, group_size)


def solvable_group_mask(
    outcomes: torch.Tensor,
    group_size: int,
    lower: float = 0.2,
    upper: float = 0.8,
) -> torch.Tensor:
    """Return one inclusive accuracy-filter decision per prompt group."""
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("Expected 0 <= lower <= upper <= 1")
    grouped = _group(outcomes.float(), group_size)
    accuracy = grouped.mean(dim=1)
    return (accuracy >= lower) & (accuracy <= upper)


def non_truncated_group_mask(
    completion_mask: torch.Tensor,
    group_size: int,
    max_completion_length: int,
) -> torch.Tensor:
    """Keep a prompt group only when none of its responses hits the length cap."""
    if completion_mask.ndim != 2 or completion_mask.size(0) % group_size:
        raise ValueError("completion_mask must contain complete prompt groups")
    lengths = completion_mask.sum(dim=1).reshape(-1, group_size)
    return ~(lengths >= max_completion_length).any(dim=1)


def dpo_z_baseline(
    rewards: torch.Tensor,
    group_size: int,
    beta: float,
    *,
    leave_one_out: bool = True,
) -> torch.Tensor:
    """Monte-Carlo DPO partition baseline ``beta * log E exp(R / beta)``.

    Leave-one-out is the default because the same trajectories are normally
    used both to estimate the prompt baseline and to form policy advantages.
    """
    if beta <= 0:
        raise ValueError("DPO-Z beta must be positive")
    grouped = _group(rewards, group_size)
    scaled = grouped / beta
    if leave_one_out:
        eye = torch.eye(group_size, dtype=torch.bool, device=rewards.device).unsqueeze(
            0
        )
        candidates = (
            scaled.unsqueeze(1).expand(-1, group_size, -1).masked_fill(eye, -torch.inf)
        )
        log_mean_exp = torch.logsumexp(candidates, dim=-1) - math.log(group_size - 1)
    else:
        log_mean_exp = torch.logsumexp(scaled, dim=-1, keepdim=True) - math.log(
            group_size
        )
        log_mean_exp = log_mean_exp.expand_as(grouped)
    return (beta * log_mean_exp).reshape_as(rewards)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    baseline: str = "rloo",
    dpo_z_beta: float = 0.1,
    dpo_z_leave_one_out: bool = True,
) -> torch.Tensor:
    """Compute group advantages without an additional centering pass."""
    grouped = _group(rewards, group_size)
    if baseline == "rloo":
        return (
            (group_size * grouped - grouped.sum(dim=1, keepdim=True)) / (group_size - 1)
        ).reshape_as(rewards)
    if baseline == "group_mean":
        return (grouped - grouped.mean(dim=1, keepdim=True)).reshape_as(rewards)
    if baseline == "dpo_z":
        return rewards - dpo_z_baseline(
            rewards,
            group_size,
            dpo_z_beta,
            leave_one_out=dpo_z_leave_one_out,
        )
    raise ValueError(f"Unknown advantage baseline: {baseline}")


def reverse_discounted_returns(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Return ``sum_{s=t} gamma**(s-t) r_s`` while respecting right padding."""
    if rewards.shape != mask.shape:
        raise ValueError("rewards and mask must have identical shapes")
    masked = rewards * mask.to(rewards.dtype)
    if gamma == 1.0:
        result = torch.flip(torch.cumsum(torch.flip(masked, dims=[1]), dim=1), dims=[1])
        return result * mask.to(result.dtype)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    result = torch.zeros_like(masked)
    running = torch.zeros(masked.size(0), dtype=masked.dtype, device=masked.device)
    for token_idx in range(masked.size(1) - 1, -1, -1):
        running = (masked[:, token_idx] + gamma * running) * mask[:, token_idx].to(
            masked.dtype
        )
        result[:, token_idx] = running
    return result


def implicit_process_rewards(
    prm_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the raw token reward ``log pi_prm - log pi_ref`` used by PRIME."""
    if prm_logps.shape != reference_logps.shape or prm_logps.shape != mask.shape:
        raise ValueError("PRM log-probs, reference log-probs, and mask must match")
    return (prm_logps - reference_logps) * mask.to(prm_logps.dtype)


def prime_prm_bce_loss(
    prm_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    mask: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    beta: float,
    prompt_intercept: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official PRIME sequence BCE, optionally with an experimental intercept."""
    if beta <= 0:
        raise ValueError("PRM beta must be positive")
    if outcomes.ndim != 1 or outcomes.numel() != prm_logps.size(0):
        raise ValueError("Expected one binary outcome per trajectory")
    logits = beta * implicit_process_rewards(prm_logps, reference_logps, mask).sum(
        dim=1
    )
    if prompt_intercept is not None:
        if prompt_intercept.shape != logits.shape:
            raise ValueError("prompt_intercept must have one value per trajectory")
        logits = logits + prompt_intercept
    loss = F.binary_cross_entropy_with_logits(logits, outcomes.to(logits.dtype))
    return loss, logits


def normalize_process_rewards(
    process_rewards: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the global PRIME ``batch_norm`` used in ``dp_prime.py``."""
    masked = process_rewards * mask.to(process_rewards.dtype)
    reverse_cumsum = torch.flip(
        torch.cumsum(torch.flip(masked, dims=[1]), dim=1), dims=[1]
    )
    factor = reverse_cumsum.abs().max() + eps
    return masked / factor, factor


def _center_process_rewards(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    *,
    baseline: str,
    dpo_z_beta: float,
    dpo_z_leave_one_out: bool,
) -> torch.Tensor:
    token_counts = mask.sum(dim=1).clamp(min=1).to(rewards.dtype)
    sequence_means = (rewards * mask).sum(dim=1) / token_counts
    if baseline == "rloo":
        grouped_means = _group(sequence_means, group_size)
        prompt_sum = (
            grouped_means.sum(dim=1, keepdim=True).expand_as(grouped_means).reshape(-1)
        )
        centered = rewards * (group_size / (group_size - 1)) - prompt_sum.unsqueeze(
            1
        ) / (group_size - 1)
    elif baseline == "group_mean":
        grouped_means = _group(sequence_means, group_size)
        prompt_mean = (
            grouped_means.mean(dim=1, keepdim=True).expand_as(grouped_means).reshape(-1)
        )
        centered = rewards - prompt_mean.unsqueeze(1)
    elif baseline == "dpo_z":
        prompt_baseline = dpo_z_baseline(
            sequence_means,
            group_size,
            dpo_z_beta,
            leave_one_out=dpo_z_leave_one_out,
        )
        centered = rewards - prompt_baseline.unsqueeze(1)
    else:
        raise ValueError(f"Unknown advantage baseline: {baseline}")
    return centered * mask.to(centered.dtype)


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return torch.zeros_like(values)
    selected = values[valid]
    mean = selected.mean()
    variance = (selected - mean).square().mean()
    whitened = (values - mean) * torch.rsqrt(variance + eps)
    return whitened * mask.to(whitened.dtype)


def process_reward_weight(
    *,
    step: int,
    schedule: str = "constant",
    warmup_steps: int = 0,
    reliability: float | None = None,
    reliability_floor: float = 0.5,
    reliability_full: float = 0.7,
) -> float:
    """Weight process returns using an isolated, auditable schedule."""
    if schedule == "constant":
        return 1.0
    warmup = 1.0 if warmup_steps <= 0 else min(1.0, max(0.0, step / warmup_steps))
    if schedule == "linear":
        return warmup
    if schedule == "reliability_gate":
        if reliability is None:
            raise ValueError("reliability_gate requires a reliability value")
        if reliability_full <= reliability_floor:
            raise ValueError("reliability_full must be greater than reliability_floor")
        gate = (reliability - reliability_floor) / (
            reliability_full - reliability_floor
        )
        return warmup * min(1.0, max(0.0, gate))
    raise ValueError(f"Unknown process reward schedule: {schedule}")


def compute_prime_advantages(
    outcomes: torch.Tensor,
    process_rewards: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    group_size: int,
    filter_lower: float = 0.2,
    filter_upper: float = 0.8,
    baseline: str = "rloo",
    dpo_z_beta: float = 0.1,
    dpo_z_leave_one_out: bool = True,
    gamma: float = 1.0,
    process_weight: float = 1.0,
) -> PrimeAdvantageResult:
    """Compute filtered, separately-baselined, token-level PRIME advantages."""
    if outcomes.numel() != process_rewards.size(0):
        raise ValueError("One outcome is required for every trajectory")
    if process_rewards.shape != completion_mask.shape:
        raise ValueError(
            "process_rewards and completion_mask must have identical shapes"
        )

    keep_groups = solvable_group_mask(outcomes, group_size, filter_lower, filter_upper)
    keep_rows = keep_groups.repeat_interleave(group_size)
    advantages = torch.zeros_like(process_rewards)
    normalized_all = torch.zeros_like(process_rewards)
    if not keep_rows.any():
        factor = process_rewards.new_tensor(1.0)
        return PrimeAdvantageResult(
            advantages, keep_groups, normalized_all, factor, float(process_weight)
        )

    kept_outcomes = outcomes[keep_rows]
    kept_process = process_rewards[keep_rows]
    kept_mask = completion_mask[keep_rows]
    normalized, factor = normalize_process_rewards(kept_process, kept_mask)
    normalized_all[keep_rows] = normalized

    outcome_advantage = compute_group_advantages(
        kept_outcomes,
        group_size,
        baseline=baseline,
        dpo_z_beta=dpo_z_beta,
        dpo_z_leave_one_out=dpo_z_leave_one_out,
    )
    centered_process = _center_process_rewards(
        normalized,
        kept_mask,
        group_size,
        baseline=baseline,
        dpo_z_beta=dpo_z_beta,
        dpo_z_leave_one_out=dpo_z_leave_one_out,
    )
    process_returns = reverse_discounted_returns(
        centered_process, kept_mask, gamma=gamma
    )
    dense = process_weight * process_returns + outcome_advantage.unsqueeze(1)
    advantages[keep_rows] = masked_whiten(dense, kept_mask)
    return PrimeAdvantageResult(
        advantages, keep_groups, normalized_all, factor, float(process_weight)
    )
