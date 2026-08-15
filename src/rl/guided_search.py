"""Candidate-pool selection for experimental PRM-guided rollouts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GuidedSelection:
    selected_mask: torch.Tensor
    sample_weights: torch.Tensor
    probabilities: torch.Tensor
    is_off_policy: bool


def select_prm_guided_candidates(
    sequence_scores: torch.Tensor,
    *,
    group_size: int,
    keep: int,
    mode: str = "topk",
    temperature: float = 1.0,
    allow_biased_update: bool = False,
    generator: torch.Generator | None = None,
) -> GuidedSelection:
    """Select candidate trajectories while making policy bias explicit.

    ``topk`` is deterministic best-of-N and therefore off-policy. It is blocked
    unless the experiment explicitly opts into biased policy updates.

    ``stochastic_ht`` samples with replacement from a PRM-softmax proposal and
    returns conditional Horvitz-Thompson weights targeting the uniform
    candidate-pool mean. The weights include the factor needed by TRL's
    downstream mean over the original ``group_size`` candidate rows; duplicate
    proposal draws are collapsed into their count.
    """
    if sequence_scores.ndim != 1 or sequence_scores.numel() % group_size:
        raise ValueError("sequence_scores must be a 1D tensor of complete groups")
    if not 1 <= keep <= group_size:
        raise ValueError("keep must be between 1 and group_size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    grouped = sequence_scores.reshape(-1, group_size)
    probabilities = torch.softmax(grouped / temperature, dim=1)
    selected_mask = torch.zeros_like(grouped, dtype=torch.bool)
    weights = torch.zeros_like(grouped, dtype=torch.float32)

    if mode == "topk":
        if not allow_biased_update:
            raise ValueError(
                "Deterministic top-k PRM selection is off-policy; set allow_biased_update=True explicitly"
            )
        indices = torch.topk(grouped, k=keep, dim=1).indices
        selected_mask.scatter_(1, indices, True)
        weights[selected_mask] = 1.0
        return GuidedSelection(
            selected_mask.reshape(-1),
            weights.reshape(-1),
            probabilities.reshape(-1),
            True,
        )

    if mode == "stochastic_ht":
        for group_idx, probs in enumerate(probabilities):
            sampled = torch.multinomial(
                probs, keep, replacement=True, generator=generator
            )
            counts = torch.bincount(sampled, minlength=group_size).to(weights.dtype)
            selected_mask[group_idx] = counts > 0
            weights[group_idx] = counts / (keep * probs.clamp_min(1e-12))
        return GuidedSelection(
            selected_mask.reshape(-1),
            weights.reshape(-1),
            probabilities.reshape(-1),
            False,
        )

    raise ValueError(f"Unknown guided selection mode: {mode}")
