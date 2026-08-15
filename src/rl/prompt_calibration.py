"""Prompt-dependent calibration term for the experimental PRM objective."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class PromptCalibrationHead(nn.Module):
    """Learn a prompt intercept interpreted as a gauge term, not an identified Z."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, prompt_features: torch.Tensor) -> torch.Tensor:
        return self.proj(prompt_features.float()).squeeze(-1)


@dataclass(frozen=True)
class PromptCalibrationLoss:
    loss: torch.Tensor
    bce: torch.Tensor
    ranking: torch.Tensor
    intercept_penalty: torch.Tensor
    calibrated_scores: torch.Tensor
    intercepts: torch.Tensor


def _within_prompt_ranking_loss(
    scores: torch.Tensor, outcomes: torch.Tensor, group_size: int
) -> torch.Tensor:
    if scores.numel() % group_size:
        raise ValueError("scores must contain complete prompt groups")
    losses = []
    for group_scores, group_labels in zip(
        scores.reshape(-1, group_size), outcomes.reshape(-1, group_size)
    ):
        positives = group_scores[group_labels > 0.5]
        negatives = group_scores[group_labels <= 0.5]
        if positives.numel() and negatives.numel():
            losses.append(F.softplus(-(positives[:, None] - negatives[None, :])).mean())
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


def prompt_calibrated_prm_loss(
    raw_sequence_scores: torch.Tensor,
    prompt_features: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    group_size: int,
    calibration_head: PromptCalibrationHead,
    ranking_weight: float = 0.0,
    intercept_l2: float = 0.0,
) -> PromptCalibrationLoss:
    """Train absolute PRM logits plus within-prompt ordering.

    ``raw_sequence_scores`` is the implicit log-ratio reward. The learned
    prompt intercept completes its otherwise unidentifiable additive gauge. It
    can be interpreted as a ``beta log Z(x)`` proxy, but is deliberately not
    reported as a recovered partition function.
    """
    if raw_sequence_scores.ndim != 1 or outcomes.shape != raw_sequence_scores.shape:
        raise ValueError(
            "raw_sequence_scores and outcomes must be 1D tensors of the same shape"
        )
    if prompt_features.size(0) != raw_sequence_scores.size(0):
        raise ValueError("One prompt feature is required per trajectory")
    intercepts = calibration_head(prompt_features)
    calibrated = raw_sequence_scores + intercepts
    bce = F.binary_cross_entropy_with_logits(calibrated, outcomes.float())
    ranking = _within_prompt_ranking_loss(calibrated, outcomes.float(), group_size)
    penalty = intercepts.square().mean()
    loss = bce + ranking_weight * ranking + intercept_l2 * penalty
    return PromptCalibrationLoss(loss, bce, ranking, penalty, calibrated, intercepts)
