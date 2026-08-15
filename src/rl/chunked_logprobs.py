"""Exact selected-token log-probabilities without a full sequence-vocab tensor."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def estimate_logits_bytes(
    *,
    batch_size: int,
    sequence_tokens: int,
    vocab_size: int,
    dtype_bytes: int,
) -> int:
    return batch_size * sequence_tokens * vocab_size * dtype_bytes


def selected_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_tokens: int = 256,
    temperature: float = 1.0,
    checkpoint_chunks: bool = False,
) -> torch.Tensor:
    """Project only ``chunk_tokens`` positions to the vocabulary at a time.

    With ``checkpoint_chunks=True`` the vocab projection and log-softmax are
    recomputed during backward, so autograd does not retain every chunk's
    logits. Peak vocab-tensor memory is therefore proportional to
    ``chunk_tokens * vocab_size`` rather than ``sequence_length * vocab_size``.
    """
    if hidden_states.ndim != 3 or target_ids.shape != hidden_states.shape[:2]:
        raise ValueError("Expected hidden_states=(B,T,H) and target_ids=(B,T)")
    if lm_head_weight.ndim != 2 or lm_head_weight.size(1) != hidden_states.size(-1):
        raise ValueError("lm_head_weight must have shape (vocab_size, hidden_size)")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size, sequence_length, hidden_size = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    flat_targets = target_ids.reshape(-1)
    selected = []

    for start in range(0, flat_hidden.size(0), chunk_tokens):
        end = min(start + chunk_tokens, flat_hidden.size(0))
        hidden_chunk = flat_hidden[start:end]
        target_chunk = flat_targets[start:end]

        if bias is None:

            def project(chunk, weight, selected_ids):
                logits = F.linear(chunk, weight) / temperature
                return (
                    F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    .gather(-1, selected_ids.unsqueeze(-1))
                    .squeeze(-1)
                )

            use_checkpoint = checkpoint_chunks and (
                hidden_chunk.requires_grad or lm_head_weight.requires_grad
            )
            logps = (
                checkpoint(
                    project,
                    hidden_chunk,
                    lm_head_weight,
                    target_chunk,
                    use_reentrant=False,
                )
                if use_checkpoint
                else project(hidden_chunk, lm_head_weight, target_chunk)
            )
        else:

            def project(chunk, weight, linear_bias, selected_ids):
                logits = F.linear(chunk, weight, linear_bias) / temperature
                return (
                    F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    .gather(-1, selected_ids.unsqueeze(-1))
                    .squeeze(-1)
                )

            use_checkpoint = checkpoint_chunks and (
                hidden_chunk.requires_grad
                or lm_head_weight.requires_grad
                or bias.requires_grad
            )
            logps = (
                checkpoint(
                    project,
                    hidden_chunk,
                    lm_head_weight,
                    bias,
                    target_chunk,
                    use_reentrant=False,
                )
                if use_checkpoint
                else project(hidden_chunk, lm_head_weight, bias, target_chunk)
            )
        selected.append(logps)

    return torch.cat(selected, dim=0).reshape(batch_size, sequence_length)


def memory_bounded_grpo_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    bias: torch.Tensor | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    chunk_tokens: int = 256,
    temperature: float = 1.0,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact on-policy vanilla GRPO with bounded vocab projections.

    This is the ``beta=0``, single-iteration case where TRL/Liger defines the
    old policy log-probability as ``current_logp.detach()``. The importance
    ratio is therefore one in the forward pass while retaining the policy
    gradient through the non-detached side. Only selected-token log-probabilities
    are kept; each temporary vocabulary tensor contains at most
    ``chunk_tokens`` positions.

    Returns the scalar sequence-normalized GRPO loss and its clip ratio.
    """
    if completion_mask.shape != target_ids.shape:
        raise ValueError("completion_mask must have the same shape as target_ids")
    if advantages.ndim != 1 or advantages.size(0) != target_ids.size(0):
        raise ValueError("advantages must have shape (batch_size,)")
    if not 0 <= epsilon_low < 1:
        raise ValueError("epsilon_low must satisfy 0 <= epsilon_low < 1")
    if epsilon_high < 0:
        raise ValueError("epsilon_high must be non-negative")

    per_token_logps = selected_logprobs_from_hidden(
        hidden_states,
        lm_head_weight,
        target_ids,
        bias=bias,
        chunk_tokens=chunk_tokens,
        temperature=temperature,
        checkpoint_chunks=checkpoint_chunks,
    )
    log_ratio = per_token_logps - per_token_logps.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - epsilon_low,
        max=1.0 + epsilon_high,
    )
    advantages = advantages.to(per_token_logps.dtype).unsqueeze(-1)
    per_token_loss = -torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages,
    )
    mask = completion_mask.to(per_token_loss.dtype)
    loss = (
        (per_token_loss * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp(min=1.0)
    ).mean()

    is_lower_clipped = ratio < (1.0 - epsilon_low)
    is_upper_clipped = ratio > (1.0 + epsilon_high)
    is_clipped = (is_lower_clipped & (advantages < 0)) | (
        is_upper_clipped & (advantages > 0)
    )
    clip_ratio = (is_clipped * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, clip_ratio.detach()


@torch.no_grad()
def entropy_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_tokens: int = 256,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute exact categorical entropy with the same bounded token chunks."""
    if hidden_states.ndim != 3:
        raise ValueError("Expected hidden_states=(B,T,H)")
    batch_size, sequence_length, hidden_size = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    entropies = []
    for start in range(0, flat_hidden.size(0), chunk_tokens):
        chunk = flat_hidden[start : start + chunk_tokens]
        logits = F.linear(chunk, lm_head_weight, bias).float() / temperature
        log_normalizer = torch.logsumexp(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        entropies.append(log_normalizer - (probabilities * logits).sum(dim=-1))
    return torch.cat(entropies).reshape(batch_size, sequence_length)
