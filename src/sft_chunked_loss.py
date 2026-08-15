"""Exact bounded-memory causal language-model cross entropy for SFT."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def liger_fused_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Exact causal CE using the generic Liger fused linear-cross-entropy.

    This calls the model-independent loss primitive directly, bypassing the
    Liger model-type dispatcher, which does not support Qwen3.5 in version 0.7.0.
    """
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    flat_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
    flat_targets = labels[:, 1:].reshape(-1)
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss_fn = LigerFusedLinearCrossEntropyLoss(
        ignore_index=ignore_index,
        reduction=reduction,
    )
    loss = loss_fn(lm_head_weight, flat_hidden, flat_targets, bias=bias)
    if num_items_in_batch is not None:
        denominator = torch.as_tensor(
            num_items_in_batch, device=loss.device, dtype=loss.dtype
        )
        loss = loss / denominator
    return loss


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    chunk_tokens: int = 1024,
    checkpoint_chunks: bool = True,
    ignore_index: int = -100,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Return exact next-token CE without retaining ``[tokens, vocab]`` logits.

    Each vocabulary projection is limited to ``chunk_tokens`` positions. When
    checkpointing is enabled, the projection and CE are recomputed during
    backward, so autograd does not retain every chunk's logits simultaneously.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError("labels must match hidden_states batch and sequence dimensions")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    flat_hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
    flat_targets = labels[:, 1:].reshape(-1)
    valid_tokens = (flat_targets != ignore_index).sum()
    if not bool(valid_tokens):
        # Preserve a differentiable zero for fully masked micro-batches.
        return flat_hidden.sum() * 0.0

    def project_and_sum(
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        weight: torch.Tensor,
        linear_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = F.linear(hidden_chunk, weight, linear_bias)
        return F.cross_entropy(
            logits,
            target_chunk,
            ignore_index=ignore_index,
            reduction="sum",
        )

    loss_sum = flat_hidden.new_zeros(())
    for start in range(0, flat_hidden.size(0), chunk_tokens):
        end = min(start + chunk_tokens, flat_hidden.size(0))
        hidden_chunk = flat_hidden[start:end]
        target_chunk = flat_targets[start:end]

        if bias is None:
            chunk_fn: Callable[..., torch.Tensor] = lambda h, t, w: project_and_sum(h, t, w)
            args = (hidden_chunk, target_chunk, lm_head_weight)
        else:
            chunk_fn = project_and_sum
            args = (hidden_chunk, target_chunk, lm_head_weight, bias)

        needs_grad = any(tensor.requires_grad for tensor in args if tensor.is_floating_point())
        chunk_loss = (
            checkpoint(chunk_fn, *args, use_reentrant=False)
            if checkpoint_chunks and needs_grad
            else chunk_fn(*args)
        )
        loss_sum = loss_sum + chunk_loss

    denominator = valid_tokens if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, device=loss_sum.device, dtype=loss_sum.dtype)
    return loss_sum / denominator


def estimate_vocab_projection_bytes(
    *, chunk_tokens: int, vocab_size: int, dtype_bytes: int = 2
) -> int:
    """Estimate the largest raw chunked vocabulary projection tensor."""
    if chunk_tokens <= 0 or vocab_size <= 0 or dtype_bytes <= 0:
        raise ValueError("chunk_tokens, vocab_size, and dtype_bytes must be positive")
    return chunk_tokens * vocab_size * dtype_bytes
