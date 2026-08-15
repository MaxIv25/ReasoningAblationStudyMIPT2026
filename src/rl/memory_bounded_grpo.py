"""Token/vocabulary-memory-bounded loss path for on-policy vanilla GRPO."""

from __future__ import annotations

from typing import Any

import torch
from accelerate.utils import is_peft_model
from trl import GRPOTrainer
from trl.models.utils import _ForwardRedirection

from src.rl.chunked_logprobs import memory_bounded_grpo_loss


def validate_memory_bounded_grpo_args(args: Any) -> None:
    """Fail closed unless TRL can use ``current_logp.detach()`` as old policy."""
    required = {
        "loss_type": "grpo",
        "beta": 0.0,
        "num_iterations": 1,
        "importance_sampling_level": "token",
        "vllm_importance_sampling_correction": False,
        "use_liger_kernel": False,
    }
    mismatches = {
        name: (getattr(args, name, None), expected)
        for name, expected in required.items()
        if getattr(args, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            "memory-bounded GRPO requires exact on-policy vanilla settings; "
            f"mismatches={mismatches}"
        )
    if getattr(args, "off_policy_mask_threshold", None) is not None:
        raise ValueError("memory-bounded GRPO does not support off-policy masking")
    if getattr(args, "top_entropy_quantile", 1.0) != 1.0:
        raise ValueError("memory-bounded GRPO does not support entropy filtering")
    steps = int(getattr(args, "steps_per_generation", 0))
    accumulation = int(getattr(args, "gradient_accumulation_steps", 0))
    if steps <= 0 or accumulation <= 0 or accumulation % steps != 0:
        raise ValueError(
            "gradient_accumulation_steps must be divisible by "
            "steps_per_generation so no stored old-policy log-probs are needed"
        )


class MemoryBoundedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose policy projection is chunked over token positions.

    The rollout/reward/buffering implementation remains TRL 1.2's. Only the
    policy ``compute_loss`` path is replaced, and only for the fail-closed
    configuration checked above.
    """

    def __init__(
        self,
        *args,
        logprob_chunk_tokens: int = 256,
        **kwargs,
    ):
        trainer_args = kwargs.get("args")
        if trainer_args is None:
            raise TypeError("MemoryBoundedGRPOTrainer requires keyword argument args")
        validate_memory_bounded_grpo_args(trainer_args)
        if logprob_chunk_tokens <= 0:
            raise ValueError("logprob_chunk_tokens must be positive")
        self.logprob_chunk_tokens = int(logprob_chunk_tokens)
        self._memory_bounded_forward_redirection = _ForwardRedirection()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _output_projection(model):
        projection = model.get_output_embeddings()
        if projection is None or not hasattr(projection, "weight"):
            raise TypeError("policy must expose a weighted output embedding")
        return projection

    def compute_memory_bounded_loss(self, unwrapped_model, inputs):
        forbidden = (
            "old_per_token_logps",
            "ref_per_token_logps",
            "importance_sampling_ratio",
        )
        present = [name for name in forbidden if inputs.get(name) is not None]
        if present:
            raise RuntimeError(
                "memory-bounded GRPO received unsupported off-policy/reference "
                f"inputs: {present}"
            )
        advantages = inputs["advantages"]
        if advantages.ndim != 1:
            raise RuntimeError("memory-bounded GRPO requires sequence advantages (B,)")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        hidden = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
        )
        projection_owner = (
            unwrapped_model.base_model.model
            if is_peft_model(unwrapped_model)
            else unwrapped_model
        )
        projection = self._output_projection(projection_owner)
        loss_mask = completion_mask
        if inputs.get("tool_mask") is not None:
            loss_mask = loss_mask * inputs["tool_mask"]
        loss, clip_ratio = memory_bounded_grpo_loss(
            hidden,
            projection.weight,
            completion_ids,
            completion_mask=loss_mask,
            advantages=advantages,
            bias=getattr(projection, "bias", None),
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            chunk_tokens=self.logprob_chunk_tokens,
            temperature=self.temperature,
            checkpoint_chunks=True,
        )
        mode = "train" if self.model.training else "eval"
        gathered_clip_ratio = self.accelerator.gather(clip_ratio).mean().item()
        self._metrics[mode]["clip_ratio"].append(gathered_clip_ratio)
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        return loss / normalizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        unwrapped_model = self.accelerator.unwrap_model(model)
        return self._memory_bounded_forward_redirection(
            model,
            unwrapped_model,
            self.compute_memory_bounded_loss,
            unwrapped_model,
            inputs,
        )
