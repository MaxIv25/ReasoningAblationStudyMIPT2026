import torch
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch.nn.functional as F
from torch import nn

from src.rl.chunked_logprobs import (
    entropy_from_hidden,
    estimate_logits_bytes,
    memory_bounded_grpo_loss,
    selected_logprobs_from_hidden,
)
from src.rl.dpo_z_trainer import (
    DPOZGRPOTrainer,
    MemoryBoundedDPOZGRPOTrainer,
)
from src.rl.memory_bounded_grpo import (
    MemoryBoundedGRPOTrainer,
    validate_memory_bounded_grpo_args,
)


def _full_selected_logps(hidden, weight, targets):
    logits = F.linear(hidden, weight)
    return (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )


def test_chunked_selected_logps_match_full_forward_and_backward():
    torch.manual_seed(0)
    hidden_full = torch.randn(2, 5, 7, requires_grad=True)
    weight_full = torch.randn(13, 7, requires_grad=True)
    targets = torch.randint(0, 13, (2, 5))

    expected = _full_selected_logps(hidden_full, weight_full, targets)
    expected.sum().backward()

    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    weight_chunked = weight_full.detach().clone().requires_grad_(True)
    actual = selected_logprobs_from_hidden(
        hidden_chunked,
        weight_chunked,
        targets,
        chunk_tokens=3,
        checkpoint_chunks=True,
    )
    actual.sum().backward()

    assert torch.allclose(actual, expected.detach(), atol=1e-6)
    assert torch.allclose(hidden_chunked.grad, hidden_full.grad, atol=1e-5)
    assert torch.allclose(weight_chunked.grad, weight_full.grad, atol=1e-5)


def test_chunking_changes_vocab_tensor_peak_from_sequence_to_chunk_size():
    full = estimate_logits_bytes(
        batch_size=1, sequence_tokens=16384, vocab_size=248_320, dtype_bytes=2
    )
    chunked = estimate_logits_bytes(
        batch_size=1, sequence_tokens=256, vocab_size=248_320, dtype_bytes=2
    )
    assert full > 8_000_000_000
    assert chunked < 130_000_000
    assert full / chunked == 64


def test_chunked_entropy_matches_full_distribution():
    torch.manual_seed(3)
    hidden = torch.randn(2, 4, 5)
    weight = torch.randn(11, 5)
    logits = F.linear(hidden, weight).float()
    expected = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(
        -1
    )
    actual = entropy_from_hidden(hidden, weight, chunk_tokens=3)
    assert torch.allclose(actual, expected, atol=1e-6)


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 7)
        self.backbone = nn.Linear(7, 7)
        self.lm_head = nn.Linear(7, 13, bias=True)

    def hidden(self, input_ids):
        return torch.tanh(self.backbone(self.embed(input_ids)))


def _full_grpo_loss(model, input_ids, target_ids, mask, advantages):
    logits = model.lm_head(model.hidden(input_ids))
    logps = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    ratio = torch.exp(logps - logps.detach())
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    advantage = advantages.unsqueeze(-1)
    per_token_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
    mask = mask.to(per_token_loss.dtype)
    return (
        (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    ).mean()


def test_memory_bounded_grpo_matches_full_loss_and_backward_on_toy_policy():
    torch.manual_seed(11)
    full_model = _TinyPolicy()
    chunked_model = _TinyPolicy()
    chunked_model.load_state_dict(full_model.state_dict())
    input_ids = torch.randint(0, 17, (2, 6))
    target_ids = torch.randint(0, 13, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
    advantages = torch.tensor([0.75, -0.4])

    expected = _full_grpo_loss(
        full_model, input_ids, target_ids, mask, advantages
    )
    expected.backward()

    hidden = chunked_model.hidden(input_ids)
    actual, clip_ratio = memory_bounded_grpo_loss(
        hidden,
        chunked_model.lm_head.weight,
        target_ids,
        completion_mask=mask,
        advantages=advantages,
        bias=chunked_model.lm_head.bias,
        epsilon_low=0.2,
        epsilon_high=0.2,
        chunk_tokens=3,
        checkpoint_chunks=True,
    )
    actual.backward()

    assert torch.allclose(actual, expected.detach(), atol=1e-6)
    assert clip_ratio.item() == 0.0
    for expected_parameter, actual_parameter in zip(
        full_model.parameters(), chunked_model.parameters(), strict=True
    ):
        assert torch.allclose(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=1e-5,
        )


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 7)
        self.projection = nn.Linear(7, 7)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = torch.tanh(self.projection(self.embed(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBackbone()
        self.lm_head = nn.Linear(7, 13, bias=True)

    def get_output_embeddings(self):
        return self.lm_head


class _TinyAccelerator:
    is_main_process = True
    @staticmethod
    def gather(value):
        return value.reshape(1)


def _trainer_without_runtime(model, chunk_tokens=3):
    trainer = object.__new__(MemoryBoundedGRPOTrainer)
    trainer.model = model
    trainer.model_kwarg_keys = set()
    trainer.logprob_chunk_tokens = chunk_tokens
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    trainer.temperature = 1.0
    trainer.accelerator = _TinyAccelerator()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.current_gradient_accumulation_steps = 1
    return trainer


def test_trainer_override_matches_full_policy_loss_and_backward():
    torch.manual_seed(23)
    expected_model = _TinyCausalLM()
    actual_model = _TinyCausalLM()
    actual_model.load_state_dict(expected_model.state_dict())
    expected_model.train()
    actual_model.train()
    prompt_ids = torch.randint(0, 17, (2, 3))
    completion_ids = torch.randint(0, 13, (2, 5))
    inputs = {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
        "advantages": torch.tensor([0.6, -0.25]),
    }

    all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    expected_hidden = expected_model.model(all_ids).last_hidden_state[:, :-1]
    expected_hidden = expected_hidden[:, -completion_ids.size(1) :]
    expected = _full_grpo_loss(
        SimpleNamespace(hidden=lambda _: expected_hidden, lm_head=expected_model.lm_head),
        all_ids,
        completion_ids,
        inputs["completion_mask"],
        inputs["advantages"],
    )
    expected.backward()

    trainer = _trainer_without_runtime(actual_model)
    actual = trainer.compute_memory_bounded_loss(actual_model, inputs)
    actual.backward()

    assert torch.allclose(actual, expected.detach(), atol=1e-6)
    assert trainer._metrics["train"]["clip_ratio"] == [0.0]
    for expected_parameter, actual_parameter in zip(
        expected_model.parameters(), actual_model.parameters(), strict=True
    ):
        assert torch.allclose(actual_parameter.grad, expected_parameter.grad, atol=1e-5)


def test_dpo_z_memory_bounded_trainer_composes_both_responsibilities():
    assert issubclass(MemoryBoundedDPOZGRPOTrainer, DPOZGRPOTrainer)
    assert issubclass(MemoryBoundedDPOZGRPOTrainer, MemoryBoundedGRPOTrainer)
    assert (
        MemoryBoundedDPOZGRPOTrainer._generate_and_score_completions
        is DPOZGRPOTrainer._generate_and_score_completions
    )
    assert (
        MemoryBoundedDPOZGRPOTrainer.compute_loss is MemoryBoundedGRPOTrainer.compute_loss
    )


def _safe_memory_bounded_args(**overrides):
    values = {
        "loss_type": "grpo",
        "beta": 0.0,
        "num_iterations": 1,
        "importance_sampling_level": "token",
        "vllm_importance_sampling_correction": False,
        "use_liger_kernel": False,
        "off_policy_mask_threshold": None,
        "top_entropy_quantile": 1.0,
        "steps_per_generation": 8,
        "gradient_accumulation_steps": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_memory_bounded_args_accept_only_exact_on_policy_contract():
    validate_memory_bounded_grpo_args(_safe_memory_bounded_args())


@pytest.mark.parametrize(
    "override",
    [
        {"beta": 0.01},
        {"num_iterations": 2},
        {"use_liger_kernel": True},
        {"vllm_importance_sampling_correction": True},
        {"off_policy_mask_threshold": 2.0},
        {"gradient_accumulation_steps": 7},
    ],
)
def test_memory_bounded_args_fail_closed_outside_supported_contract(override):
    with pytest.raises(ValueError):
        validate_memory_bounded_grpo_args(_safe_memory_bounded_args(**override))
