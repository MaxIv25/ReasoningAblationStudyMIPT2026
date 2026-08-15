import math

import pytest
import torch

from src.rl.prime_core import (
    compute_group_advantages,
    compute_prime_advantages,
    dpo_z_baseline,
    implicit_process_rewards,
    prime_prm_bce_loss,
    normalize_process_rewards,
    non_truncated_group_mask,
    process_reward_weight,
    reverse_discounted_returns,
    solvable_group_mask,
)


def test_author_accuracy_filter_is_inclusive_and_group_level():
    outcomes = torch.tensor(
        [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.float32,
    )
    assert solvable_group_mask(outcomes, 4, lower=0.2, upper=0.8).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_dpo_z_uses_beta_log_mean_exp_reward_over_beta():
    rewards = torch.tensor([0.0, 1.0, 2.0])
    beta = 0.5
    baseline = dpo_z_baseline(rewards, group_size=3, beta=beta, leave_one_out=False)
    expected = beta * torch.logsumexp(rewards / beta, dim=0) - beta * math.log(3)
    assert torch.allclose(baseline, expected.expand_as(rewards))


def test_dpo_z_leave_one_out_excludes_current_trajectory():
    rewards = torch.tensor([0.0, 1.0, 2.0])
    baseline = dpo_z_baseline(rewards, group_size=3, beta=1.0, leave_one_out=True)
    expected_0 = torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - math.log(2)
    expected_2 = torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0) - math.log(2)
    assert baseline[0] == pytest.approx(expected_0.item())
    assert baseline[2] == pytest.approx(expected_2.item())


def test_dpo_z_replaces_centering_instead_of_being_cancelled():
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    advantages = compute_group_advantages(rewards, 4, baseline="dpo_z", dpo_z_beta=0.2)
    assert not torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-5)


def test_implicit_process_reward_is_raw_log_ratio_without_beta():
    prm = torch.tensor([[0.2, -0.3, 4.0]])
    ref = torch.tensor([[0.1, -0.5, 1.0]])
    mask = torch.tensor([[1, 1, 0]])
    reward = implicit_process_rewards(prm, ref, mask)
    assert torch.allclose(reward, torch.tensor([[0.1, 0.2, 0.0]]))
    assert torch.count_nonzero(implicit_process_rewards(ref, ref, mask)) == 0


def test_prime_prm_bce_matches_official_sigmoid_then_binary_cross_entropy():
    prm = torch.tensor([[0.4, 0.2], [-0.1, 0.3]])
    ref = torch.zeros_like(prm)
    mask = torch.ones_like(prm)
    outcomes = torch.tensor([1.0, 0.0])
    beta = 0.05
    loss, logits = prime_prm_bce_loss(prm, ref, mask, outcomes, beta=beta)
    expected_logits = beta * (prm - ref).sum(dim=1)
    expected = torch.nn.functional.binary_cross_entropy(
        expected_logits.sigmoid(), outcomes
    )
    assert torch.allclose(logits, expected_logits)
    assert torch.allclose(loss, expected)


def test_process_normalization_matches_author_global_reverse_cumsum_max():
    scores = torch.tensor([[1.0, 2.0, 0.0], [2.0, -1.0, 0.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    normalized, factor = normalize_process_rewards(scores, mask)
    assert factor == pytest.approx(3.0 + 1e-6)
    assert torch.allclose(normalized, scores * mask / factor)


def test_reverse_discounted_returns_respects_padding():
    rewards = torch.tensor([[1.0, 2.0, 99.0]])
    mask = torch.tensor([[1, 1, 0]])
    returns = reverse_discounted_returns(rewards, mask, gamma=1.0)
    assert returns.tolist() == [[3.0, 2.0, 0.0]]


def test_faithful_prime_filters_before_advantages_and_whitens_kept_tokens():
    outcomes = torch.tensor([0, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
    process = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.3, -0.1],
            [-0.1, 0.3],
        ]
    )
    mask = torch.ones_like(process)
    result = compute_prime_advantages(
        outcomes,
        process,
        mask,
        group_size=4,
        filter_lower=0.2,
        filter_upper=0.8,
        baseline="rloo",
    )
    assert result.keep_groups.tolist() == [False, True]
    assert torch.count_nonzero(result.advantages[:4]) == 0
    kept = result.advantages[4:][mask[4:].bool()]
    assert kept.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert kept.std(unbiased=False).item() == pytest.approx(1.0, abs=2e-4)


def test_process_reward_schedule_supports_linear_and_reliability_gate():
    assert process_reward_weight(
        step=5, schedule="linear", warmup_steps=10
    ) == pytest.approx(0.5)
    assert process_reward_weight(
        step=10,
        schedule="reliability_gate",
        warmup_steps=10,
        reliability=0.55,
        reliability_floor=0.5,
        reliability_full=0.7,
    ) == pytest.approx(0.25)


def test_official_process_coefficient_scales_the_schedule_weight():
    schedule_weight = process_reward_weight(step=0, schedule="constant")
    assert 5.0 * schedule_weight == pytest.approx(5.0)


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        compute_group_advantages(torch.zeros(4), 4, baseline="mystery")


def test_whole_group_truncation_filter_is_optional_but_exact():
    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    assert non_truncated_group_mask(
        mask, group_size=4, max_completion_length=3
    ).tolist() == [True, False]
