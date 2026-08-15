import pytest
import torch

from src.rl.guided_search import select_prm_guided_candidates
from src.rl.prompt_calibration import PromptCalibrationHead, prompt_calibrated_prm_loss


def test_prompt_calibration_head_and_prm_scores_receive_gradients():
    torch.manual_seed(0)
    head = PromptCalibrationHead(hidden_size=3)
    raw_scores = torch.tensor([0.2, -0.1, 0.3, -0.4], requires_grad=True)
    features = torch.randn(4, 3)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = prompt_calibrated_prm_loss(
        raw_scores,
        features,
        outcomes,
        group_size=4,
        calibration_head=head,
        ranking_weight=0.25,
        intercept_l2=1e-3,
    )
    result.loss.backward()
    assert raw_scores.grad is not None
    assert head.proj.weight.grad is not None
    assert result.intercepts.shape == outcomes.shape


def test_guided_topk_requires_explicit_off_policy_opt_in():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="off-policy"):
        select_prm_guided_candidates(
            scores, group_size=4, keep=2, mode="topk", allow_biased_update=False
        )


def test_guided_topk_selects_highest_prm_scores_when_opted_in():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 3.0, 1.0, 4.0, 2.0])
    result = select_prm_guided_candidates(
        scores,
        group_size=4,
        keep=2,
        mode="topk",
        allow_biased_update=True,
    )
    assert result.selected_mask.tolist() == [
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
    ]
    assert result.is_off_policy


def test_stochastic_pool_selection_returns_finite_ht_weights():
    generator = torch.Generator().manual_seed(7)
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    result = select_prm_guided_candidates(
        scores,
        group_size=4,
        keep=3,
        mode="stochastic_ht",
        temperature=0.5,
        generator=generator,
    )
    assert result.selected_mask.any()
    assert torch.isfinite(result.sample_weights).all()
    assert torch.all(result.sample_weights[~result.selected_mask] == 0)


def test_stochastic_ht_is_unbiased_under_trl_mean_with_duplicate_draws():
    generator = torch.Generator().manual_seed(17)
    group_size = 4
    keep = 3
    num_trials = 10_000
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8]).repeat(num_trials)
    candidate_losses = torch.tensor([0.5, -1.0, 2.0, 4.0])

    result = select_prm_guided_candidates(
        scores,
        group_size=group_size,
        keep=keep,
        mode="stochastic_ht",
        temperature=0.5,
        generator=generator,
    )
    selected = result.selected_mask.reshape(num_trials, group_size)
    weights = result.sample_weights.reshape(num_trials, group_size)

    # TRL's GRPO loss averages over the original G candidate rows. Repeated
    # proposal draws are represented by a single row with their count in weight.
    estimated_uniform_mean = (weights * candidate_losses).mean(dim=1).mean()
    assert (selected.sum(dim=1) < keep).any()
    assert estimated_uniform_mean.item() == pytest.approx(
        candidate_losses.mean().item(), abs=0.05
    )
