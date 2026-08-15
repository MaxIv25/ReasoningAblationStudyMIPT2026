import torch

from scripts.compare_grpo_policies import distribution_metrics


def test_distribution_metrics_separates_identical_and_divergent_policies():
    targets = torch.tensor([0, 2])
    vanilla = torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])

    identical = distribution_metrics(vanilla, vanilla.clone(), targets)
    assert identical["mean_kl_vanilla_dpoz"] == 0.0
    assert identical["mean_total_variation"] == 0.0
    assert identical["top1_disagreement_fraction"] == 0.0
    assert identical["mean_abs_target_logprob_difference"] == 0.0

    dpoz = torch.tensor([[0.0, 1.0, 3.0], [3.0, 1.0, 0.0]])
    divergent = distribution_metrics(vanilla, dpoz, targets)
    assert divergent["mean_kl_vanilla_dpoz"] > 0.0
    assert divergent["mean_total_variation"] > 0.0
    assert divergent["top1_disagreement_fraction"] == 1.0
    assert divergent["mean_abs_target_logprob_difference"] > 0.0
