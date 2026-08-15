from types import SimpleNamespace

import torch
from trl import GRPOTrainer

from src.rl.dpo_z_trainer import DPOZGRPOTrainer
from src.train_prime import PrimeGRPOTrainer


def _base_eval_output():
    return {
        "completion_ids": torch.tensor([[1, 2]]),
        "completion_mask": torch.ones(1, 2),
        "advantages": torch.tensor([0.5]),
    }


def test_dpo_z_eval_uses_algorithm_neutral_outcome_path(monkeypatch):
    expected = _base_eval_output()
    monkeypatch.setattr(
        GRPOTrainer,
        "_generate_and_score_completions",
        lambda self, inputs: expected,
    )
    trainer = object.__new__(DPOZGRPOTrainer)
    trainer.model = SimpleNamespace(training=False)
    trainer._dpo_z_rewards_per_func = torch.tensor([[1.0]])

    assert trainer._generate_and_score_completions([{"prompt": "p"}]) is expected
    assert trainer._dpo_z_rewards_per_func is None


def test_prime_eval_uses_algorithm_neutral_outcome_path(monkeypatch):
    expected = _base_eval_output()
    monkeypatch.setattr(
        GRPOTrainer,
        "_generate_and_score_completions",
        lambda self, inputs: expected,
    )
    trainer = object.__new__(PrimeGRPOTrainer)
    trainer.model = SimpleNamespace(training=False)
    trainer._prime_step = 7

    assert trainer._generate_and_score_completions([{"prompt": "p"}]) is expected
    assert trainer._prime_step == 7
