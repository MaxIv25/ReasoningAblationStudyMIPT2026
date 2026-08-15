from collections import defaultdict
from types import MethodType, SimpleNamespace

import pytest
import torch

from src.train_prime import PrimeGRPOTrainer


def _output(outcomes, group_size=4):
    rows = len(outcomes)
    return {
        "prompt_ids": torch.arange(rows * 2).reshape(rows, 2),
        "prompt_mask": torch.ones(rows, 2, dtype=torch.long),
        "completion_ids": torch.ones(rows, 3, dtype=torch.long),
        "completion_mask": torch.ones(rows, 3, dtype=torch.long),
        "advantages": torch.zeros(rows),
        "num_items_in_batch": torch.tensor(rows * 3),
    }, torch.tensor(outcomes, dtype=torch.float32)


def _trainer():
    trainer = PrimeGRPOTrainer.__new__(PrimeGRPOTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=1)
    trainer.args = SimpleNamespace(seed=42)
    trainer._prime_step = 3
    trainer.filter_lower = 0.2
    trainer.filter_upper = 0.8
    trainer.filter_accuracy = True
    trainer.filter_truncated_groups = False
    trainer.max_completion_length = 16
    trainer.pad_token_id = 99
    trainer._metrics = {"train": defaultdict(list)}
    return trainer


def test_official_style_refill_continues_past_legacy_eight_round_cap():
    trainer = _trainer()
    trainer.max_refill_rounds = 8  # legacy local field must no longer stop collection
    calls = 0

    def score(self, inputs, device):
        nonlocal calls
        calls += 1
        return _output([0, 0, 0, 0] if calls <= 8 else [1, 0, 1, 0])

    trainer._score_candidate_inputs = MethodType(score, trainer)
    trainer._sample_refill_inputs = MethodType(
        lambda self, num_groups, group_size, refill_round: [
            {"prompt": f"refill-{refill_round}", "solution": "x"}
        ]
        * group_size,
        trainer,
    )
    trainer._prepare_refill_stream = MethodType(
        lambda self, inputs, group_size: None, trainer
    )

    output, outcomes = trainer._generate_with_accuracy_refill(
        [{"prompt": "initial", "solution": "x"}] * 4,
        torch.device("cpu"),
        group_size=4,
    )

    assert calls == 9
    assert outcomes.tolist() == [1, 0, 1, 0]
    assert output["completion_ids"].shape[0] == 4
    assert trainer._metrics["train"]["prime/refill_rounds"] == [8]


def test_refill_stream_is_without_replacement_and_excludes_initial_prompts():
    trainer = _trainer()
    trainer.train_dataset = [
        {"prompt": f"prompt-{index}", "solution": str(index)} for index in range(4)
    ]
    initial = [dict(trainer.train_dataset[0]) for _ in range(4)]

    trainer._prepare_refill_stream(initial, group_size=4)
    first = trainer._sample_refill_inputs(2, group_size=4, refill_round=1)
    second = trainer._sample_refill_inputs(1, group_size=4, refill_round=2)

    sampled_prompts = [first[0]["prompt"], first[4]["prompt"], second[0]["prompt"]]
    assert "prompt-0" not in sampled_prompts
    assert len(set(sampled_prompts)) == 3
    with pytest.raises(RuntimeError, match="refill dataset exhausted"):
        trainer._sample_refill_inputs(1, group_size=4, refill_round=3)
