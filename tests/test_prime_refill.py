from collections import defaultdict
from types import MethodType, SimpleNamespace

import torch

from src.train_prime import PrimeGRPOTrainer


def _output(prompt_len, outcomes, group_size=4):
    rows = len(outcomes)
    completion_len = 3
    return {
        "prompt_ids": torch.arange(rows * prompt_len).reshape(rows, prompt_len),
        "prompt_mask": torch.ones(rows, prompt_len, dtype=torch.long),
        "completion_ids": torch.ones(rows, completion_len, dtype=torch.long),
        "completion_mask": torch.ones(rows, completion_len, dtype=torch.long),
        "advantages": torch.zeros(rows),
        "num_items_in_batch": torch.tensor(rows * completion_len),
    }, torch.tensor(outcomes, dtype=torch.float32)


def test_accuracy_refill_replaces_invalid_groups_and_pads_batches():
    trainer = PrimeGRPOTrainer.__new__(PrimeGRPOTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=1)
    trainer.filter_lower = 0.2
    trainer.filter_upper = 0.8
    trainer.filter_accuracy = True
    trainer.filter_truncated_groups = False
    trainer.pad_token_id = 99
    trainer._metrics = {"train": defaultdict(list)}

    queue = [
        _output(2, [0, 0, 0, 0, 1, 0, 1, 0]),
        _output(4, [1, 1, 0, 0]),
    ]

    def score(self, inputs, device):
        return queue.pop(0)

    def refill(self, num_groups, group_size, refill_round):
        assert (num_groups, group_size, refill_round) == (1, 4, 1)
        return [{"solution": "x"}] * 4

    trainer._score_candidate_inputs = MethodType(score, trainer)
    trainer._sample_refill_inputs = MethodType(refill, trainer)
    trainer._prepare_refill_stream = MethodType(
        lambda self, inputs, group_size: None, trainer
    )
    inputs = [{"solution": "x"}] * 8
    output, outcomes = trainer._generate_with_accuracy_refill(
        inputs, torch.device("cpu"), group_size=4
    )

    assert outcomes.tolist() == [1, 0, 1, 0, 1, 1, 0, 0]
    assert output["prompt_ids"].shape == (8, 4)
    assert torch.all(output["prompt_ids"][:4, :2] == 99)
    assert trainer._metrics["train"]["prime/refill_rounds"] == [1]
