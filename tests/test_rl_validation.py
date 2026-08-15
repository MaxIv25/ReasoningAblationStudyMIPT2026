from datasets import Dataset

from src.rl.validation_data import build_unseen_validation_dataset


def _messages(problem: str):
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": "<think>\n"},
    ]


def test_build_unseen_validation_dataset_is_disjoint_raw_and_deterministic():
    train_prompt = (
        "<|im_start|>user\ntrain problem<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    train = Dataset.from_list([{"prompt": train_prompt, "solution": "1"}])
    candidates = Dataset.from_list(
        [
            {"prompt": _messages("train problem"), "solution": "1"},
            {"prompt": _messages("validation A"), "solution": "2"},
            {"prompt": _messages("validation B"), "solution": "3"},
            {"prompt": _messages("validation C"), "solution": "4"},
        ]
    )

    first = build_unseen_validation_dataset(candidates, train, size=2, seed=42)
    second = build_unseen_validation_dataset(candidates, train, size=2, seed=42)

    assert first[:] == second[:]
    assert len(first) == 2
    assert train_prompt not in first["prompt"]
    assert all(isinstance(prompt, str) for prompt in first["prompt"])
    assert all(
        prompt.endswith("<|im_start|>assistant\n<think>\n")
        for prompt in first["prompt"]
    )
