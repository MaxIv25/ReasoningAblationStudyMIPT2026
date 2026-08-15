from datasets import Dataset
import pytest

from src.rl.dataset_contract import (
    validate_preformatted_math_rl_dataset,
    validate_prompt_token_lengths,
)


def test_accepts_raw_prompt_ending_at_reasoning_prefix():
    dataset = Dataset.from_dict(
        {
            "prompt": [
                "<|im_start|>user\nSolve.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            ],
            "solution": ["42"],
        }
    )
    validate_preformatted_math_rl_dataset(dataset)


def test_rejects_partial_assistant_conversational_prompt():
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": "Solve."},
                    {"role": "assistant", "content": "<think>\n"},
                ],
                "solution": "42",
            }
        ]
    )
    with pytest.raises(ValueError, match="preformatted strings"):
        validate_preformatted_math_rl_dataset(dataset)


class _WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


def test_prompt_token_cap_is_checked_before_training():
    dataset = Dataset.from_dict(
        {"prompt": ["one two", "one two three"], "solution": ["1", "2"]}
    )
    assert validate_prompt_token_lengths(dataset, _WhitespaceTokenizer(), 3) == 3
    with pytest.raises(ValueError, match="exceeding cap 2"):
        validate_prompt_token_lengths(dataset, _WhitespaceTokenizer(), 2)
