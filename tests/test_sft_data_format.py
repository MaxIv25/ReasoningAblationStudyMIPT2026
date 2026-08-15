from datasets import Dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from src.data_utils import ensure_prompt_completion_format


def test_legacy_messages_are_split_for_completion_only_loss():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "reasoning"},
                ],
                "difficulty": 0.5,
            }
        ]
    )

    converted = ensure_prompt_completion_format(dataset)

    assert "messages" not in converted.column_names
    assert converted[0]["prompt"] == (
        "<|im_start|>user\nproblem<|im_end|>\n<|im_start|>assistant\n"
    )
    assert converted[0]["completion"] == "reasoning<|im_end|>\n"
    assert converted[0]["difficulty"] == 0.5


def test_completion_mask_hides_prompt_labels():
    collator = DataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=True,
    )

    batch = collator(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "completion_mask": [0, 0, 1, 1],
            }
        ]
    )

    assert batch["labels"].tolist() == [[-100, -100, 12, 13]]
