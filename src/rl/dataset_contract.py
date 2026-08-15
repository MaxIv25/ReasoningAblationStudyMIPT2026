"""Fail-fast validation for preformatted single-turn RL datasets."""

from datasets import Dataset


def validate_preformatted_math_rl_dataset(dataset: Dataset) -> None:
    """Require raw prompts that already end at the assistant reasoning prefix.

    TRL applies a chat template to conversational rows. A row that already
    contains a partial assistant ``<think>`` message would therefore acquire a
    second assistant turn. Raw strings are passed through unchanged.
    """
    required = {"prompt", "solution"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"RL dataset is missing columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("RL dataset is empty")

    sample_count = min(len(dataset), 32)
    expected_suffix = "<|im_start|>assistant\n<think>\n"
    for index in range(sample_count):
        row = dataset[index]
        prompt = row["prompt"]
        solution = row["solution"]
        if not isinstance(prompt, str):
            raise ValueError(
                "RL prompts must be preformatted strings, not conversational "
                f"messages (row {index}, type={type(prompt).__name__})"
            )
        if not prompt.endswith(expected_suffix):
            raise ValueError(
                f"RL prompt row {index} must end with {expected_suffix!r}"
            )
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"RL solution row {index} must be a non-empty string")


def validate_prompt_token_lengths(dataset: Dataset, tokenizer, max_length: int) -> int:
    """Return the maximum prompt length and reject any prompt over the cap."""
    if max_length <= 0:
        raise ValueError("max prompt length must be positive")
    observed_max = 0
    for index, prompt in enumerate(dataset["prompt"]):
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        length = len(token_ids)
        observed_max = max(observed_max, length)
        if length > max_length:
            raise ValueError(
                f"RL prompt row {index} has {length} tokens, exceeding cap {max_length}"
            )
    return observed_max
