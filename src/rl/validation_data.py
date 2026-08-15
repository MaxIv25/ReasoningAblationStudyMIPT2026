"""Build fixed, train-disjoint validation datasets for single-turn math RL."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from datasets import Dataset


_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n"


def render_preformatted_prompt(prompt: str | Sequence[Mapping[str, str]]) -> str:
    """Normalize an existing raw prompt or a legacy chat row to Qwen text form."""
    if isinstance(prompt, str):
        return prompt
    user_messages = [message for message in prompt if message.get("role") == "user"]
    if len(user_messages) != 1 or not user_messages[0].get("content"):
        raise ValueError("Validation candidates require exactly one non-empty user turn")
    return (
        f"<|im_start|>user\n{user_messages[0]['content']}<|im_end|>\n"
        f"{_ASSISTANT_PREFIX}"
    )


def build_unseen_validation_dataset(
    candidates: Dataset,
    train_dataset: Dataset,
    *,
    size: int,
    seed: int,
) -> Dataset:
    """Select a deterministic prompt-disjoint validation set from candidates."""
    if size <= 0:
        raise ValueError("Validation size must be positive")
    train_prompts = {
        render_preformatted_prompt(prompt) for prompt in train_dataset["prompt"]
    }
    seen = set(train_prompts)
    available: list[dict[str, str]] = []
    for row in candidates:
        prompt = render_preformatted_prompt(row["prompt"])
        solution = row["solution"]
        if prompt in seen:
            continue
        if not isinstance(solution, str) or not solution.strip():
            continue
        seen.add(prompt)
        available.append({"prompt": prompt, "solution": solution})

    if len(available) < size:
        raise ValueError(
            f"Only {len(available)} train-disjoint validation rows are available; "
            f"requested {size}"
        )
    random.Random(seed).shuffle(available)
    return Dataset.from_list(available[:size])
