import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _calls(filename: str, function_name: str):
    tree = ast.parse((ROOT / "src" / filename).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == function_name
    ]


def test_grpo_and_prime_wire_validation_into_trl():
    for filename, trainer_name in (
        ("train_grpo.py", "trainer_cls"),
        ("train_prime.py", "PrimeGRPOTrainer"),
    ):
        config_calls = _calls(filename, "GRPOConfig")
        assert len(config_calls) == 1
        config_keywords = {keyword.arg for keyword in config_calls[0].keywords}
        assert {
            "eval_strategy",
            "eval_steps",
            "eval_on_start",
            "per_device_eval_batch_size",
            "num_generations_eval",
        }.issubset(config_keywords)

        trainer_calls = _calls(filename, trainer_name)
        assert len(trainer_calls) == 1
        trainer_keywords = {keyword.arg for keyword in trainer_calls[0].keywords}
        assert "eval_dataset" in trainer_keywords
