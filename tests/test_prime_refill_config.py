from pathlib import Path

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "grpo_base.yaml"


def _load(name):
    return load_config(str(ROOT / "configs" / name), base_config_path=str(BASE))


def test_research_g8_filter_preserves_public_non_unanimous_criterion():
    config = _load("prime_research_16k.yaml")

    assert config["grpo"]["num_generations"] == 8
    assert config["prime"]["filter_lower"] == 0.1
    assert config["prime"]["filter_upper"] == 0.9
    assert "max_refill_rounds" not in config["prime"]
    assert config["training"]["save_steps"] == 5


def test_refill_smoke_exercises_filter_sync_and_complete_checkpoint():
    config = _load("prime_research_refill_smoke.yaml")

    assert config["training"]["max_steps"] == 2
    assert config["training"]["save_steps"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["grpo"]["generation_batch_size"] == 8
    assert config["grpo"]["max_completion_length"] == 2048
    assert config["grpo"]["verify_vllm_weight_sync"] is True
    assert config["grpo"]["require_vllm_weight_change_after_step"] is True
    assert config["prime"]["filter_accuracy"] is True
    assert config["prime"]["filter_refill"] is True
