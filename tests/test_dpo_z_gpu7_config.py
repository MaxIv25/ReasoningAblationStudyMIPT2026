from pathlib import Path

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_dpo_z_gpu7_full_run_changes_only_vllm_reservation():
    base = str(ROOT / "configs" / "grpo_base.yaml")
    dpo_z = load_config(str(ROOT / "configs" / "grpo_dpo_z.yaml"), base)
    gpu7 = load_config(str(ROOT / "configs" / "grpo_dpo_z_gpu7.yaml"), base)

    assert gpu7["run_name"] == "grpo_dpo_z_full_gpu7"
    assert gpu7["grpo"]["vllm_gpu_memory_utilization"] == 0.20
    assert gpu7["grpo"]["advantage_baseline"] == "dpo_z"
    assert gpu7["grpo"]["dpo_z_beta"] == 0.1
    assert gpu7["grpo"]["dpo_z_leave_one_out"] is True
    assert gpu7["grpo"]["generation_batch_size"] == 64
    assert gpu7["grpo"]["max_completion_length"] == 16384
    assert gpu7["grpo"]["memory_bounded_policy_loss"] is True
    assert gpu7["training"]["per_device_train_batch_size"] == 8
    assert gpu7["training"]["gradient_accumulation_steps"] == 8
    assert gpu7["training"]["learning_rate"] == 1.5e-6
    assert gpu7["model"] == dpo_z["model"]

    expected_grpo = dict(dpo_z["grpo"])
    expected_grpo["vllm_gpu_memory_utilization"] = 0.20
    assert gpu7["grpo"] == expected_grpo
    assert gpu7["training"] == dpo_z["training"]
    assert gpu7["peft"] == dpo_z["peft"]
