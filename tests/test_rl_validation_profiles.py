from copy import deepcopy
from pathlib import Path

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_full_rl_validation_profiles_share_the_on_policy_contract():
    for filename in (
        "grpo_vanilla_with_val.yaml",
        "grpo_dpo_z_gpu7_with_val.yaml",
        "prime_research_16k_with_val.yaml",
    ):
        cfg = load_config(
            str(ROOT / "configs" / filename),
            base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
        )
        validation = cfg["validation"]
        assert validation["data_dir"] == "data/grpo_validation_unseen_100_v1"
        assert validation["num_samples"] == 100
        assert validation["eval_steps"] == 20
        assert validation["num_generations"] == 1
        assert validation["eval_on_start"] is True
        assert cfg["grpo"]["temperature"] == 1.0
        assert cfg["grpo"]["top_p"] == 1.0
        assert cfg["grpo"]["top_k"] == 0


def test_vanilla_constant_scheduler_profile_is_a_single_variable_ablation():
    cosine = load_config(
        str(ROOT / "configs" / "grpo_vanilla_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    constant = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )

    assert cosine["training"]["lr_scheduler_type"] == "cosine"
    assert cosine["training"]["warmup_ratio"] == 0.05
    assert constant["training"]["lr_scheduler_type"] == "constant"
    assert constant["training"]["warmup_ratio"] == 0.0
    assert constant["training"]["learning_rate"] == 1.5e-6

    cosine_contract = deepcopy(cosine)
    constant_contract = deepcopy(constant)
    cosine_contract.pop("run_name")
    constant_contract.pop("run_name")
    for contract in (cosine_contract, constant_contract):
        contract["training"].pop("lr_scheduler_type")
        contract["training"].pop("warmup_ratio")
    assert constant_contract == cosine_contract


def test_vanilla_constant_lr5e6_profile_changes_only_learning_rate():
    lr1e5 = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    lr5e6 = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_lr5e6_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )

    assert lr1e5["training"]["learning_rate"] == 1.5e-6
    assert lr5e6["training"]["learning_rate"] == 5.0e-6
    assert lr5e6["training"]["lr_scheduler_type"] == "constant"
    assert lr5e6["training"]["warmup_ratio"] == 0.0

    lr1e5_contract = deepcopy(lr1e5)
    lr5e6_contract = deepcopy(lr5e6)
    lr1e5_contract.pop("run_name")
    lr5e6_contract.pop("run_name")
    for contract in (lr1e5_contract, lr5e6_contract):
        contract["training"].pop("learning_rate")
    assert lr5e6_contract == lr1e5_contract


def test_recoverable_gpu7_profile_changes_only_resource_and_save_settings():
    treatment = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_lr5e6_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    gpu7 = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_lr5e6_gpu7_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )

    assert gpu7["training"]["learning_rate"] == 5.0e-6
    assert gpu7["training"]["lr_scheduler_type"] == "constant"
    assert gpu7["grpo"]["vllm_gpu_memory_utilization"] == 0.20
    assert gpu7["training"]["save_steps"] == 20

    treatment_contract = deepcopy(treatment)
    gpu7_contract = deepcopy(gpu7)
    treatment_contract.pop("run_name")
    gpu7_contract.pop("run_name")
    treatment_contract["grpo"].pop("vllm_gpu_memory_utilization")
    gpu7_contract["grpo"].pop("vllm_gpu_memory_utilization")
    for contract in (treatment_contract, gpu7_contract):
        contract["training"].pop("save_steps")
        contract["training"].pop("save_total_limit")
    assert gpu7_contract == treatment_contract


def test_opt_lr1e5_profile_changes_only_lr_and_save_settings():
    control = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    lr1e5 = load_config(
        str(ROOT / "configs" / "grpo_vanilla_constant_lr1e5_opt_with_val.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )

    assert lr1e5["training"]["learning_rate"] == 1.0e-5
    assert lr1e5["training"]["lr_scheduler_type"] == "constant"
    assert lr1e5["training"]["warmup_ratio"] == 0.0
    assert lr1e5["training"]["save_steps"] == 20

    control_contract = deepcopy(control)
    lr1e5_contract = deepcopy(lr1e5)
    control_contract.pop("run_name")
    lr1e5_contract.pop("run_name")
    for contract in (control_contract, lr1e5_contract):
        contract["training"].pop("learning_rate")
        contract["training"].pop("save_steps")
        contract["training"].pop("save_total_limit")
    assert lr1e5_contract == control_contract
