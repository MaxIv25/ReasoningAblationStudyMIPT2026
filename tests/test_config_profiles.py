from pathlib import Path
import ast

import yaml

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    with (ROOT / "configs" / name).open() as handle:
        return yaml.safe_load(handle)


def test_faithful_profile_matches_published_prime_launch_defaults():
    cfg = _load("prime_faithful.yaml")
    assert cfg["model"]["name"] == "Qwen/Qwen3.5-0.8B"
    assert cfg["grpo"]["num_generations"] == 4
    assert cfg["execution_target"] == "literal_public_rloo"
    assert cfg["grpo"]["max_completion_length"] == 3072
    assert cfg["training"]["learning_rate"] == 5e-7
    assert cfg["training"]["weight_decay"] == 0.0
    assert cfg["prime"]["prm_lr"] == 1e-6
    assert cfg["published_reference"]["declared_rm_coef"] == 5.0
    assert cfg["published_reference"]["effective_rloo_rm_coef"] == 1.0
    assert cfg["prime"]["rm_coef"] == 1.0
    assert cfg["prime"]["prm_grad_clip"] == 10.0
    assert cfg["prime"]["filter_accuracy"] is True
    assert cfg["prime"]["filter_lower"] == 0.2
    assert cfg["prime"]["filter_upper"] == 0.8
    assert cfg["prime"]["prm_update_timing"] == "after"
    assert cfg["prime"]["filter_refill"] is True
    assert cfg["grpo"]["vllm_gpu_memory_utilization"] <= 0.12
    assert cfg["peft"]["enabled"] is False


def test_16k_profile_is_explicitly_not_the_reproduction_profile():
    cfg = _load("prime_research_16k.yaml")
    assert cfg["profile"] == "research_16k"
    assert cfg["grpo"]["max_completion_length"] == 16384
    assert cfg["memory"]["logprob_chunk_tokens"] <= 512
    assert cfg["peft"]["enabled"] is True
    assert cfg["prime"]["rm_coef"] == 5.0
    assert cfg["model"]["name"].endswith("sft_lora_r64_16k_two_epochs_merged")


def test_prime_research_actor_lr_matches_vanilla_grpo_but_prm_lr_is_independent():
    vanilla = load_config(
        str(ROOT / "configs" / "grpo_vanilla_sft_lora.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    prime = load_config(
        str(ROOT / "configs" / "prime_research_16k.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert prime["training"]["learning_rate"] == vanilla["training"]["learning_rate"]
    assert prime["training"]["per_device_train_batch_size"] == 8
    assert prime["training"]["gradient_accumulation_steps"] == 8
    assert prime["grpo"]["vllm_gpu_memory_utilization"] == 0.25
    assert prime["prime"]["prm_lr"] == 1.0e-5
    assert prime["prime"]["prm_ref_batch_size"] == 8
    assert prime["prime"]["prm_grad_batch_size"] == 4

    gate = load_config(
        str(ROOT / "configs" / "prime_research_16k_matched_lr_gate.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert gate["training"]["learning_rate"] == 1.5e-6
    assert gate["training"]["max_steps"] == 1


def test_idea_overlays_inherit_lora_16k_research_contract():
    cfg = load_config(
        str(ROOT / "configs" / "prime_idea_dpo_z.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert cfg["grpo"]["max_prompt_length"] == 1024
    assert cfg["grpo"]["max_completion_length"] == 16384
    assert cfg["prime"]["filter_refill"] is True
    assert cfg["prime"]["advantage_baseline"] == "dpo_z"
    assert cfg["peft"]["enabled"] is True
    assert cfg["peft"]["actor"] is True
    assert cfg["peft"]["prm"] is True
    assert cfg["model"]["name"].endswith("sft_lora_r64_16k_two_epochs_merged")


def test_grpo_and_prime_smokes_run_two_sync_cycles_on_the_same_checkpoint():
    grpo = load_config(
        str(ROOT / "configs" / "grpo_vanilla_sft_lora_smoke.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    prime = load_config(
        str(ROOT / "configs" / "prime_research_16k_smoke.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert grpo["model"]["name"] == prime["model"]["name"]
    assert grpo["grpo"]["format_reward_weight"] == 0.0
    for cfg in (grpo, prime):
        assert cfg["training"]["max_steps"] == 2
        assert cfg["grpo"]["generation_batch_size"] == 8
        assert cfg["training"]["gradient_accumulation_steps"] == 8
        assert cfg["training"]["lr_scheduler_type"] == "constant"
        assert cfg["training"]["warmup_ratio"] == 0.0


def test_dpo_z_smoke_keeps_token_bounded_two_step_contract():
    cfg = load_config(
        str(ROOT / "configs" / "grpo_dpo_z_smoke.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert cfg["grpo"]["advantage_baseline"] == "dpo_z"
    assert cfg["grpo"]["memory_bounded_policy_loss"] is True
    assert cfg["grpo"]["use_liger_kernel"] is False
    assert cfg["grpo"]["generation_batch_size"] == 8
    assert cfg["training"]["gradient_accumulation_steps"] == 8
    assert cfg["training"]["max_steps"] == 2



def test_dpo_z_grpo_inherits_the_same_vanilla_baseline_contract():
    vanilla = load_config(
        str(ROOT / "configs" / "grpo_vanilla_sft_lora.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    dpo_z = load_config(
        str(ROOT / "configs" / "grpo_dpo_z.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    for key in ("name", "attn_implementation"):
        assert dpo_z["model"][key] == vanilla["model"][key]
    for key in ("num_samples", "num_generations", "max_completion_length"):
        assert dpo_z["grpo"][key] == vanilla["grpo"][key]
    assert dpo_z["grpo"]["advantage_baseline"] == "dpo_z"
    assert dpo_z["grpo"]["memory_bounded_policy_loss"] is True
    assert dpo_z["grpo"]["use_liger_kernel"] is False
    assert dpo_z["grpo"]["logprob_chunk_tokens"] <= 256


def test_trl_12_grpo_config_calls_do_not_use_removed_prompt_length_kwarg():
    for filename in ("train_grpo.py", "train_prime.py"):
        tree = ast.parse((ROOT / "src" / filename).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "GRPOConfig"
        ]
        assert len(calls) == 1
        assert "max_prompt_length" not in {kw.arg for kw in calls[0].keywords}


def test_vanilla_grpo_uses_token_bounded_policy_loss():
    cfg = load_config(
        str(ROOT / "configs" / "grpo_vanilla_sft_lora.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert cfg["grpo"]["memory_bounded_policy_loss"] is True
    assert cfg["grpo"]["use_liger_kernel"] is False
    assert cfg["grpo"]["logprob_chunk_tokens"] <= 256
    assert cfg["grpo"]["generation_batch_size"] == 64
    assert cfg["grpo"]["vllm_gpu_memory_utilization"] == 0.25
    assert cfg["training"]["per_device_train_batch_size"] == 8
    assert cfg["training"]["gradient_accumulation_steps"] == 8


def test_grpo_microbatch_profiles_keep_one_full_64_trajectory_update():
    for microbatch in (2, 4, 8, 16):
        cfg = load_config(
            str(ROOT / "configs" / f"grpo_epoch1_profile_g8_mb{microbatch}.yaml"),
            base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
        )
        assert cfg["grpo"]["num_generations"] == 8
        assert cfg["grpo"]["generation_batch_size"] == 64
        assert cfg["model"]["name"].endswith(
            "sft_lora_r64_16k_two_epochs_merged"
        )
        assert cfg["training"]["per_device_train_batch_size"] == microbatch
        assert cfg["training"]["gradient_accumulation_steps"] * microbatch == 64
        assert cfg["training"]["max_steps"] == 1


def test_prime_microbatch_profile_keeps_full_64_trajectory_update():
    cfg = load_config(
        str(ROOT / "configs" / "prime_research_16k_profile_g8_mb8.yaml"),
        base_config_path=str(ROOT / "configs" / "grpo_base.yaml"),
    )
    assert cfg["grpo"]["generation_batch_size"] == 64
    assert cfg["training"]["per_device_train_batch_size"] == 8
    assert cfg["training"]["gradient_accumulation_steps"] == 8
    assert cfg["training"]["learning_rate"] == 1.5e-6
    assert cfg["prime"]["prm_lr"] == 1.0e-5
    assert cfg["prime"]["prm_ref_batch_size"] == 8
    assert cfg["prime"]["prm_grad_batch_size"] == 4
