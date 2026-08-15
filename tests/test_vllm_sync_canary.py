from types import SimpleNamespace

import pytest
import torch

from src.rl.vllm_sync_canary import (
    capture_vllm_synced_weight,
    install_vllm_sync_canary,
    vllm_checkpoint_weight_name,
    vllm_load_weight_name,
    vllm_resident_weight_name,
)


class FakeVLLMModel:
    def __init__(self):
        self.resident = torch.zeros(2, 2)

    def named_parameters(self):
        return [
            (
                "language_model.model.layers.0.linear_attn.out_proj.weight",
                self.resident,
            )
        ]

    def load_weights(self, weights):
        for name, tensor in weights:
            assert name == "model.language_model.layers.0.linear_attn.out_proj.weight"
            self.resident.copy_(tensor)


def test_qwen35_policy_name_maps_to_resident_vllm_namespace():
    assert vllm_resident_weight_name(
        "model.layers.0.linear_attn.out_proj.weight"
    ) == "language_model.model.layers.0.linear_attn.out_proj.weight"
    assert vllm_checkpoint_weight_name(
        "model.layers.0.linear_attn.out_proj.weight"
    ) == "model.language_model.layers.0.linear_attn.out_proj.weight"


def test_conditional_vllm_maps_unfused_actor_weight_even_without_exact_resident():
    resident_names = {
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"
    }

    assert vllm_load_weight_name(
        "model.layers.0.linear_attn.in_proj_qkv.weight", resident_names
    ) == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"


def test_capture_observes_tensor_actually_loaded_by_trl():
    model = FakeVLLMModel()

    def sync_weights():
        model.load_weights(
            [
                (
                    "model.layers.0.linear_attn.out_proj.weight",
                    torch.full((2, 2), 7.0),
                )
            ]
        )

    name, loaded, resident = capture_vllm_synced_weight(sync_weights, model)

    assert name == "language_model.model.layers.0.linear_attn.out_proj.weight"
    assert torch.equal(loaded, resident)
    assert torch.equal(resident, torch.full((2, 2), 7.0))


def test_canary_detects_stale_tensor_after_optimizer_step(tmp_path):
    model = FakeVLLMModel()
    next_weight = torch.ones(2, 2)

    def sync_weights():
        model.load_weights(
            [
                (
                    "model.layers.0.linear_attn.out_proj.weight",
                    next_weight,
                )
            ]
        )

    generation = SimpleNamespace(
        mode="colocate",
        sync_weights=sync_weights,
        llm=SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(model=model)
                    )
                )
            )
        ),
    )
    trainer = SimpleNamespace(
        vllm_generation=generation,
        state=SimpleNamespace(global_step=0),
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    install_vllm_sync_canary(
        trainer,
        output_dir=tmp_path,
        logger=logger,
        enabled=True,
        max_consecutive_unchanged_steps=2,
    )

    generation.sync_weights()
    next_weight.fill_(2.0)
    trainer.state.global_step = 1
    generation.sync_weights()
    assert trainer.vllm_weight_sync_checks[-1]["status"] == "ok"
    assert trainer.vllm_weight_sync_checks[-1][
        "synced_tensor_changed_since_previous"
    ]

    trainer.state.global_step = 2
    generation.sync_weights()
    assert trainer.vllm_weight_sync_checks[-1]["status"] == "ok_source_unchanged"
    assert trainer.vllm_weight_sync_checks[-1]["consecutive_unchanged_steps"] == 1

    trainer.state.global_step = 3
    with pytest.raises(RuntimeError, match="unchanged for 2 optimizer steps"):
        generation.sync_weights()

    assert trainer.vllm_weight_sync_checks[-1]["status"] == (
        "stalled_source_after_optimizer_steps"
    )


def test_canary_allows_unchanged_tensor_after_zero_lr_warmup_step(tmp_path):
    model = FakeVLLMModel()
    next_weight = torch.ones(2, 2)

    def sync_weights():
        model.load_weights(
            [
                (
                    "model.layers.0.linear_attn.out_proj.weight",
                    next_weight,
                )
            ]
        )

    generation = SimpleNamespace(
        mode="colocate",
        sync_weights=sync_weights,
        llm=SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(model=model)
                    )
                )
            )
        ),
    )
    trainer = SimpleNamespace(
        vllm_generation=generation,
        state=SimpleNamespace(global_step=0),
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    install_vllm_sync_canary(
        trainer, output_dir=tmp_path, logger=logger, enabled=True
    )

    generation.sync_weights()
    trainer.state.global_step = 1
    generation.sync_weights()

    check = trainer.vllm_weight_sync_checks[-1]
    assert check["status"] == "ok_no_parameter_change_expected"
    assert check["optimizer_lr_used_since_previous_sync"] == 0.0
    assert check["parameter_change_expected"] is False
