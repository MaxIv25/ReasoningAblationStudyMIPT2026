from types import SimpleNamespace

import torch

from src.rl.vllm_sync_canary import install_vllm_sync_canary


class FakeResidentModel:
    def __init__(self):
        self.weight = torch.zeros(2, 2)

    def named_parameters(self):
        return [
            (
                "language_model.model.layers.0.linear_attn.out_proj.weight",
                self.weight,
            )
        ]

    def load_weights(self, weights):
        for name, tensor in weights:
            assert name == "model.language_model.layers.0.linear_attn.out_proj.weight"
            self.weight.copy_(tensor)


class FakeLLM:
    def __init__(self, model):
        self.model = model
        self.disk_reloads = 0
        self.llm_engine = SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=model)
                )
            )
        )

    def collective_rpc(self, method, *args, **kwargs):
        assert method == "reload_weights"
        self.disk_reloads += 1
        self.model.weight.zero_()  # vLLM 0.19 reloads the original disk checkpoint.
        return [None]


def test_sleep_mode_disk_reload_is_followed_by_current_policy_resync(tmp_path):
    model = FakeResidentModel()
    llm = FakeLLM(model)
    policy_weight = torch.full((2, 2), 7.0)

    def sync_weights():
        model.load_weights(
            [("model.layers.0.linear_attn.out_proj.weight", policy_weight)]
        )

    generation = SimpleNamespace(
        mode="colocate",
        enable_sleep_mode=True,
        sync_weights=sync_weights,
        llm=llm,
    )
    trainer = SimpleNamespace(
        vllm_generation=generation,
        state=SimpleNamespace(global_step=0),
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    install_vllm_sync_canary(
        trainer, output_dir=tmp_path, logger=logger, enabled=True
    )

    generation.sync_weights()
    assert torch.equal(model.weight, torch.zeros_like(policy_weight))
    assert trainer.vllm_deferred_syncs == 1

    # This is the call made inside TRL VLLMGeneration.generate immediately
    # before llm.generate. It must not leave the original disk weights resident.
    llm.collective_rpc("reload_weights")

    assert llm.disk_reloads == 1
    assert torch.equal(model.weight, policy_weight)
    assert trainer.vllm_post_reload_resyncs == 1
    assert trainer.vllm_weight_sync_checks[-1]["resident_matches_synced_tensor"]
