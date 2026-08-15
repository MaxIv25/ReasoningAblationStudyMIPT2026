from types import SimpleNamespace

from src.rl.model_utils import (
    _patch_vllm_language_only_multimodal_detection,
    _register_vllm_qwen35_text_model,
)


def test_qwen35_text_model_is_registered_with_vllm_native_class():
    from vllm.model_executor.models import ModelRegistry

    _register_vllm_qwen35_text_model()

    registered = ModelRegistry.models["Qwen3_5ForCausalLM"]
    assert registered.module_name == "vllm.model_executor.models.qwen3_5"
    assert registered.class_name == "Qwen3_5ForCausalLM"


def test_qwen35_text_model_registration_is_idempotent():
    from vllm.model_executor.models import ModelRegistry

    _register_vllm_qwen35_text_model()
    first = ModelRegistry.models["Qwen3_5ForCausalLM"]
    _register_vllm_qwen35_text_model()

    assert ModelRegistry.models["Qwen3_5ForCausalLM"] is first


def test_vllm_language_only_is_not_routed_through_multimodal_renderer():
    from vllm.config import ModelConfig

    _patch_vllm_language_only_multimodal_detection()
    getter = ModelConfig.is_multimodal_model.fget

    language_only = SimpleNamespace(
        multimodal_config=SimpleNamespace(language_model_only=True)
    )
    multimodal = SimpleNamespace(
        multimodal_config=SimpleNamespace(language_model_only=False)
    )
    text_only = SimpleNamespace(multimodal_config=None)

    assert getter(language_only) is False
    assert getter(multimodal) is True
    assert getter(text_only) is False


def test_vllm_language_only_multimodal_patch_is_idempotent():
    from vllm.config import ModelConfig

    _patch_vllm_language_only_multimodal_detection()
    first_getter = ModelConfig.is_multimodal_model.fget
    _patch_vllm_language_only_multimodal_detection()

    assert ModelConfig.is_multimodal_model.fget is first_getter
