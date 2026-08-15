"""Qwen3.5 text-only loading helpers shared by GRPO and PRIME."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _register_vllm_qwen35_text_model():
    """Register vLLM's shipped Qwen3.5 causal class under its HF arch name.

    vLLM 0.19.1 ships ``Qwen3_5ForCausalLM`` but omits it from
    ``ModelRegistry``. The generic architecture normalizer consequently maps
    the HF causal architecture to ``Qwen3_5ForConditionalGeneration`` and then
    tries to construct the vision tower from a ``Qwen3_5TextConfig``.
    """
    from vllm.model_executor.models import ModelRegistry

    architecture = "Qwen3_5ForCausalLM"
    if architecture in ModelRegistry.models:
        return
    ModelRegistry.register_model(
        architecture,
        "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
    )


def _patch_vllm_language_only_multimodal_detection():
    """Keep vLLM 0.19's renderer out of the multimodal path in LM-only mode.

    vLLM 0.19.1 creates ``MultiModalConfig`` for a model architecture that is
    multimodal-capable even when ``language_model_only=True``. Its
    ``is_multimodal_model`` property only checks whether that object exists, so
    the renderer tries to build a Qwen3.5 vision processor from the already
    extracted ``Qwen3_5TextConfig`` and fails before the engine starts.

    Preserve the original behaviour for real multimodal runs and only suppress
    multimodal routing when vLLM's own language-only flag is active.
    """
    from vllm.config import ModelConfig

    original = ModelConfig.is_multimodal_model
    original_getter = original.fget
    if getattr(original_getter, "_prime_language_only_guard", False):
        return

    def is_active_multimodal_model(model_config):
        multimodal_config = getattr(model_config, "multimodal_config", None)
        if multimodal_config is not None and getattr(
            multimodal_config, "language_model_only", False
        ):
            return False
        return original_getter(model_config)

    is_active_multimodal_model._prime_language_only_guard = True
    ModelConfig.is_multimodal_model = property(
        is_active_multimodal_model,
        original.fset,
        original.fdel,
        original.__doc__,
    )


def load_text_causal_lm(
    model_id: str,
    *,
    dtype=torch.bfloat16,
    device="cpu",
    attn_implementation: str = "sdpa",
    logger=None,
):
    full_config = AutoConfig.from_pretrained(model_id)
    text_config = (
        full_config.get_text_config()
        if hasattr(full_config, "get_text_config")
        else full_config
    )
    if logger and getattr(text_config, "model_type", None) != getattr(
        full_config, "model_type", None
    ):
        logger.info(
            "Loading text-only config %s from multimodal %s",
            text_config.model_type,
            full_config.model_type,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        config=text_config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)


def patch_vllm_language_model_only(enabled: bool, *, logger=None):
    """Forward Qwen3.5's official language-only flag through TRL colocate mode."""
    if not enabled:
        return
    _register_vllm_qwen35_text_model()
    _patch_vllm_language_only_multimodal_detection()
    import trl.generation.vllm_generation as trl_vllm

    original = trl_vllm.LLM
    if getattr(original, "_prime_language_only_patch", False):
        return

    class LanguageOnlyLLM(original):
        _prime_language_only_patch = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("language_model_only", True)
            super().__init__(*args, **kwargs)

    trl_vllm.LLM = LanguageOnlyLLM
    if logger:
        logger.info(
            "Enabled vLLM language_model_only and text-renderer guard for "
            "Qwen3.5 colocate generation"
        )
