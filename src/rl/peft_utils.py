"""Shared PEFT configuration helpers for actor and PRM adapters."""


def build_lora_config(config: dict, *, enabled_key: str | None = None):
    enabled = bool(config.get("enabled", False))
    if enabled_key is not None:
        enabled = enabled and bool(config.get(enabled_key, False))
    if not enabled:
        return None

    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        lora_dropout=float(config.get("dropout", 0.0)),
        target_modules=config.get("target_modules", "all-linear"),
        bias=config.get("bias", "none"),
    )
