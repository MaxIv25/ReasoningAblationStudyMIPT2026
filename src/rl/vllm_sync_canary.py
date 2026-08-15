"""Fail-closed verification of colocated TRL -> vLLM weight synchronization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


_PROJECTION_SUFFIXES = (
    "o_proj.weight",
    "out_proj.weight",
    "down_proj.weight",
    "up_proj.weight",
)


def _current_optimizer_lr(trainer: Any) -> float | None:
    """Return the largest absolute LR that will be used by the next actor step."""
    optimizer = getattr(trainer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return None
    learning_rates = [
        abs(float(group["lr"])) for group in param_groups if "lr" in group
    ]
    return max(learning_rates) if learning_rates else None


def vllm_resident_weight_name(policy_name: str) -> str:
    """Map a Qwen3.5 HF policy namespace to the resident vLLM namespace."""
    checkpoint_name = vllm_checkpoint_weight_name(policy_name)
    return checkpoint_name.replace(
        "model.language_model.", "language_model.model.", 1
    )


def vllm_checkpoint_weight_name(policy_name: str) -> str:
    """Map the text-only actor namespace to Qwen3.5 conditional HF keys."""
    if policy_name.startswith("model.language_model."):
        return policy_name
    if policy_name.startswith("model."):
        return "model.language_model." + policy_name[len("model.") :]
    return policy_name


def vllm_load_weight_name(policy_name: str, resident_names: Any) -> str:
    """Choose the input namespace expected by the active vLLM model loader."""
    is_conditional_qwen = any(
        name.startswith("language_model.model.") for name in resident_names
    )
    if is_conditional_qwen:
        return vllm_checkpoint_weight_name(policy_name)
    return policy_name


def capture_vllm_synced_weight(
    sync_weights: Any,
    llm_model: Any,
) -> tuple[str, Any, Any]:
    """Capture one tensor actually passed to vLLM and return its resident peer.

    The wrapper does not merge or unmerge PEFT adapters itself. It observes the
    tensor produced by TRL's own sync path, then calls the original
    ``load_weights`` implementation exactly once.
    """
    resident_parameters = dict(llm_model.named_parameters())
    original_load_weights = llm_model.load_weights
    captured: tuple[str, Any] | None = None

    def recording_load_weights(weights: Any) -> Any:
        nonlocal captured
        entries = list(weights)
        mapped_entries = []
        for policy_name, policy_tensor in entries:
            load_name = vllm_load_weight_name(
                policy_name, resident_parameters
            )
            mapped_entries.append((load_name, policy_tensor))

        if captured is None:
            for policy_name, policy_tensor in entries:
                resident_name = vllm_resident_weight_name(policy_name)
                resident_tensor = resident_parameters.get(resident_name)
                if (
                    resident_tensor is not None
                    and resident_tensor.shape == policy_tensor.shape
                    and resident_name.endswith(_PROJECTION_SUFFIXES)
                ):
                    captured = (resident_name, policy_tensor.detach().clone())
                    break
        try:
            return original_load_weights(mapped_entries)
        except Exception as error:
            mapping = [
                {
                    "policy_name": policy_name,
                    "load_name": load_name,
                    "resident_name": vllm_resident_weight_name(policy_name),
                }
                for (policy_name, _), (load_name, _) in zip(entries, mapped_entries)
            ]
            raise RuntimeError(
                f"vLLM weight namespace mapping failed: {mapping}"
            ) from error

    llm_model.load_weights = recording_load_weights
    try:
        sync_weights()
    finally:
        llm_model.load_weights = original_load_weights

    if captured is None:
        raise RuntimeError(
            "No directly comparable projection tensor was passed to resident vLLM"
        )
    tensor_name, loaded_tensor = captured
    return tensor_name, loaded_tensor, resident_parameters[tensor_name]


class VLLMSyncCanary:
    """Verify resident weights at every colocated vLLM sync event."""

    def __init__(
        self,
        trainer: Any,
        *,
        output_dir: str | Path,
        logger: Any,
        require_change_after_step: bool = True,
        max_consecutive_unchanged_steps: int = 8,
    ) -> None:
        generation = trainer.vllm_generation
        if generation.mode != "colocate":
            raise ValueError("Resident weight verification supports vLLM colocate only")

        self.trainer = trainer
        self.generation = generation
        self.logger = logger
        self.require_change_after_step = require_change_after_step
        self.max_consecutive_unchanged_steps = int(
            max_consecutive_unchanged_steps
        )
        if self.max_consecutive_unchanged_steps < 1:
            raise ValueError("max_consecutive_unchanged_steps must be positive")
        self.report_path = Path(output_dir) / "vllm_sync_checks.jsonl"
        self.original_sync_weights = generation.sync_weights
        self.previous_loaded_tensor: torch.Tensor | None = None
        self.previous_global_step: int | None = None
        self.previous_optimizer_lr: float | None = None
        self.consecutive_unchanged_steps = 0
        self.checks: list[dict[str, Any]] = []

    def install(self) -> None:
        self.trainer.vllm_weight_sync_checks = self.checks
        self.generation.sync_weights = self.sync_weights_and_verify
        self._install_post_reload_resync_guard()

    def _install_post_reload_resync_guard(self) -> None:
        """Make the final pre-generation weights the current actor policy.

        TRL 1.2 first synchronizes the merged PEFT policy, but its sleep-mode
        ``generate`` path then calls vLLM 0.19 ``reload_weights``. That reload
        reads the original model path from disk and overwrites the synchronized
        actor immediately before generation. Defer the trainer's early sync and
        perform exactly one verified sync after every compatibility reload.
        """
        if not bool(getattr(self.generation, "enable_sleep_mode", False)):
            return
        llm = self.generation.llm
        if getattr(llm, "_policy_post_reload_resync_installed", False):
            return

        verified_sync = self.generation.sync_weights
        original_collective_rpc = llm.collective_rpc
        trainer = self.trainer
        logger = self.logger
        trainer.vllm_deferred_syncs = 0
        trainer.vllm_post_reload_resyncs = 0

        def defer_sync_until_after_reload() -> None:
            trainer.vllm_deferred_syncs += 1
            logger.info(
                "vLLM policy sync deferred until post-reload: count=%d step=%d",
                trainer.vllm_deferred_syncs,
                int(trainer.state.global_step),
            )

        def collective_rpc_with_policy_resync(method, *args, **kwargs):
            try:
                result = original_collective_rpc(method, *args, **kwargs)
            except NotImplementedError:
                if method == "reload_weights":
                    verified_sync()
                raise
            if method == "reload_weights":
                verified_sync()
                trainer.vllm_post_reload_resyncs += 1
                logger.info(
                    "vLLM post-reload policy resync: count=%d step=%d",
                    trainer.vllm_post_reload_resyncs,
                    int(trainer.state.global_step),
                )
            return result

        self.generation.sync_weights = defer_sync_until_after_reload
        llm.collective_rpc = collective_rpc_with_policy_resync
        llm._policy_post_reload_resync_installed = True

    def _record(self, record: dict[str, Any]) -> None:
        self.checks.append(record)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @torch.no_grad()
    def sync_weights_and_verify(self) -> None:
        started = time.perf_counter()
        global_step = int(self.trainer.state.global_step)
        optimizer_lr_before_update = _current_optimizer_lr(self.trainer)
        llm_model = (
            self.generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        try:
            tensor_name, loaded_tensor, resident_parameter = capture_vllm_synced_weight(
                self.original_sync_weights,
                llm_model,
            )
            difference = (
                resident_parameter.detach().float() - loaded_tensor.float()
            ).abs()
            resident_matches = bool(
                torch.allclose(
                    resident_parameter.detach(), loaded_tensor, rtol=1e-4, atol=1e-5
                )
            )
            loaded_cpu = loaded_tensor.detach().cpu().clone()
            step_advanced = (
                self.previous_global_step is not None
                and global_step > self.previous_global_step
            )
            weight_changed = (
                None
                if self.previous_loaded_tensor is None
                else not torch.equal(loaded_cpu, self.previous_loaded_tensor)
            )
            optimizer_lr_used = (
                self.previous_optimizer_lr if step_advanced else None
            )
            parameter_change_expected = (
                None
                if not step_advanced
                else optimizer_lr_used is None or optimizer_lr_used > 0.0
            )
            if step_advanced:
                if weight_changed:
                    self.consecutive_unchanged_steps = 0
                elif parameter_change_expected:
                    self.consecutive_unchanged_steps += 1
                else:
                    self.consecutive_unchanged_steps = 0
            record = {
                "event_index": len(self.checks),
                "global_step_before_generation": global_step,
                "elapsed_seconds": time.perf_counter() - started,
                "tensor_name": tensor_name,
                "max_abs_difference": difference.max().item(),
                "mean_abs_difference": difference.mean().item(),
                "resident_matches_synced_tensor": resident_matches,
                "optimizer_step_advanced": step_advanced,
                "optimizer_lr_before_next_update": optimizer_lr_before_update,
                "optimizer_lr_used_since_previous_sync": optimizer_lr_used,
                "parameter_change_expected": parameter_change_expected,
                "synced_tensor_changed_since_previous": weight_changed,
                "consecutive_unchanged_steps": self.consecutive_unchanged_steps,
                "status": "ok",
            }
            if not resident_matches:
                record["status"] = "resident_mismatch"
            elif (
                self.require_change_after_step
                and parameter_change_expected
                and not weight_changed
                and self.consecutive_unchanged_steps
                >= self.max_consecutive_unchanged_steps
            ):
                record["status"] = "stalled_source_after_optimizer_steps"
            elif parameter_change_expected and not weight_changed:
                record["status"] = "ok_source_unchanged"
            elif step_advanced and not parameter_change_expected and not weight_changed:
                record["status"] = "ok_no_parameter_change_expected"
            self._record(record)
            self.previous_loaded_tensor = loaded_cpu
            self.previous_global_step = global_step
            self.previous_optimizer_lr = optimizer_lr_before_update
        except Exception as error:
            self._record(
                {
                    "event_index": len(self.checks),
                    "global_step_before_generation": global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            raise

        self.logger.info(
            "vLLM sync canary: step=%d tensor=%s max_diff=%.3g changed=%s unchanged_streak=%d",
            global_step,
            tensor_name,
            record["max_abs_difference"],
            weight_changed,
            self.consecutive_unchanged_steps,
        )
        if record["status"] == "resident_mismatch":
            raise RuntimeError(
                f"Resident vLLM tensor {tensor_name} does not match the synced tensor"
            )
        if record["status"] == "stalled_source_after_optimizer_steps":
            raise RuntimeError(
                f"Synced vLLM tensor {tensor_name} remained unchanged for "
                f"{self.consecutive_unchanged_steps} optimizer steps"
            )


def install_vllm_sync_canary(
    trainer: Any,
    *,
    output_dir: str | Path,
    logger: Any,
    enabled: bool,
    require_change_after_step: bool = True,
    max_consecutive_unchanged_steps: int = 8,
) -> VLLMSyncCanary | None:
    """Install and return the canary when explicitly enabled."""
    if not enabled:
        return None
    canary = VLLMSyncCanary(
        trainer,
        output_dir=output_dir,
        logger=logger,
        require_change_after_step=require_change_after_step,
        max_consecutive_unchanged_steps=max_consecutive_unchanged_steps,
    )
    canary.install()
    return canary
