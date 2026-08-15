"""Memory-bounded, author-aligned PRIME training on a post-trained causal LM.

The baseline follows the public PRIME launch and implementation: solvability
filtering, pre-update implicit rewards (``update=after``), separate RLOO for
verifier/process sources, global reverse-return normalization, and final masked
whitening. Research extensions are opt-in and logged separately.
"""

import argparse
import gc
import glob
import json
import os
import sys
from pathlib import Path

# Avoid a shared, permission-sensitive /tmp/tvm-debug-mode-tempdirs root.
# TileLang's persistent kernel cache is independent of these compiler temp files.
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rl.chunked_logprobs import entropy_from_hidden, selected_logprobs_from_hidden
from src.rl.dataset_contract import (
    validate_preformatted_math_rl_dataset,
    validate_prompt_token_lengths,
)
from src.rl.guided_search import select_prm_guided_candidates
from src.rl.model_utils import load_text_causal_lm, patch_vllm_language_model_only
from src.rl.peft_utils import build_lora_config
from src.rl.vllm_sync_canary import install_vllm_sync_canary
from src.rl.prime_core import (
    compute_prime_advantages,
    implicit_process_rewards,
    prime_prm_bce_loss,
    non_truncated_group_mask,
    process_reward_weight,
    solvable_group_mask,
)
from src.rl.prompt_calibration import PromptCalibrationHead
from src.utils import (
    extract_boxed_answer,
    get_gpu_memory_info,
    load_config,
    setup_logging,
    verify_answer,
)

logger = setup_logging("train_prime")


def accuracy_reward(completions, solution, log_metric=None, **kwargs):
    rewards = []
    for completion, target in zip(completions, solution):
        content = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        prediction = extract_boxed_answer(content)
        rewards.append(
            float(prediction is not None and verify_answer(prediction, target))
        )
    if log_metric and rewards:
        log_metric("accuracy", sum(rewards) / len(rewards))
    return rewards


class PrimeGRPOTrainer(GRPOTrainer):
    """TRL trainer with author-aligned PRIME rewards and bounded logits memory."""

    def __init__(
        self,
        *,
        prime_cfg: dict,
        memory_cfg: dict,
        peft_cfg: dict,
        model_id: str,
        attn_implementation: str = "sdpa",
        **kwargs,
    ):
        self.prime_cfg = prime_cfg
        self.memory_cfg = memory_cfg
        self.peft_cfg = peft_cfg
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        super().__init__(**kwargs)

        self.prime_beta = float(prime_cfg.get("beta", 0.05))
        self.prime_rm_coef = float(prime_cfg.get("rm_coef", 5.0))
        self.prime_prm_lr = float(prime_cfg.get("prm_lr", 1e-6))
        self.prime_prm_epochs = int(prime_cfg.get("prm_update_epochs", 1))
        self.prime_prm_grad_clip = float(prime_cfg.get("prm_grad_clip", 10.0))
        self.prime_baseline = prime_cfg.get("advantage_baseline", "rloo")
        self.prime_gamma = float(prime_cfg.get("gamma", 1.0))
        self.filter_accuracy = bool(prime_cfg.get("filter_accuracy", True))
        self.filter_lower = float(prime_cfg.get("filter_lower", 0.2))
        self.filter_upper = float(prime_cfg.get("filter_upper", 0.8))
        self.filter_refill = bool(prime_cfg.get("filter_refill", True))
        self.filter_truncated_groups = bool(
            prime_cfg.get("filter_truncated_groups", False)
        )
        self.prm_update_timing = prime_cfg.get("prm_update_timing", "after")
        if self.prm_update_timing != "after":
            raise ValueError("The faithful profile requires prm_update_timing='after'")

        self.logprob_chunk_tokens = int(memory_cfg.get("logprob_chunk_tokens", 256))
        self.checkpoint_logprob_chunks = bool(
            memory_cfg.get("checkpoint_logprob_chunks", True)
        )
        self.cpu_offload_policy = bool(memory_cfg.get("cpu_offload_policy", True))
        self.cpu_offload_aux = bool(memory_cfg.get("cpu_offload_aux", True))
        self.prm_ref_batch_size = int(prime_cfg.get("prm_ref_batch_size", 1))
        self.prm_grad_batch_size = int(prime_cfg.get("prm_grad_batch_size", 1))
        self._prime_step = 0

        self.process_schedule = prime_cfg.get("process_reward_schedule", "constant")
        self.process_warmup_steps = int(prime_cfg.get("process_reward_warmup_steps", 0))
        self.reliability_floor = float(prime_cfg.get("reliability_floor", 0.5))
        self.reliability_full = float(prime_cfg.get("reliability_full", 0.7))

        calibration_cfg = prime_cfg.get("zx_calibrated_prm", {})
        if isinstance(calibration_cfg, bool):
            calibration_cfg = {"enabled": calibration_cfg}
        self.calibration_cfg = calibration_cfg
        self.use_prompt_calibration = bool(calibration_cfg.get("enabled", False))

        guided_cfg = prime_cfg.get("guided_search", {})
        self.guided_cfg = (
            guided_cfg
            if isinstance(guided_cfg, dict)
            else {"enabled": bool(guided_cfg)}
        )
        self.use_guided_search = bool(self.guided_cfg.get("enabled", False))

        aux_device = "cpu" if self.cpu_offload_aux else self.accelerator.device
        self.prm_model = load_text_causal_lm(
            model_id,
            device=aux_device,
            attn_implementation=attn_implementation,
            logger=logger,
        )
        prm_lora = build_lora_config(peft_cfg, enabled_key="prm")
        if prm_lora is not None:
            from peft import get_peft_model

            self.prm_model = get_peft_model(self.prm_model, prm_lora)
            # Frozen embeddings otherwise make checkpointed LoRA blocks lose grads.
            self.prm_model.enable_input_require_grads()
        self.prm_model.gradient_checkpointing_enable()
        self.prm_model.train()

        self.prime_ref_model = load_text_causal_lm(
            model_id,
            device=aux_device,
            attn_implementation=attn_implementation,
            logger=logger,
        )
        self.prime_ref_model.eval()
        for parameter in self.prime_ref_model.parameters():
            parameter.requires_grad_(False)

        hidden_size = self.prm_model.config.hidden_size
        self.prompt_calibration_head = None
        if self.use_prompt_calibration:
            self.prompt_calibration_head = PromptCalibrationHead(hidden_size).to(
                aux_device
            )

        trainable = [
            parameter
            for parameter in self.prm_model.parameters()
            if parameter.requires_grad
        ]
        if self.prompt_calibration_head is not None:
            trainable.extend(self.prompt_calibration_head.parameters())
        self.prm_optimizer = torch.optim.AdamW(
            trainable,
            lr=self.prime_prm_lr,
            weight_decay=float(prime_cfg.get("prm_weight_decay", 0.0)),
        )
        logger.info(
            "PRIME beta=%s baseline=%s PRM=%s chunk_tokens=%s actor=%s",
            self.prime_beta,
            self.prime_baseline,
            "LoRA" if prm_lora is not None else "full",
            self.logprob_chunk_tokens,
            "LoRA"
            if build_lora_config(peft_cfg, enabled_key="actor") is not None
            else "full",
        )
        logger.info("PRIME process reward coefficient=%s", self.prime_rm_coef)

    def _unwrap_causal_lm(self, model):
        try:
            unwrapped = self.accelerator.unwrap_model(model)
        except (AttributeError, ValueError):
            unwrapped = model
        if hasattr(unwrapped, "get_base_model"):
            unwrapped = unwrapped.get_base_model()
        return unwrapped

    def _last_completion_hidden(self, model, input_ids, attention_mask, logits_to_keep):
        causal_lm = self._unwrap_causal_lm(model)
        outputs = causal_lm.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        hidden = outputs.last_hidden_state[:, :-1, :]
        return hidden[:, -logits_to_keep:, :], causal_lm.lm_head

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        **unused_multimodal_inputs,
    ):
        """Override every TRL actor/ref log-prob path with token-chunk projection."""
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            hidden, lm_head = self._last_completion_hidden(
                model, ids, mask, logits_to_keep
            )
            targets = ids[:, -logits_to_keep:]
            logps = selected_logprobs_from_hidden(
                hidden,
                lm_head.weight,
                targets,
                bias=lm_head.bias,
                chunk_tokens=self.logprob_chunk_tokens,
                temperature=self.temperature,
                checkpoint_chunks=self.checkpoint_logprob_chunks
                and torch.is_grad_enabled(),
            )
            all_logps.append(logps)
            if compute_entropy:
                all_entropies.append(
                    entropy_from_hidden(
                        hidden.detach(),
                        lm_head.weight.detach(),
                        bias=lm_head.bias.detach()
                        if lm_head.bias is not None
                        else None,
                        chunk_tokens=self.logprob_chunk_tokens,
                        temperature=self.temperature,
                    )
                )
        return torch.cat(all_logps), torch.cat(
            all_entropies
        ) if compute_entropy else None

    def _token_logps(
        self, model, ids, attention_mask, logits_to_keep, *, batch_size, grad=False
    ):
        context = torch.enable_grad() if grad else torch.no_grad()
        chunks = []
        with context:
            for start in range(0, ids.size(0), batch_size):
                chunk_ids = ids[start : start + batch_size]
                chunk_mask = attention_mask[start : start + batch_size]
                hidden, lm_head = self._last_completion_hidden(
                    model, chunk_ids, chunk_mask, logits_to_keep
                )
                chunks.append(
                    selected_logprobs_from_hidden(
                        hidden,
                        lm_head.weight,
                        chunk_ids[:, -logits_to_keep:],
                        bias=lm_head.bias,
                        chunk_tokens=self.logprob_chunk_tokens,
                        checkpoint_chunks=grad and self.checkpoint_logprob_chunks,
                    )
                )
        return torch.cat(chunks)

    def _move_model(self, model, device):
        model.to(device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_optimizer(self, device):
        for state in self.prm_optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _policy_to_cpu(self):
        original_device = next(self.model.parameters()).device
        was_training = self.model.training
        if self.cpu_offload_policy:
            self.model.to("cpu")
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        state[key] = value.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        return original_device, was_training

    def _restore_policy(self, device, was_training):
        if self.cpu_offload_policy:
            self.model.to(device)
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        self.model.train(was_training)

    @torch.no_grad()
    def _process_rewards(self, ids, attention_mask, completion_mask):
        device = self.accelerator.device
        logits_to_keep = completion_mask.size(1)
        if self.cpu_offload_aux:
            self._move_model(self.prm_model, device)
        self.prm_model.eval()
        prm_logps = self._token_logps(
            self.prm_model,
            ids,
            attention_mask,
            logits_to_keep,
            batch_size=self.prm_ref_batch_size,
        )
        if self.cpu_offload_aux:
            self._move_model(self.prm_model, "cpu")
            self._move_model(self.prime_ref_model, device)
        ref_logps = self._token_logps(
            self.prime_ref_model,
            ids,
            attention_mask,
            logits_to_keep,
            batch_size=self.prm_ref_batch_size,
        )
        if self.cpu_offload_aux:
            self._move_model(self.prime_ref_model, "cpu")
        # Official dp_prime.py uses the raw ratio here; beta belongs to PRM BCE.
        return (
            implicit_process_rewards(prm_logps, ref_logps, completion_mask),
            ref_logps.detach(),
        )

    def _prompt_features(self, ids, attention_mask, completion_length):
        causal_lm = self._unwrap_causal_lm(self.prm_model)
        prompt_ids = ids[:, :-completion_length]
        prompt_mask = attention_mask[:, :-completion_length].to(torch.float32)
        with torch.no_grad():
            embeddings = causal_lm.get_input_embeddings()(prompt_ids)
            return (embeddings * prompt_mask.unsqueeze(-1)).sum(1) / prompt_mask.sum(
                1, keepdim=True
            ).clamp_min(1)

    def _update_prm(self, ids, attention_mask, completion_mask, outcomes, group_size):
        if ids.numel() == 0:
            return None
        device = self.accelerator.device
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)
        outcomes = outcomes.to(device)
        completion_length = completion_mask.size(1)

        if self.cpu_offload_aux:
            self._move_model(self.prime_ref_model, device)
        ref_logps = self._token_logps(
            self.prime_ref_model,
            ids,
            attention_mask,
            completion_length,
            batch_size=self.prm_ref_batch_size,
        ).detach()
        if self.cpu_offload_aux:
            self._move_model(self.prime_ref_model, "cpu")
            self._move_model(self.prm_model, device)
            self._move_optimizer(device)
            if self.prompt_calibration_head is not None:
                self.prompt_calibration_head.to(device)

        prompt_features = (
            self._prompt_features(ids, attention_mask, completion_length)
            if self.use_prompt_calibration
            else None
        )
        self.prm_model.train()
        last_loss = None
        last_ranking = None
        ranking_weight = (
            float(self.calibration_cfg.get("ranking_weight", 0.0))
            if self.use_prompt_calibration
            else 0.0
        )
        for _ in range(self.prime_prm_epochs):
            self.prm_optimizer.zero_grad(set_to_none=True)
            total = ids.size(0)
            for start in range(0, total, self.prm_grad_batch_size):
                end = min(start + self.prm_grad_batch_size, total)
                prm_logps = self._token_logps(
                    self.prm_model,
                    ids[start:end],
                    attention_mask[start:end],
                    completion_length,
                    batch_size=end - start,
                    grad=True,
                )
                intercept = (
                    self.prompt_calibration_head(prompt_features[start:end])
                    if self.prompt_calibration_head is not None
                    else None
                )
                bce, logits = prime_prm_bce_loss(
                    prm_logps,
                    ref_logps[start:end],
                    completion_mask[start:end],
                    outcomes[start:end],
                    beta=self.prime_beta,
                    prompt_intercept=intercept,
                )
                intercept_penalty = logits.new_zeros(())
                if intercept is not None:
                    intercept_penalty = (
                        float(self.calibration_cfg.get("intercept_l2", 0.0))
                        * intercept.square().mean()
                    )
                loss = (bce + intercept_penalty) * ((end - start) / total)
                loss.backward()
                last_loss = bce.detach()

            if ranking_weight > 0:
                pairs = []
                for group_start in range(0, total, group_size):
                    labels = outcomes[group_start : group_start + group_size]
                    positives = (
                        torch.nonzero(labels > 0.5, as_tuple=False).flatten()
                        + group_start
                    )
                    negatives = (
                        torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
                        + group_start
                    )
                    pairs.extend(
                        (int(pos), int(neg)) for pos in positives for neg in negatives
                    )
                ranking_values = []
                for positive, negative in pairs:
                    pair_scores = []
                    for index in (positive, negative):
                        prm_logps = self._token_logps(
                            self.prm_model,
                            ids[index : index + 1],
                            attention_mask[index : index + 1],
                            completion_length,
                            batch_size=1,
                            grad=True,
                        )
                        pair_scores.append(
                            self.prime_beta
                            * (
                                (prm_logps - ref_logps[index : index + 1])
                                * completion_mask[index : index + 1]
                            ).sum()
                        )
                    pair_loss = F.softplus(-(pair_scores[0] - pair_scores[1]))
                    (ranking_weight * pair_loss / max(len(pairs), 1)).backward()
                    ranking_values.append(pair_loss.detach())
                if ranking_values:
                    last_ranking = torch.stack(ranking_values).mean()

            parameters = [
                parameter
                for parameter in self.prm_model.parameters()
                if parameter.requires_grad
            ]
            if self.prompt_calibration_head is not None:
                parameters.extend(self.prompt_calibration_head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, self.prime_prm_grad_clip)
            self.prm_optimizer.step()

        if self.cpu_offload_aux:
            self._move_optimizer("cpu")
            self._move_model(self.prm_model, "cpu")
            if self.prompt_calibration_head is not None:
                self.prompt_calibration_head.to("cpu")
        if last_loss is None:
            return None
        return {
            "bce": float(last_loss),
            "ranking": float(last_ranking) if last_ranking is not None else None,
        }

    @staticmethod
    def _pairwise_reliability(sequence_scores, outcomes, group_size):
        correct = 0.0
        pairs = 0.0
        for scores, labels in zip(
            sequence_scores.reshape(-1, group_size), outcomes.reshape(-1, group_size)
        ):
            for positive in scores[labels > 0.5]:
                for negative in scores[labels <= 0.5]:
                    correct += float(positive > negative) + 0.5 * float(
                        positive == negative
                    )
                    pairs += 1.0
        return correct / pairs if pairs else 0.5

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_root = (
            Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        )
        self._save_prm_artifacts(checkpoint_root)

    def _save_prm_artifacts(self, checkpoint_root):
        """Persist the complete auxiliary PRIME state below ``root/prm``."""
        prm_dir = Path(checkpoint_root) / "prm"
        prm_dir.mkdir(parents=True, exist_ok=True)
        from peft import PeftModel, get_peft_model_state_dict

        is_peft = isinstance(self.prm_model, PeftModel)
        model_state = (
            get_peft_model_state_dict(self.prm_model)
            if is_peft
            else self.prm_model.state_dict()
        )
        torch.save(model_state, prm_dir / "model.pt")
        torch.save(self.prm_optimizer.state_dict(), prm_dir / "optimizer.pt")
        state = {"prime_step": self._prime_step, "prm_is_peft": is_peft}
        if self.prompt_calibration_head is not None:
            state["prompt_calibration_head"] = self.prompt_calibration_head.state_dict()
        torch.save(state, prm_dir / "state.pt")

    def _load_prm_checkpoint(self, checkpoint_path):
        prm_dir = Path(checkpoint_path) / "prm"
        if not prm_dir.exists():
            logger.warning("No PRM checkpoint under %s", checkpoint_path)
            return
        state = torch.load(prm_dir / "state.pt", map_location="cpu", weights_only=True)
        model_state = torch.load(
            prm_dir / "model.pt", map_location="cpu", weights_only=True
        )
        if state.get("prm_is_peft", False):
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.prm_model, model_state)
        else:
            self.prm_model.load_state_dict(model_state)
        self.prm_optimizer.load_state_dict(
            torch.load(prm_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        self._prime_step = int(state.get("prime_step", 0))
        if (
            self.prompt_calibration_head is not None
            and "prompt_calibration_head" in state
        ):
            self.prompt_calibration_head.load_state_dict(
                state["prompt_calibration_head"]
            )

    def _score_candidate_inputs(self, inputs, device):
        output = super()._generate_and_score_completions(inputs)
        completion_text = self.processing_class.batch_decode(
            output["completion_ids"], skip_special_tokens=True
        )
        reward_inputs = [
            [{"role": "assistant", "content": text}] for text in completion_text
        ]
        outcomes = torch.tensor(
            accuracy_reward(reward_inputs, [item["solution"] for item in inputs]),
            dtype=torch.float32,
            device=device,
        )
        return output, outcomes

    @staticmethod
    def _refill_example_key(example):
        """Stable identity for excluding prompts already generated this step."""
        return json.dumps(
            [example.get("prompt"), example.get("solution")],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _prepare_refill_stream(self, inputs, group_size):
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Accuracy-filter refill is currently verified only for single-GPU training"
            )
        # Some small unit harnesses replace _sample_refill_inputs directly.
        if not hasattr(self, "train_dataset"):
            return
        excluded = {
            self._refill_example_key(inputs[index])
            for index in range(0, len(inputs), group_size)
        }
        generator = torch.Generator().manual_seed(
            int(self.args.seed) + self._prime_step * 1009
        )
        order = torch.randperm(len(self.train_dataset), generator=generator).tolist()
        self._refill_indices = [
            index
            for index in order
            if self._refill_example_key(self.train_dataset[index]) not in excluded
        ]
        self._refill_cursor = 0

    def _sample_refill_inputs(self, num_groups, group_size, refill_round):
        del refill_round  # The stream order is fixed once per optimizer step.
        remaining = len(self._refill_indices) - self._refill_cursor
        if remaining <= 0:
            raise RuntimeError(
                "Official-style refill dataset exhausted before a complete filtered batch"
            )
        take = min(num_groups, remaining)
        indices = self._refill_indices[
            self._refill_cursor : self._refill_cursor + take
        ]
        self._refill_cursor += take
        rows = []
        for index in indices:
            example = self.train_dataset[index]
            rows.extend([dict(example) for _ in range(group_size)])
        return rows

    def _select_rows(self, output, row_mask):
        selected = {}
        row_count = row_mask.numel()
        for key, value in output.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.size(0) == row_count
            ):
                selected[key] = value[row_mask]
            else:
                selected[key] = value
        return selected

    def _concat_refill_outputs(self, parts, target_rows):
        result = {}
        keys = parts[0].keys()
        row_keys = {
            key
            for key in keys
            if isinstance(parts[0][key], torch.Tensor) and parts[0][key].ndim > 0
        }
        for key in keys:
            values = [part[key] for part in parts]
            if key not in row_keys:
                result[key] = values[0]
                continue
            if values[0].ndim == 1:
                result[key] = torch.cat(values, dim=0)[:target_rows]
                continue
            max_length = max(value.size(1) for value in values)
            padded = []
            for value in values:
                missing = max_length - value.size(1)
                if missing == 0:
                    padded.append(value)
                    continue
                if key == "prompt_ids":
                    padded.append(F.pad(value, (missing, 0), value=self.pad_token_id))
                elif key == "prompt_mask":
                    padded.append(F.pad(value, (missing, 0), value=0))
                else:
                    padded.append(F.pad(value, (0, missing), value=0))
            result[key] = torch.cat(padded, dim=0)[:target_rows]
        if "num_items_in_batch" in result:
            result["num_items_in_batch"] = result["completion_mask"].sum()
        return result

    def _valid_group_mask(self, outcomes, output, group_size):
        keep = (
            solvable_group_mask(
                outcomes, group_size, self.filter_lower, self.filter_upper
            )
            if self.filter_accuracy
            else torch.ones(
                outcomes.numel() // group_size, dtype=torch.bool, device=outcomes.device
            )
        )
        if self.filter_truncated_groups:
            keep = keep & non_truncated_group_mask(
                output["completion_mask"], group_size, self.max_completion_length
            ).to(keep.device)
        return keep

    def _generate_with_accuracy_refill(self, inputs, device, group_size):
        target_groups = len(inputs) // group_size
        self._prepare_refill_stream(inputs, group_size)
        pending_inputs = inputs
        valid_outputs, valid_outcomes = [], []
        valid_groups = 0
        generated_groups = 0
        rounds = 0
        while valid_groups < target_groups:
            candidate_output, candidate_outcomes = self._score_candidate_inputs(
                pending_inputs, device
            )
            group_mask = self._valid_group_mask(
                candidate_outcomes, candidate_output, group_size
            )
            generated_groups += int(group_mask.numel())
            row_mask = group_mask.repeat_interleave(group_size)
            if row_mask.any():
                valid_outputs.append(self._select_rows(candidate_output, row_mask))
                valid_outcomes.append(candidate_outcomes[row_mask])
                valid_groups += int(group_mask.sum())
            rounds += 1
            if valid_groups >= target_groups:
                break
            missing = target_groups - valid_groups
            pending_inputs = self._sample_refill_inputs(missing, group_size, rounds)
        target_rows = target_groups * group_size
        self._metrics["train"]["prime/refill_rounds"].append(rounds - 1)
        self._metrics["train"]["prime/refill_generated_groups"].append(
            generated_groups
        )
        self._metrics["train"]["prime/refill_acceptance"].append(
            target_groups / generated_groups
        )
        return self._concat_refill_outputs(valid_outputs, target_rows), torch.cat(
            valid_outcomes
        )[:target_rows]

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        group_size = (
            self.num_generations if mode == "train" else self.num_generations_eval
        )
        if mode == "train" and self.filter_accuracy and self.filter_refill:
            output, outcomes = self._generate_with_accuracy_refill(
                inputs, device, group_size
            )
        else:
            output, outcomes = self._score_candidate_inputs(inputs, device)

        ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        attention_mask = torch.cat(
            [output["prompt_mask"], output["completion_mask"]], dim=1
        )
        completion_mask = output["completion_mask"]

        policy_device, policy_was_training = self._policy_to_cpu()
        process_rewards, _ = self._process_rewards(ids, attention_mask, completion_mask)

        keep_groups = self._valid_group_mask(outcomes, output, group_size)
        keep_rows = keep_groups.repeat_interleave(group_size)
        prm_rows = keep_rows.clone()

        sequence_scores = (process_rewards * completion_mask).sum(dim=1)
        reliability = (
            self._pairwise_reliability(
                sequence_scores[keep_rows], outcomes[keep_rows], group_size
            )
            if keep_rows.any()
            else 0.5
        )

        guided_weights = torch.ones_like(outcomes)
        if self.use_guided_search:
            selection = select_prm_guided_candidates(
                sequence_scores,
                group_size=group_size,
                keep=int(self.guided_cfg.get("keep", group_size)),
                mode=self.guided_cfg.get("mode", "topk"),
                temperature=float(self.guided_cfg.get("temperature", 1.0)),
                allow_biased_update=bool(
                    self.guided_cfg.get("allow_biased_update", False)
                ),
            )
            keep_rows = keep_rows & selection.selected_mask.to(device)
            guided_weights = selection.sample_weights.to(device)
            self._metrics[mode]["prime/guided_off_policy"].append(
                float(selection.is_off_policy)
            )

        alpha = process_reward_weight(
            step=self._prime_step,
            schedule=self.process_schedule,
            warmup_steps=self.process_warmup_steps,
            reliability=reliability,
            reliability_floor=self.reliability_floor,
            reliability_full=self.reliability_full,
        ) * self.prime_rm_coef
        result = compute_prime_advantages(
            outcomes,
            process_rewards,
            completion_mask,
            group_size=group_size,
            filter_lower=self.filter_lower if self.filter_accuracy else 0.0,
            filter_upper=self.filter_upper if self.filter_accuracy else 1.0,
            baseline=self.prime_baseline,
            dpo_z_beta=float(self.prime_cfg.get("dpo_z_beta", 0.1)),
            dpo_z_leave_one_out=bool(self.prime_cfg.get("dpo_z_leave_one_out", True)),
            gamma=self.prime_gamma,
            process_weight=alpha,
        )
        advantages = result.advantages * guided_weights.unsqueeze(1)
        advantages[~keep_rows] = 0
        output["completion_mask"] = output["completion_mask"] * keep_rows.unsqueeze(1)
        output["advantages"] = advantages
        output["num_items_in_batch"] = output["completion_mask"].sum()

        prm_loss = None
        if mode == "train" and prm_rows.any():
            prm_loss = self._update_prm(
                ids[prm_rows],
                attention_mask[prm_rows],
                completion_mask[prm_rows],
                outcomes[prm_rows],
                group_size,
            )
        self._restore_policy(policy_device, policy_was_training)

        filtered_fraction = 1.0 - keep_groups.float().mean().item()
        self._metrics[mode]["prime/filtered_prompts_frac"].append(filtered_fraction)
        self._metrics[mode]["prime/process_weight"].append(alpha)
        self._metrics[mode]["prime/prm_reliability"].append(reliability)
        self._metrics[mode]["prime/process_norm"].append(
            float(result.process_normalization_factor)
        )
        if prm_loss is not None:
            self._metrics[mode]["prime/prm_loss"].append(prm_loss["bce"])
            if prm_loss["ranking"] is not None:
                self._metrics[mode]["prime/prm_ranking_loss"].append(
                    prm_loss["ranking"]
                )
        self._prime_step += int(mode == "train")
        return output


def train(config: dict, data_dir: str | None = None, output_dir: str | None = None):
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})
    train_cfg = config.get("training", {})
    validation_cfg = config.get("validation", {})
    prime_cfg = config.get("prime", {})
    memory_cfg = config.get("memory", {})
    peft_cfg = config.get("peft", {})
    model_name = model_cfg.get("name", "Qwen/Qwen3.5-0.8B")
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    run_name = config.get("run_name", "prime_qwen35_08b")
    output_dir = output_dir or f"./outputs/{run_name}"

    if data_dir is None:
        raise ValueError("--data-dir is required")
    dataset = load_from_disk(data_dir)
    validate_preformatted_math_rl_dataset(dataset)
    if grpo_cfg.get("num_samples") and grpo_cfg["num_samples"] < len(dataset):
        dataset = dataset.select(range(grpo_cfg["num_samples"]))

    eval_dataset = None
    eval_data_dir = validation_cfg.get("data_dir")
    if eval_data_dir:
        eval_dataset = load_from_disk(eval_data_dir)
        validate_preformatted_math_rl_dataset(eval_dataset)
        eval_samples = int(validation_cfg.get("num_samples", len(eval_dataset)))
        eval_dataset = eval_dataset.select(
            range(min(eval_samples, len(eval_dataset)))
        )
        logger.info("Loaded validation dataset: %d examples", len(eval_dataset))

    patch_vllm_language_model_only(
        bool(model_cfg.get("language_model_only", True)), logger=logger
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_prompt_tokens = validate_prompt_token_lengths(
        dataset,
        tokenizer,
        int(grpo_cfg.get("max_prompt_length", 1024)),
    )
    logger.info("Prompt token audit: max=%d", max_prompt_tokens)
    if eval_dataset is not None:
        max_eval_prompt_tokens = validate_prompt_token_lengths(
            eval_dataset,
            tokenizer,
            int(grpo_cfg.get("max_prompt_length", 1024)),
        )
        logger.info(
            "Validation: n=%d max_prompt_tokens=%d eval_steps=%d "
            "num_generations=1 sampler=(temperature=%s, top_p=%s, top_k=%s)",
            len(eval_dataset),
            max_eval_prompt_tokens,
            int(validation_cfg.get("eval_steps", 20)),
            grpo_cfg.get("temperature", 1.0),
            grpo_cfg.get("top_p", 1.0),
            grpo_cfg.get("top_k", 0),
        )
    actor_model = load_text_causal_lm(
        model_name,
        device="cpu",
        attn_implementation=attn_implementation,
        logger=logger,
    )

    args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        loss_type=grpo_cfg.get("loss_type", "grpo"),
        scale_rewards="none",
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high"),
        beta=grpo_cfg.get("beta", 0.0),
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 3072),
        temperature=grpo_cfg.get("temperature", 1.0),
        top_p=grpo_cfg.get("top_p", 1.0),
        top_k=grpo_cfg.get("top_k", 0),
        mask_truncated_completions=grpo_cfg.get("mask_truncated_completions", False),
        generation_batch_size=grpo_cfg.get("generation_batch_size"),
        num_iterations=train_cfg.get("num_iterations", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 64),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "constant"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=train_cfg.get("seed", 42),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(validation_cfg.get("eval_steps", 20)),
        eval_on_start=bool(validation_cfg.get("eval_on_start", False)),
        per_device_eval_batch_size=int(
            validation_cfg.get("per_device_eval_batch_size", 8)
        ),
        num_generations_eval=int(validation_cfg.get("num_generations", 1)),
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="tensorboard",
        log_completions=grpo_cfg.get("log_completions", True),
        num_completions_to_print=grpo_cfg.get("num_completions_to_print", 8),
        optim=train_cfg.get("optim", "adamw_torch"),
        use_vllm=grpo_cfg.get("use_vllm", True),
        vllm_mode=grpo_cfg.get("vllm_mode", "colocate"),
        vllm_enable_sleep_mode=grpo_cfg.get("vllm_enable_sleep_mode", True),
        vllm_importance_sampling_correction=grpo_cfg.get(
            "vllm_importance_sampling_correction", False
        ),
        vllm_gpu_memory_utilization=grpo_cfg.get("vllm_gpu_memory_utilization", 0.30),
        vllm_max_model_length=grpo_cfg.get(
            "max_model_len",
            grpo_cfg.get("max_prompt_length", 1024)
            + grpo_cfg.get("max_completion_length", 3072),
        ),
        use_liger_kernel=False,
        reward_weights=[1.0],
    )
    actor_lora = build_lora_config(peft_cfg, enabled_key="actor")
    # GRPOTrainer creates the PEFT actor before Trainer.__init__ seeds RNG.
    # Seed explicitly so PRIME variants share the same initial LoRA subspace.
    set_seed(int(args.seed))

    trainer = PrimeGRPOTrainer(
        prime_cfg=prime_cfg,
        memory_cfg=memory_cfg,
        peft_cfg=peft_cfg,
        model_id=model_name,
        attn_implementation=attn_implementation,
        model=actor_model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[accuracy_reward],
        peft_config=actor_lora,
    )
    install_vllm_sync_canary(
        trainer,
        output_dir=output_dir,
        logger=logger,
        enabled=bool(grpo_cfg.get("verify_vllm_weight_sync", True)),
        require_change_after_step=bool(
            grpo_cfg.get("require_vllm_weight_change_after_step", True)
        ),
    )

    resume = train_cfg.get("resume_from_checkpoint")
    if resume is True:
        checkpoints = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]),
        )
        resume = checkpoints[-1] if checkpoints else None
    if resume:
        trainer._load_prm_checkpoint(resume)
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    trainer._save_prm_artifacts(output_dir)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    logger.info("Training complete: %s; GPU: %s", result.metrics, get_gpu_memory_info())
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Author-aligned memory-safe PRIME")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    cli = parser.parse_args()
    root = Path(__file__).parent.parent
    config = load_config(
        cli.config, base_config_path=str(root / "configs" / "grpo_base.yaml")
    )
    train(config, cli.data_dir, cli.output_dir)


if __name__ == "__main__":
    main()
