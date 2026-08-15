"""TRL GRPO extension that replaces the group-mean baseline with DPO-Z."""

from trl import GRPOTrainer

from src.rl.prime_core import compute_group_advantages
from src.rl.memory_bounded_grpo import MemoryBoundedGRPOTrainer


class DPOZGRPOTrainer(GRPOTrainer):
    """Use ``R - beta log E exp(R/beta)`` as the ordinary GRPO advantage."""

    def __init__(
        self, *args, dpo_z_beta: float = 0.1, dpo_z_leave_one_out: bool = True, **kwargs
    ):
        self.dpo_z_beta = dpo_z_beta
        self.dpo_z_leave_one_out = dpo_z_leave_one_out
        self._dpo_z_rewards_per_func = None
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._dpo_z_rewards_per_func = rewards_per_func.detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not self.model.training:
            self._dpo_z_rewards_per_func = None
            return output
        if self._dpo_z_rewards_per_func is None:
            raise RuntimeError("TRL did not expose rewards through _calculate_rewards")

        mode = "train" if self.model.training else "eval"
        group_size = (
            self.num_generations if mode == "train" else self.num_generations_eval
        )
        rewards = (
            self._dpo_z_rewards_per_func
            * self.reward_weights.to(self._dpo_z_rewards_per_func.device).unsqueeze(0)
        ).nansum(dim=1)
        advantages = compute_group_advantages(
            rewards,
            group_size,
            baseline="dpo_z",
            dpo_z_beta=self.dpo_z_beta,
            dpo_z_leave_one_out=self.dpo_z_leave_one_out,
        )

        if self.scale_rewards == "group":
            scale = (
                rewards.reshape(-1, group_size).std(dim=1).repeat_interleave(group_size)
            )
            advantages = advantages / (scale + 1e-4)
        elif self.scale_rewards == "batch":
            scale = rewards.std() if rewards.numel() > 1 else rewards.new_tensor(0.0)
            advantages = advantages / (scale + 1e-4)
        elif self.scale_rewards != "none":
            raise ValueError(
                f"Unsupported scale_rewards={self.scale_rewards!r} for DPO-Z"
            )

        local_size = output["advantages"].size(0)
        process_slice = slice(
            self.accelerator.process_index * local_size,
            (self.accelerator.process_index + 1) * local_size,
        )
        output["advantages"] = advantages[process_slice]


        recent = self._logs["advantages"]
        for _ in range(min(len(advantages), len(recent))):
            recent.pop()
        recent.extend(advantages.tolist())
        self._metrics[mode]["dpo_z/beta"].append(float(self.dpo_z_beta))
        self._metrics[mode]["dpo_z/advantage_mean"].append(advantages.mean().item())
        self._dpo_z_rewards_per_func = None
        return output


class MemoryBoundedDPOZGRPOTrainer(DPOZGRPOTrainer, MemoryBoundedGRPOTrainer):
    """Compose DPO-Z advantages with exact token-chunked policy projection.

    DPO-Z replaces sequence advantages after rollout scoring, while the
    memory-bounded trainer consumes those advantages unchanged and bounds only
    the vocabulary projection. The inheritance order is intentional: DPO-Z
    generation hooks run first and its cooperative ``super().__init__`` reaches
    memory-safety validation before TRL initialization.
    """

    pass
