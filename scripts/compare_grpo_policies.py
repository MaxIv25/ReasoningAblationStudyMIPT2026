#!/usr/bin/env python3
"""Compare two policies on the same fixed token positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def distribution_metrics(
    vanilla_logits: torch.Tensor,
    dpoz_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return distribution-level differences without sampling noise."""
    vanilla_logp = vanilla_logits.float().log_softmax(dim=-1)
    dpoz_logp = dpoz_logits.float().log_softmax(dim=-1)
    vanilla_prob = vanilla_logp.exp()
    dpoz_prob = dpoz_logp.exp()

    kl = (vanilla_prob * (vanilla_logp - dpoz_logp)).sum(dim=-1)
    total_variation = 0.5 * (vanilla_prob - dpoz_prob).abs().sum(dim=-1)
    top1_disagreement = vanilla_logits.argmax(dim=-1).ne(
        dpoz_logits.argmax(dim=-1)
    )
    target_index = targets.to(device=vanilla_logits.device, dtype=torch.long).unsqueeze(-1)
    vanilla_target = vanilla_logp.gather(-1, target_index).squeeze(-1)
    dpoz_target = dpoz_logp.gather(-1, target_index).squeeze(-1)

    return {
        "mean_kl_vanilla_dpoz": float(kl.mean().item()),
        "mean_total_variation": float(total_variation.mean().item()),
        "top1_disagreement_fraction": float(top1_disagreement.float().mean().item()),
        "mean_abs_target_logprob_difference": float(
            (vanilla_target - dpoz_target).abs().mean().item()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("vanilla_logits", type=Path)
    parser.add_argument("dpoz_logits", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vanilla = torch.load(args.vanilla_logits, map_location="cpu", weights_only=True)
    dpoz = torch.load(args.dpoz_logits, map_location="cpu", weights_only=True)
    targets = torch.load(args.targets, map_location="cpu", weights_only=True)
    metrics = distribution_metrics(vanilla, dpoz, targets)
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
