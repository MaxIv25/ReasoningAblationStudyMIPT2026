#!/usr/bin/env python3
"""Create a fixed train-disjoint validation set for all RL methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.dataset_contract import validate_preformatted_math_rl_dataset
from src.rl.validation_data import build_unseen_validation_dataset
from src.utils import verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_from_disk(str(args.candidates))
    train_dataset = load_from_disk(str(args.train_data))
    validation = build_unseen_validation_dataset(
        candidates,
        train_dataset,
        size=args.size,
        seed=args.seed,
    )
    validate_preformatted_math_rl_dataset(validation)
    rejected_labels = [
        index
        for index, solution in enumerate(validation["solution"])
        if not verify_answer(solution, solution)
    ]
    if rejected_labels:
        raise RuntimeError(
            f"Verifier rejects {len(rejected_labels)} validation labels: "
            f"{rejected_labels[:10]}"
        )

    validation.save_to_disk(str(args.output))
    prompt_digest = hashlib.sha256(
        "\n".join(validation["prompt"]).encode("utf-8")
    ).hexdigest()
    manifest = {
        "candidates": str(args.candidates),
        "train_data": str(args.train_data),
        "size": len(validation),
        "seed": args.seed,
        "prompt_sha256": prompt_digest,
        "verifier_self_checks_passed": len(validation),
    }
    (args.output / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
