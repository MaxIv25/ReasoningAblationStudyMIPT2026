#!/usr/bin/env python3
"""Measure post-trained policy difficulty on held-out GSM8K/MATH train prompts.

This is a diagnostic probe, not policy-specific prompt calibration. It samples
fixed, auditable strata and reports which *strata* provide useful outcome
variation for GRPO and PRIME. Probe example IDs are persisted so they can be
excluded from the later RL training sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MATH_LEVELS = (1, 2, 3, 4, 5)
DEFAULT_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def select_stratified_probe(
    records: Sequence[dict],
    *,
    gsm8k_count: int,
    math_levels: Sequence[int],
    math_subjects: Sequence[str],
    math_per_subject: int,
    seed: int,
) -> list[dict]:
    """Select a deterministic sample independently within every stratum."""
    rng = random.Random(seed)
    by_stratum: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record["source"] == "gsm8k":
            by_stratum[("gsm8k",)].append(record)
        elif record["source"] == "math":
            by_stratum[("math", record["level"], record["subject"])].append(record)

    selections: list[dict] = []

    def draw(key: tuple, count: int) -> None:
        candidates = sorted(by_stratum[key], key=lambda item: item["example_id"])
        if len(candidates) < count:
            raise ValueError(
                f"Stratum {key!r} has {len(candidates)} eligible examples; "
                f"requested {count}."
            )
        selections.extend(rng.sample(candidates, count))

    draw(("gsm8k",), gsm8k_count)
    for level in math_levels:
        for subject in math_subjects:
            draw(("math", level, subject), math_per_subject)

    return selections


def _build_prompt(tokenizer, problem: str, *, enable_thinking: bool) -> str:
    content = (
        f"{problem}\n\nPlease reason step by step, and put your final answer "
        "within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _load_pool(
    tokenizer, max_prompt_length: int, *, enable_thinking: bool
) -> tuple[list[dict], Counter]:
    from datasets import load_dataset

    from src.utils import extract_boxed_answer

    pool: list[dict] = []
    rejected: Counter = Counter()

    def add(record: dict, problem: str) -> None:
        prompt = _build_prompt(
            tokenizer, problem, enable_thinking=enable_thinking
        )
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if prompt_tokens > max_prompt_length:
            rejected[record["source"]] += 1
            return
        pool.append({**record, "problem": problem, "prompt": prompt, "prompt_tokens": prompt_tokens})

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    for index, example in enumerate(gsm8k):
        match = re.search(r"####\s*(.+)$", example["answer"], re.MULTILINE)
        if match:
            add(
                {
                    "example_id": f"gsm8k:train:{index}",
                    "source": "gsm8k",
                    "solution": match.group(1).strip(),
                },
                example["question"],
            )

    for subject in DEFAULT_MATH_SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        for index, example in enumerate(dataset):
            solution = extract_boxed_answer(example.get("solution", ""))
            level_match = re.search(r"(\d+)", str(example.get("level", "")))
            if solution is None or level_match is None:
                rejected["math_unparseable"] += 1
                continue
            add(
                {
                    "example_id": f"math:{subject}:train:{index}",
                    "source": "math",
                    "level": int(level_match.group(1)),
                    "subject": subject,
                    "solution": solution,
                },
                example["problem"],
            )
    return pool, rejected


def _bootstrap_accuracy_ci(records: Sequence[dict], seed: int) -> list[float]:
    if not records:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    prompt_accuracies = [record["num_correct"] / record["num_generations"] for record in records]
    estimates = sorted(
        mean(rng.choices(prompt_accuracies, k=len(prompt_accuracies)))
        for _ in range(2000)
    )
    return [estimates[49], estimates[1949]]


def _request_seed(base_seed: int, example_id: str) -> int:
    """Give every prompt an order-independent reproducible sampling stream."""
    digest = hashlib.sha256(example_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def _summarize_group(records: Sequence[dict], num_generations: int, seed: int) -> dict:
    correct = [record["num_correct"] for record in records]
    lengths = sorted(
        length for record in records for length in record["completion_token_lengths"]
    )
    finish_reasons = [
        reason for record in records for reason in record["finish_reasons"]
    ]

    def percentile(values: Sequence[int], q: float) -> float:
        if not values:
            return float("nan")
        return float(values[round((len(values) - 1) * q)])

    return {
        "num_prompts": len(records),
        "num_completions": len(records) * num_generations,
        "accuracy": sum(correct) / max(1, len(records) * num_generations),
        "accuracy_prompt_bootstrap_95ci": _bootstrap_accuracy_ci(records, seed),
        "count_correct_histogram": {
            str(count): correct.count(count) for count in range(num_generations + 1)
        },
        "grpo_informative_rate": sum(0 < count < num_generations for count in correct)
        / max(1, len(records)),
        "prime_accepted_rate": sum(2 <= count <= num_generations - 2 for count in correct)
        / max(1, len(records)),
        "all_wrong_rate": correct.count(0) / max(1, len(records)),
        "all_correct_rate": correct.count(num_generations) / max(1, len(records)),
        "truncated_completion_rate": finish_reasons.count("length")
        / max(1, len(finish_reasons)),
        "completion_tokens": {
            "mean": mean(lengths) if lengths else float("nan"),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "prompt_tokens": {
            "mean": mean(record["prompt_tokens"] for record in records) if records else float("nan"),
            "max": max((record["prompt_tokens"] for record in records), default=0),
        },
    }


def summarize(records: Sequence[dict], num_generations: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = {"overall": list(records)}
    groups["gsm8k"] = [record for record in records if record["source"] == "gsm8k"]
    for level in DEFAULT_MATH_LEVELS:
        groups[f"math_L{level}"] = [
            record
            for record in records
            if record["source"] == "math" and record["level"] == level
        ]
    for subject in DEFAULT_MATH_SUBJECTS:
        groups[f"math_subject:{subject}"] = [
            record
            for record in records
            if record["source"] == "math" and record["subject"] == subject
        ]
    return {
        name: _summarize_group(group, num_generations, seed + index)
        for index, (name, group) in enumerate(groups.items())
        if group
    }


def _git_metadata() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gsm8k-count", type=int, default=100)
    parser.add_argument("--math-levels", type=int, nargs="+", default=list(DEFAULT_MATH_LEVELS))
    parser.add_argument("--math-subjects", nargs="+", default=list(DEFAULT_MATH_SUBJECTS))
    parser.add_argument("--math-per-subject", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=16384)
    parser.add_argument("--request-batch-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--kv-cache-memory-gib", type=float, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if args.num_generations < 4:
        raise ValueError("At least four generations are required for PRIME acceptance metrics")

    from transformers import AutoConfig, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "probe_manifest.jsonl"
    results_path = args.output / "raw_results.jsonl"
    summary_path = args.output / "summary.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"{manifest_path} exists; use --resume or a fresh output path")

    previous_metadata = {}
    if summary_path.exists() and args.resume:
        previous_metadata = json.loads(summary_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model_config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if manifest_path.exists():
        selected = _read_completed(manifest_path)
        rejected = Counter(
            previous_metadata.get("selection", {}).get(
                "rejected_before_sampling", {}
            )
        )
        if not rejected:
            _, rejected = _load_pool(
                tokenizer, args.max_prompt_length,
                enable_thinking=args.enable_thinking,
            )
    else:
        pool, rejected = _load_pool(
            tokenizer, args.max_prompt_length,
            enable_thinking=args.enable_thinking,
        )
        selected = select_stratified_probe(
            pool,
            gsm8k_count=args.gsm8k_count,
            math_levels=args.math_levels,
            math_subjects=args.math_subjects,
            math_per_subject=args.math_per_subject,
            seed=args.seed,
        )
        _write_jsonl(manifest_path, selected)

    metadata = {
        "created_at": previous_metadata.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        ),
        "last_resumed_at": datetime.now(timezone.utc).isoformat() if args.resume else None,
        "status": "prepared" if args.prepare_only else "running",
        "model": args.model,
        "model_revision_requested": args.revision,
        "model_commit": getattr(model_config, "_commit_hash", None),
        "git": _git_metadata(),
        "code_sha256": {
            "probe_script": _file_sha256(Path(__file__)),
            "verifier": _file_sha256(PROJECT_ROOT / "src" / "utils.py"),
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": args.enable_thinking,
            "num_generations": args.num_generations,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
        },
        "selection": {
            "seed": args.seed,
            "gsm8k_count": args.gsm8k_count,
            "math_levels": args.math_levels,
            "math_subjects": args.math_subjects,
            "math_per_subject": args.math_per_subject,
            "selected_prompts": len(selected),
            "rejected_before_sampling": dict(rejected),
        },
        "runtime": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "request_batch_size": args.request_batch_size,
            "kv_cache_memory_gib": args.kv_cache_memory_gib,
        },
    }
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    if args.prepare_only:
        print(f"Prepared {len(selected)} prompts at {manifest_path}")
        return

    completed = _read_completed(results_path) if args.resume else []
    completed_ids = {record["example_id"] for record in completed}
    pending = [record for record in selected if record["example_id"] not in completed_ids]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=(
            int(args.kv_cache_memory_gib * 1024**3)
            if args.kv_cache_memory_gib is not None
            else None
        ),
        seed=args.seed,
    )
    mode = "at" if completed else "wt"
    from src.utils import extract_boxed_answer, verify_answer

    with results_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.request_batch_size):
            batch = pending[start : start + args.request_batch_size]
            sampling = [
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    max_tokens=args.max_completion_length,
                    n=args.num_generations,
                    seed=_request_seed(args.seed, record["example_id"]),
                )
                for record in batch
            ]
            outputs = llm.generate(
                [record["prompt"] for record in batch], sampling
            )
            for record, output in zip(batch, outputs, strict=True):
                completions = [candidate.text for candidate in output.outputs]
                predictions = [extract_boxed_answer(text) for text in completions]
                correctness = [
                    verify_answer(prediction, record["solution"])
                    for prediction in predictions
                ]
                result = {
                    **record,
                    "completions": completions,
                    "predictions": predictions,
                    "correctness": correctness,
                    "num_correct": sum(correctness),
                    "num_generations": len(completions),
                    "completion_token_lengths": [
                        len(candidate.token_ids) for candidate in output.outputs
                    ],
                    "finish_reasons": [candidate.finish_reason for candidate in output.outputs],
                }
                completed.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"Completed {len(completed)}/{len(selected)} prompts", flush=True)

    metadata["status"] = "done"
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["metrics"] = summarize(completed, args.num_generations, args.seed)
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
