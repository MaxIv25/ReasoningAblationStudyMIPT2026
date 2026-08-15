#!/usr/bin/env python3
"""Parse TRL text logs and build an auditable partial-run comparison bundle."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METHODS = {
    "vanilla": "Vanilla GRPO",
    "prime": "PRIME (failed at step 9)",
    "dpo_z": "GRPO + DPO-Z",
}
COLORS = {"vanilla": "#0072B2", "prime": "#D55E00", "dpo_z": "#009E73"}
STYLES = {"vanilla": "-", "prime": "--", "dpo_z": "-."}


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_log(path: Path, method: str, total_steps: int) -> list[dict]:
    rows = []
    text = ANSI_RE.sub("", path.read_text(errors="replace")).replace("\r", "\n")
    for line in text.splitlines():
        start = line.find("{'loss':")
        if start < 0:
            continue
        end = line.rfind("}")
        if end <= start:
            continue
        try:
            payload = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        epoch = _number(payload.get("epoch"))
        if epoch is None:
            continue
        row = {key: _number(value) for key, value in payload.items()}
        row["method"] = method
        row["step"] = int(round(epoch * total_steps))
        rows.append(row)
    deduplicated = {}
    for row in rows:
        deduplicated[row["step"]] = row
    return [deduplicated[step] for step in sorted(deduplicated)]


def _series(rows, key):
    points = [(row["step"], row.get(key)) for row in rows if row.get(key) is not None]
    if not points:
        return np.array([]), np.array([])
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points])


def _plot(ax, all_rows, key, *, scale=1.0):
    for method, rows in all_rows.items():
        x, y = _series(rows, key)
        if not len(x):
            continue
        ax.plot(
            x,
            y * scale,
            label=METHODS[method],
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_xlabel("Optimizer step")


def _mark_prime_failure(ax):
    ax.axvline(9, color=COLORS["prime"], linestyle=":", alpha=0.65, linewidth=1.2)


def _save(fig, output_dir: Path, stem: str):
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_csv(all_rows: dict[str, list[dict]], path: Path):
    keys = sorted({key for rows in all_rows.values() for row in rows for key in row})
    preferred = ["method", "step", "epoch", "accuracy", "reward", "loss"]
    fieldnames = preferred + [key for key in keys if key not in preferred]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in ("vanilla", "prime", "dpo_z"):
            writer.writerows(all_rows[method])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-log", type=Path, required=True)
    parser.add_argument("--prime-log", type=Path, required=True)
    parser.add_argument("--dpo-z-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=187)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {
        "vanilla": parse_log(args.vanilla_log, "vanilla", args.total_steps),
        "prime": parse_log(args.prime_log, "prime", args.total_steps),
        "dpo_z": parse_log(args.dpo_z_log, "dpo_z", args.total_steps),
    }
    if any(not rows for rows in all_rows.values()):
        missing = [method for method, rows in all_rows.items() if not rows]
        raise RuntimeError(f"No metric snapshots parsed for: {missing}")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    _plot(axes[0], all_rows, "accuracy")
    axes[0].set_ylabel("Train-batch accuracy")
    axes[0].set_ylim(0, 1)
    _mark_prime_failure(axes[0])
    _plot(axes[1], all_rows, "completions/mean_length")
    axes[1].set_ylabel("Mean completion length (tokens)")
    axes[1].set_ylim(bottom=0)
    _mark_prime_failure(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, args.output_dir, "accuracy-and-length")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    panels = [
        ("completions/clipped_ratio", "Responses hitting 16K cap (%)", 100.0),
        ("frac_reward_zero_std", "Prompt groups with zero reward std (%)", 100.0),
        ("grad_norm", "Actor gradient norm", 1.0),
        ("step_time", "Step time (seconds)", 1.0),
    ]
    for ax, (key, ylabel, scale) in zip(axes.flat, panels):
        _plot(ax, all_rows, key, scale=scale)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        _mark_prime_failure(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, args.output_dir, "common-training-metrics")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    panels = [
        ("loss", "Reported policy loss", 1.0),
        ("reward_std", "Reward std", 1.0),
        ("learning_rate", "Actor learning rate", 1.0),
    ]
    for ax, (key, ylabel, scale) in zip(axes, panels):
        _plot(ax, all_rows, key, scale=scale)
        ax.set_ylabel(ylabel)
        _mark_prime_failure(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, args.output_dir, "optimization-signals")

    write_csv(all_rows, args.output_dir / "metrics-snapshot.csv")
    summary = {
        "total_steps_planned": args.total_steps,
        "points": {method: len(rows) for method, rows in all_rows.items()},
        "latest": {method: rows[-1] for method, rows in all_rows.items()},
        "prime_failure_step": 9,
        "limitations": [
            "single seed",
            "runs are incomplete",
            "PRIME has one logged metric snapshot and failed after step 9",
            "metrics are train-batch aggregates, not held-out evaluation",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
