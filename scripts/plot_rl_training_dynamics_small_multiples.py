#!/usr/bin/env python3
"""Render RL log comparisons as non-overlapping method small multiples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_rl_training_dynamics import COLORS, METHODS, parse_log, write_csv


ORDER = ("vanilla", "prime", "dpo_z")


def _series(rows, key):
    points = [(row["step"], row.get(key)) for row in rows if row.get(key) is not None]
    return [point[0] for point in points], [point[1] for point in points]


def _panel(ax, rows, method, key, *, scale=1.0, x_max=None):
    x, y = _series(rows, key)
    ax.plot(
        x,
        [value * scale for value in y],
        color=COLORS[method],
        linewidth=1.8,
        marker="o",
        markersize=4.5,
    )
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_xlabel("Optimizer step")
    ax.set_xlim(0, x_max)
    if method == "prime":
        ax.axvline(9, color=COLORS["prime"], linestyle=":", alpha=0.7)


def _save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _metric_grid(all_rows, metrics, output_dir, stem, *, width=9.5):
    rows_count = len(metrics)
    fig, axes = plt.subplots(
        rows_count,
        3,
        figsize=(width, 2.15 * rows_count + 0.55),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    x_max = max(row["step"] for rows in all_rows.values() for row in rows) + 5
    for column, method in enumerate(ORDER):
        axes[0, column].set_title(METHODS[method], fontsize=10)
        for row_index, (key, ylabel, scale) in enumerate(metrics):
            ax = axes[row_index, column]
            _panel(ax, all_rows[method], method, key, scale=scale, x_max=x_max)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row_index < rows_count - 1:
                ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, output_dir, stem)


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
        raise RuntimeError("Every method must have at least one parsed metric snapshot")

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    _metric_grid(
        all_rows,
        [
            ("accuracy", "Train-batch accuracy", 1.0),
            ("completions/mean_length", "Mean length (tokens)", 1.0),
        ],
        args.output_dir,
        "accuracy-and-length",
    )
    _metric_grid(
        all_rows,
        [
            ("completions/clipped_ratio", "16K clipped (%)", 100.0),
            ("frac_reward_zero_std", "Zero-std groups (%)", 100.0),
            ("grad_norm", "Actor grad norm", 1.0),
            ("step_time", "Step time (s)", 1.0),
        ],
        args.output_dir,
        "common-training-metrics",
    )
    _metric_grid(
        all_rows,
        [
            ("loss", "Reported policy loss", 1.0),
            ("reward_std", "Reward std", 1.0),
            ("learning_rate", "Actor learning rate", 1.0),
        ],
        args.output_dir,
        "optimization-signals",
    )

    write_csv(all_rows, args.output_dir / "metrics-snapshot.csv")
    common_steps = sorted(
        set(row["step"] for row in all_rows["vanilla"])
        & set(row["step"] for row in all_rows["dpo_z"])
    )
    output_keys = (
        "accuracy",
        "completions/mean_length",
        "completions/clipped_ratio",
        "reward_std",
    )
    vanilla = {row["step"]: row for row in all_rows["vanilla"]}
    dpo_z = {row["step"]: row for row in all_rows["dpo_z"]}
    exact_output_match = all(
        vanilla[step].get(key) == dpo_z[step].get(key)
        for step in common_steps
        for key in output_keys
    )
    summary = {
        "total_steps_planned": args.total_steps,
        "points": {method: len(rows) for method, rows in all_rows.items()},
        "latest": {method: rows[-1] for method, rows in all_rows.items()},
        "prime_failure_step": 9,
        "grpo_dpo_z_common_logged_steps": common_steps,
        "grpo_dpo_z_exact_output_metric_match": exact_output_match,
        "exact_match_keys": list(output_keys),
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
