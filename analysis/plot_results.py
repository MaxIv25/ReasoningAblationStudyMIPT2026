"""
Plot evaluation results from JSON files in results/ directory.

Usage:
    python analysis/plot_results.py
    python analysis/plot_results.py --results-dir results/ --output analysis/figures/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ── Friendly names for display ──
DISPLAY_NAMES = {
    "baseline_0.8b": "Base (0.8B)",
    "exp1_1_fullft": "Full FT",
    "exp2_2_lora": "LoRA r=64",
    "exp2_3_dora": "DoRA",
    "exp2_4_pissa": "PiSSA",
    "exp3_1_curriculum": "Curriculum",
    "exp3_2_prompt_mask": "Prompt Mask",
}

BENCH_DISPLAY = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "aime2026": "AIME 2026",
    "math_hard": "MATH-Hard",
}

COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#E45756",  # red
    "#72B7B2",  # teal
    "#54A24B",  # green
    "#EECA3B",  # yellow
    "#B279A2",  # purple
    "#FF9DA6",  # pink
]


def load_results(results_dir: str) -> dict:
    """Load all JSON result files from directory."""
    results = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        name = path.stem  # e.g., "baseline_0.8b", "exp2_2_lora"
        with open(path) as f:
            results[name] = json.load(f)
    return results


def plot_accuracy_comparison(results: dict, output_dir: str):
    """Bar chart comparing accuracy across experiments and benchmarks."""
    benchmarks = ["gsm8k", "math500"]
    experiments = list(results.keys())

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 5))
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        names = []
        accs = []
        colors = []

        for i, (exp_name, data) in enumerate(results.items()):
            if bench in data:
                display_name = DISPLAY_NAMES.get(exp_name, exp_name)
                names.append(display_name)
                accs.append(data[bench]["accuracy"])
                colors.append(COLORS[i % len(COLORS)])

        bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("SFT Ablation: Accuracy Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_format_compliance(results: dict, output_dir: str):
    """Bar chart for format compliance across experiments."""
    benchmarks = ["gsm8k", "math500"]
    experiments = list(results.keys())

    fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.5), 5))

    x = np.arange(len(experiments))
    width = 0.35

    for j, bench in enumerate(benchmarks):
        vals = []
        for exp_name in experiments:
            data = results[exp_name]
            vals.append(data[bench]["format_compliance"] if bench in data else 0)

        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=BENCH_DISPLAY.get(bench, bench),
                      color=COLORS[j], edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    display_names = [DISPLAY_NAMES.get(e, e) for e in experiments]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=20)
    ax.set_ylabel("Format Compliance (%)")
    ax.set_title("Format Compliance (\\boxed{} present)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "format_compliance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_response_length(results: dict, output_dir: str):
    """Bar chart for average response length."""
    benchmarks = ["gsm8k", "math500"]
    experiments = list(results.keys())

    fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.5), 5))

    x = np.arange(len(experiments))
    width = 0.35

    for j, bench in enumerate(benchmarks):
        vals = []
        for exp_name in experiments:
            data = results[exp_name]
            vals.append(data[bench]["avg_response_length"] if bench in data else 0)

        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=BENCH_DISPLAY.get(bench, bench),
                      color=COLORS[j], edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    display_names = [DISPLAY_NAMES.get(e, e) for e in experiments]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=20)
    ax.set_ylabel("Avg Response Length (words)")
    ax.set_title("Average Response Length", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "response_length.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_summary_table(results: dict, output_dir: str):
    """Generate a summary table as an image."""
    benchmarks = ["gsm8k", "math500"]
    experiments = list(results.keys())

    headers = ["Model"]
    for bench in benchmarks:
        headers.extend([f"{BENCH_DISPLAY.get(bench, bench)} Acc", "Format"])

    rows = []
    for exp_name in experiments:
        data = results[exp_name]
        display_name = DISPLAY_NAMES.get(exp_name, exp_name)
        row = [display_name]
        for bench in benchmarks:
            if bench in data:
                row.append(f"{data[bench]['accuracy']}%")
                row.append(f"{data[bench]['format_compliance']}%")
            else:
                row.extend(["-", "-"])
        rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 1 + 0.5 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for j, cell in table.get_celld().items():
        if j[0] == 0:
            cell.set_facecolor("#4C78A8")
            cell.set_text_props(color="white", fontweight="bold")
        elif j[0] % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#ddd")

    fig.suptitle("SFT Ablation Study — Results Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/", help="Directory with JSON results")
    parser.add_argument("--output", default="analysis/figures/", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} results: {list(results.keys())}")

    plot_accuracy_comparison(results, args.output)
    plot_format_compliance(results, args.output)
    plot_response_length(results, args.output)
    plot_summary_table(results, args.output)

    print(f"\nAll plots saved to {args.output}")


if __name__ == "__main__":
    main()
