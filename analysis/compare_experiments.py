"""
Compare experiment results and generate summary tables and plots.

Usage:
    python analysis/compare_experiments.py --results-dir sft/exp1_data/results
    python analysis/compare_experiments.py --all
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

logger = setup_logging("analysis")


def load_all_results(results_dirs: list[str]) -> pd.DataFrame:
    """Load all result JSON files into a DataFrame."""
    rows = []
    for results_dir in results_dirs:
        for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            with open(path) as f:
                data = json.load(f)

            row = {
                "file": os.path.basename(path),
                "experiment": data.get("experiment", {}).get("name", "unknown"),
                "model": data.get("model", "unknown"),
            }

            for bench in ["gsm8k", "math500"]:
                if bench in data and isinstance(data[bench], dict):
                    row[f"{bench}_accuracy"] = data[bench].get("accuracy", 0)
                    row[f"{bench}_format"] = data[bench].get("format_compliance", 0)
                    row[f"{bench}_avg_len"] = data[bench].get("avg_response_length", 0)

            rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    if df.empty:
        logger.info("No results found!")
        return

    cols = ["experiment"]
    for bench in ["gsm8k", "math500"]:
        if f"{bench}_accuracy" in df.columns:
            cols.extend([
                f"{bench}_accuracy",
                f"{bench}_format",
            ])

    subset = df[cols].copy()
    subset.columns = [c.replace("_accuracy", " acc%").replace("_format", " fmt%") for c in cols]

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(subset.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--results-dir", nargs="+", default=None,
        help="Directory with result JSON files",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Load all results from all phases",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.all:
        results_dirs = sorted(glob.glob(str(project_root / "sft" / "*" / "results")))
    elif args.results_dir:
        results_dirs = args.results_dir
    else:
        results_dirs = sorted(glob.glob(str(project_root / "sft" / "*" / "results")))

    df = load_all_results(results_dirs)
    print_comparison_table(df)

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
