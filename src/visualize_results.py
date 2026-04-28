import argparse
import json
import os
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Create a comparison table from evaluation results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing result JSONs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory {args.results_dir} not found.")
        return

    data_accuracy = []
    data_format = []
    data_length = []
    
    benchmarks = set()
    for filepath in sorted(results_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                res = json.load(f)
                
            run_name = filepath.stem
            
            row_acc = {"Experiment": run_name}
            row_fmt = {"Experiment": run_name}
            row_len = {"Experiment": run_name}
            
            for bench, metrics in res.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    benchmarks.add(bench)
                    row_acc[bench] = metrics.get("accuracy", "-")
                    row_fmt[bench] = metrics.get("format_compliance", "-")
                    row_len[bench] = metrics.get("avg_response_length", "-")
            
            data_accuracy.append(row_acc)
            data_format.append(row_fmt)
            data_length.append(row_len)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not data_accuracy:
        print("No results found.")
        return

    benchmarks = sorted(list(benchmarks))
    headers = ["Experiment"] + benchmarks

    def print_table(data_list, title):
        md = f"\n### {title}\n\n"
        md += "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data_list:
            md += "| " + " | ".join(str(row.get(h, "-")) for h in headers) + " |\n"
        return md

    md_output = ""
    md_output += print_table(data_accuracy, "Accuracy (%)")
    md_output += print_table(data_format, "Format Compliance (%)")
    md_output += print_table(data_length, "Average Response Length (words)")

    print(md_output)
    
    # Save to markdown file
    with open("results/comparison_table.md", "w") as f:
        f.write(md_output)
    print("\nTable saved to results/comparison_table.md")

if __name__ == "__main__":
    main()
