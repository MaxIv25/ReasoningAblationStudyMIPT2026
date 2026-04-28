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

    data = []
    for filepath in results_dir.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                res = json.load(f)
                
            run_name = filepath.stem
            
            row = {"Experiment": run_name}
            for bench, metrics in res.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    row[bench] = metrics["accuracy"]
            data.append(row)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    df = df.fillna("-")
    
    # Sort columns to have Experiment first
    cols = ["Experiment"] + [c for c in df.columns if c != "Experiment"]
    df = df[cols]

    print("\n### Сравнительная таблица результатов (Accuracy %)\n")
    print(df.to_markdown(index=False))
    
    # Save to markdown file
    with open("results/comparison_table.md", "w") as f:
        f.write("### Сравнительная таблица результатов (Accuracy %)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("\nTable saved to results/comparison_table.md")

if __name__ == "__main__":
    main()
