"""
src/eval/compare_results.py

Loads JSON result files produced by augmentation_robustness.py and prints a
side-by-side comparison table.

Usage:
    python src/eval/compare_results.py results_baseline.json results_noaug.json results_aug.json
"""

import sys
import json


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("Usage: python compare_results.py <file1.json> <file2.json> [more files...]")
        sys.exit(1)

    loaded = []
    for path in paths:
        with open(path) as f:
            loaded.append(json.load(f))

    conditions = list(loaded[0]["results"].keys())
    labels = [d["label"] if len(d["label"]) < 30 else "..." + d["label"][-27:] for d in loaded]

    col_width = max(12, max(len(l) for l in labels) + 2)
    header = f"{'Condition':<15}" + "".join(f"{l:>{col_width}}" for l in labels)
    print(header)
    print("-" * len(header))
    for cond in conditions:
        row = f"{cond:<15}"
        for d in loaded:
            row += f"{d['results'][cond]:>{col_width}.4f}"
        print(row)


if __name__ == "__main__":
    main()