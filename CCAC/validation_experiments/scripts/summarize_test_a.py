#!/usr/bin/env python3
"""Summarize Test A classifier metrics."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="validation_experiments/outputs/test_a")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append(metrics)

    if not rows:
        raise SystemExit(f"No metrics found in {output_dir}")

    cols = [
        "run_name",
        "classifier_loss",
        "ood_usage",
        "auroc",
        "auprc",
        "false_negative_rate",
        "false_positive_rate",
        "ece",
        "accuracy",
        "positive_rate",
    ]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in cols))


if __name__ == "__main__":
    main()
