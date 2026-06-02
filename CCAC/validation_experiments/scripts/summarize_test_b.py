#!/usr/bin/env python3
"""Summarize Test B cost-critic separation metrics."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="validation_experiments/outputs/test_b")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        with open(metrics_path) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"No metrics found in {output_dir}")

    cols = [
        "run_name",
        "classifier_loss",
        "ood_mode",
        "qc_ind_mean",
        "qc_matched_ind_mean",
        "qc_ood_mean",
        "qc_gap",
        "qc_matched_gap",
        "qc_selected_ood_mean",
        "qc_selected_gap",
        "qc_selected_matched_gap",
        "ood_score_mean",
        "ood_selected_rate",
    ]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in cols))


if __name__ == "__main__":
    main()
