"""CLI for evaluating real prediction files without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import binary_classification_metrics


def evaluate_file(path: Path, output: Path, threshold: float, label_col: str, probability_col: str) -> dict:
    frame = pd.read_csv(path)
    missing = {label_col, probability_col} - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file is missing required columns: {sorted(missing)}")
    metrics = binary_classification_metrics(frame[label_col], frame[probability_col], threshold=threshold)
    payload = {"status": "available", "source_file": str(path.resolve()), "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Validation-selected threshold; no test optimization is performed")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--probability-col", default="probability")
    args = parser.parse_args()
    evaluate_file(args.predictions, args.output, args.threshold, args.label_col, args.probability_col)


if __name__ == "__main__":
    main()

