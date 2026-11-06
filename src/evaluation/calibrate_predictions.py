"""Fit calibration/thresholds on validation data and evaluate a protected test file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .calibration import TemperatureScaler, calibration_comparison
from .metrics import binary_classification_metrics, select_thresholds


def calibrate_and_evaluate(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    threshold_strategy: str = "f1_optimal",
    n_bins: int = 15,
) -> dict:
    required = {"label", "probability"}
    if not required.issubset(validation) or not required.issubset(test):
        raise ValueError("validation and test frames require label and probability columns")
    if "instance_id" in validation and "instance_id" in test:
        overlap = set(validation["instance_id"]).intersection(test["instance_id"])
        if overlap:
            raise ValueError(f"validation/test overlap detected for {len(overlap)} instance IDs")
    scaler = TemperatureScaler().fit(validation["label"], validation["probability"])
    validation_calibrated = scaler.transform(validation["probability"])
    thresholds = select_thresholds(validation["label"], validation_calibrated)
    if threshold_strategy not in thresholds:
        raise ValueError(f"unknown threshold strategy {threshold_strategy!r}; choose from {sorted(thresholds)}")
    threshold = thresholds[threshold_strategy]
    test_calibrated = scaler.transform(test["probability"])
    comparison = calibration_comparison(test["label"], test["probability"], test_calibrated, n_bins)
    calibrated_metrics = binary_classification_metrics(test["label"], test_calibrated, threshold, n_bins)
    uncalibrated_metrics = binary_classification_metrics(test["label"], test["probability"], threshold, n_bins)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated = test.copy()
    calibrated["uncalibrated_probability"] = calibrated.pop("probability")
    calibrated["probability"] = test_calibrated
    calibrated.to_csv(output_dir / "calibrated_test_predictions.csv", index=False)
    reliability_rows = []
    for stage in ("pre", "post"):
        for item in comparison[f"reliability_{stage}"]:
            reliability_rows.append({"stage": stage, **item})
    pd.DataFrame(reliability_rows).to_csv(output_dir / "calibration_reliability.csv", index=False)
    summary_rows = []
    for stage, metrics in (("pre", uncalibrated_metrics), ("post", calibrated_metrics)):
        summary_rows.append({"stage": stage, "temperature": 1.0 if stage == "pre" else scaler.temperature, **{key: metrics[key] for key in ("brier_score", "ece", "calibration_slope", "calibration_intercept")}})
    pd.DataFrame(summary_rows).to_csv(output_dir / "calibration_results.csv", index=False)
    payload = {
        "status": "available",
        "calibration_fit_partition": "validation",
        "evaluation_partition": "test",
        "temperature": scaler.temperature,
        "threshold_strategy": threshold_strategy,
        "threshold": threshold,
        "all_validation_selected_thresholds": thresholds,
        "uncalibrated_test_metrics": uncalibrated_metrics,
        "calibrated_test_metrics": calibrated_metrics,
    }
    (output_dir / "calibration_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    test_log = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": "final_test_calibration_evaluation",
        "calibration_fit_partition": "validation",
        "threshold_selection_partition": "validation",
        "test_rows": int(len(test)),
    }
    (output_dir.parent / "test_evaluation_log.json").write_text(json.dumps(test_log, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--threshold-strategy", default="f1_optimal", choices=["f1_optimal", "youden_j", "precision_oriented", "recall_oriented"])
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()
    calibrate_and_evaluate(pd.read_csv(args.validation), pd.read_csv(args.test), args.output_dir, args.threshold_strategy, args.n_bins)


if __name__ == "__main__":
    main()

