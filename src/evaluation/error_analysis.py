"""Generate subgroup failure analysis strictly from prediction records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .metrics import binary_classification_metrics


def analyze_subgroups(
    predictions: pd.DataFrame,
    group_columns: Iterable[str],
    output_dir: Path,
    threshold: float,
    minimum_n: int = 100,
) -> pd.DataFrame:
    required = {"label", "probability"}
    if not required.issubset(predictions):
        raise ValueError(f"predictions require columns {sorted(required)}")
    rows = []
    for column in group_columns:
        if column not in predictions:
            continue
        for level, frame in predictions.groupby(column, dropna=False, observed=True):
            if len(frame) < minimum_n:
                rows.append({"subgroup_variable": column, "subgroup": str(level), "n": len(frame), "status": "withheld_small_subgroup"})
                continue
            metrics = binary_classification_metrics(frame["label"], frame["probability"], threshold=threshold)
            rows.append({"subgroup_variable": column, "subgroup": str(level), "status": "available", **{k: metrics[k] for k in ("n", "positive_rate", "auroc", "auprc", "f1", "brier_score", "ece")}})
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "subgroup_error_analysis.csv", index=False)
    return result


def write_error_analysis_report(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if table.empty or not table.get("status", pd.Series(dtype=str)).eq("available").any():
        text = "# Error Analysis\n\nExperiment unavailable/failed: no eligible prediction subgroups were supplied. No failure-condition claims are made.\n"
    else:
        text = (
            "# Error Analysis\n\n"
            "This report is derived from `subgroup_error_analysis.csv`. Small subgroups are withheld and all analyses are descriptive, not causal.\n\n"
            + table.to_markdown(index=False)
            + "\n"
        )
    destination.write_text(text, encoding="utf-8")

