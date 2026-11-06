"""N8 - is one global temperature enough?

Temperature scaling is fitted once on validation and applied to everyone. That is
standard practice and usually reported as a single expected-calibration-error
number. It hides the question that matters for deployment: whether the resulting
probabilities are equally trustworthy for a beginner and an advanced learner, on
a phone and on the web, in one country and another.

This module fits the global temperature, then measures calibration within each
subgroup, and finally fits per-group temperatures to bound how much could be
gained by group-wise recalibration. Small subgroups are flagged rather than
quietly reported, because calibration error is badly estimated at low counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

MIN_SUBGROUP_SIZE = 1000


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 15
) -> dict[str, float]:
    """Equal-width binned ECE plus the maximum single-bin deviation."""

    if len(labels) == 0:
        raise ValueError("labels must be non-empty")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assigned = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(labels)
    ece = 0.0
    max_gap = 0.0
    for index in range(n_bins):
        mask = assigned == index
        count = int(mask.sum())
        if not count:
            continue
        gap = abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
        ece += (count / total) * gap
        max_gap = max(max_gap, gap)
    return {
        "ece": float(ece),
        "max_calibration_error": float(max_gap),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "mean_predicted": float(probabilities.mean()),
        "observed_rate": float(labels.mean()),
        "n": total,
    }


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit a single scalar temperature on logits by minimizing log loss."""

    from scipy.optimize import minimize_scalar

    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))

    def negative_log_likelihood(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
        scaled = np.clip(scaled, 1e-12, 1 - 1e-12)
        return float(-np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)))

    result = minimize_scalar(negative_log_likelihood, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def proficiency_decile(frame: pd.DataFrame, column: str = "prior_error_rate") -> pd.Series:
    """Bucket interactions by the learner's past-only error rate."""

    if column not in frame.columns:
        raise KeyError(f"Missing proficiency column {column!r}")
    return pd.qcut(frame[column].rank(method="first"), q=10, labels=False, duplicates="drop")


def subgroup_calibration(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    attributes: Iterable[str],
    min_size: int = MIN_SUBGROUP_SIZE,
) -> dict[str, Any]:
    """Global versus group-wise temperature scaling, per subgrouping attribute.

    The global temperature and every group temperature are fitted on validation
    only; all reported calibration is measured on test.
    """

    global_temperature = fit_temperature(
        validation["label"].to_numpy(dtype=float), validation["probability"].to_numpy(dtype=float)
    )
    test_probability = test["probability"].to_numpy(dtype=float)
    test_labels = test["label"].to_numpy(dtype=float)
    globally_scaled = apply_temperature(test_probability, global_temperature)

    overall = {
        "global_temperature": global_temperature,
        "uncalibrated": expected_calibration_error(test_labels, test_probability),
        "globally_calibrated": expected_calibration_error(test_labels, globally_scaled),
    }

    results: dict[str, Any] = {}
    for attribute in attributes:
        if attribute not in test.columns or attribute not in validation.columns:
            results[attribute] = {"status": "unavailable"}
            continue
        rows: list[dict[str, Any]] = []
        flagged: list[str] = []
        for value, block in test.groupby(test[attribute].astype(str), observed=True):
            if len(block) < min_size:
                flagged.append(str(value))
                continue
            reference = validation[validation[attribute].astype(str) == str(value)]
            group_temperature = None
            group_metrics = None
            if len(reference) >= min_size and reference["label"].nunique() == 2:
                group_temperature = fit_temperature(
                    reference["label"].to_numpy(dtype=float),
                    reference["probability"].to_numpy(dtype=float),
                )
                group_metrics = expected_calibration_error(
                    block["label"].to_numpy(dtype=float),
                    apply_temperature(block["probability"].to_numpy(dtype=float), group_temperature),
                )
            under_global = expected_calibration_error(
                block["label"].to_numpy(dtype=float),
                apply_temperature(block["probability"].to_numpy(dtype=float), global_temperature),
            )
            rows.append(
                {
                    "subgroup": str(value),
                    "n": int(len(block)),
                    "global_temperature_ece": under_global["ece"],
                    "global_temperature_max_error": under_global["max_calibration_error"],
                    "observed_rate": under_global["observed_rate"],
                    "mean_predicted": under_global["mean_predicted"],
                    "group_temperature": group_temperature,
                    "group_temperature_ece": group_metrics["ece"] if group_metrics else None,
                    "ece_reduction_from_group_temperature": (
                        under_global["ece"] - group_metrics["ece"] if group_metrics else None
                    ),
                }
            )
        if not rows:
            results[attribute] = {"status": "insufficient_data", "suppressed_subgroups": flagged}
            continue
        table = pd.DataFrame(rows)
        gains = table["ece_reduction_from_group_temperature"].dropna()
        results[attribute] = {
            "status": "tested",
            "subgroups": rows,
            "worst_subgroup_ece": float(table["global_temperature_ece"].max()),
            "best_subgroup_ece": float(table["global_temperature_ece"].min()),
            "ece_spread": float(table["global_temperature_ece"].max() - table["global_temperature_ece"].min()),
            "median_gain_from_group_temperature": float(gains.median()) if len(gains) else None,
            "group_temperature_helps": bool(len(gains) and gains.median() > 0.005),
            "suppressed_small_subgroups": flagged,
        }
    return {
        "overall": overall,
        "by_attribute": results,
        "protocol_note": (
            "Every temperature, global or group-wise, is fitted on the validation partition. "
            "Test labels are used only to measure the resulting calibration."
        ),
        "small_subgroup_policy": f"Subgroups with fewer than {min_size} test interactions are suppressed, not reported.",
    }


def run(
    root: Path,
    *,
    seed: int = 42,
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Execute N8 against the frozen proposed checkpoint."""

    from src.analysis.inference import (
        collect_predictions,
        indexed_dataloader,
        join_corpus_attributes,
        load_proposed_model,
    )

    output_dir = root / "results" / "analysis" / "subgroup_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    attributes = ["countries", "client", "session_type", "exercise_format"]
    collected: dict[str, pd.DataFrame] = {}
    frame = None
    for split in ("validation", "test"):
        loader, frame = indexed_dataloader(
            root, split=split, sequence_length=sequence_length, batch_size=batch_size, frame=frame
        )
        predictions = collect_predictions(model, loader, resolved_device, with_expert_weights=False)
        enriched = join_corpus_attributes(predictions, frame, attributes + ["prior_error_rate"])
        enriched["probability"] = enriched["logit_probability"]
        enriched["proficiency_decile"] = proficiency_decile(enriched)
        collected[split] = enriched

    payload = subgroup_calibration(
        collected["validation"],
        collected["test"],
        attributes=attributes + ["proficiency_decile"],
    )
    payload["provenance"] = provenance
    for attribute, result in payload["by_attribute"].items():
        if result.get("status") == "tested":
            pd.DataFrame(result["subgroups"]).to_csv(
                output_dir / f"calibration_by_{attribute}.csv", index=False
            )
    (output_dir / "subgroup_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
