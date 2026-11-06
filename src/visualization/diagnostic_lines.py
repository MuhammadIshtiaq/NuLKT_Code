"""Additional evidence-based diagnostic visualizations for model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.evaluation.metrics import binary_classification_metrics
from .style import PALETTE, configure_publication_style


def _save(fig: plt.Figure, stem: Path, dpi: int) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def _validate(predictions: Mapping[str, pd.DataFrame]) -> None:
    if not predictions:
        raise ValueError("predictions cannot be empty")
    for name, frame in predictions.items():
        if not {"label", "probability"}.issubset(frame.columns):
            raise ValueError(f"{name} requires label and probability columns")
        if frame.empty or frame["label"].nunique() != 2:
            raise ValueError(f"{name} requires both outcome classes")


def generate_diagnostic_visualizations(
    predictions: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    title_suffix: str = "",
    watermark: str | None = None,
    dpi: int = 800,
) -> dict[str, list[str]]:
    """Generate calibration, threshold, gain, score-ECDF, and confusion plots."""

    _validate(predictions)
    configure_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" — {title_suffix}" if title_suffix else ""
    registry: dict[str, list[str]] = {}

    # Reliability diagram with numeric calibration evidence in each legend label.
    fig, axis = plt.subplots(figsize=(7.8, 6.7), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.4", label="Perfect calibration")
    for index, (name, frame) in enumerate(predictions.items()):
        y = frame["label"].to_numpy(dtype=int)
        p = frame["probability"].to_numpy(dtype=float)
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        metrics = binary_classification_metrics(y, p)
        color = PALETTE[index % len(PALETTE)]
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2.1,
            color=color,
            label=f"{name} (ECE={metrics['ece']:.3f}, Brier={metrics['brier_score']:.3f})",
        )
        for label_index, (x_value, y_value) in enumerate(zip(predicted[::3], observed[::3])):
            axis.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                xytext=(6, 6 + 11 * (label_index % 2)),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted error probability", ylabel="Observed error frequency", title=f"Reliability diagram{suffix}")
    axis.grid(False)
    axis.legend(frameon=False, loc="upper left")
    if watermark:
        axis.text(0.5, -0.14, watermark, transform=axis.transAxes, ha="center", fontsize=10)
    registry["calibration_reliability"] = _save(fig, output_dir / "calibration_reliability_line", dpi)

    # Validation threshold behavior, one panel per model to avoid crossing-label clutter.
    thresholds = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(len(predictions), 1, figsize=(10.2, 4.8 * len(predictions)), sharex=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for panel, ((name, frame), axis) in enumerate(zip(predictions.items(), axes_array)):
        y = frame["label"].to_numpy(dtype=int)
        p = frame["probability"].to_numpy(dtype=float)
        curves = {"Precision": [], "Recall": [], "F1": [], "Accuracy": []}
        for threshold in thresholds:
            decision = (p >= threshold).astype(int)
            curves["Precision"].append(precision_score(y, decision, zero_division=0))
            curves["Recall"].append(recall_score(y, decision, zero_division=0))
            curves["F1"].append(f1_score(y, decision, zero_division=0))
            curves["Accuracy"].append(accuracy_score(y, decision))
        for index, (metric, values) in enumerate(curves.items()):
            axis.plot(thresholds, values, linewidth=2.0, marker="o", markersize=3.5, color=PALETTE[index % len(PALETTE)], label=metric)
        best_index = int(np.argmax(curves["F1"]))
        best_threshold, best_f1 = thresholds[best_index], curves["F1"][best_index]
        axis.axvline(best_threshold, linestyle="--", linewidth=1.2, color="0.35")
        axis.annotate(
            f"Best validation F1={best_f1:.3f}\nat threshold={best_threshold:.2f}",
            (best_threshold, best_f1),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "0.35"},
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("Metric value")
        axis.set_title(name)
        axis.grid(False)
        axis.legend(frameon=False, ncol=4, loc="lower center")
    axes_array[-1].set_xlabel("Decision threshold")
    fig.suptitle(f"Validation threshold trade-offs{suffix}", fontsize=16)
    if watermark:
        fig.text(0.5, -0.015, watermark, ha="center", fontsize=10)
    registry["threshold_tradeoff"] = _save(fig, output_dir / "threshold_tradeoff_lines", dpi)

    # Cumulative gain: proportion of all errors captured by reviewing highest-risk cases.
    fig, axis = plt.subplots(figsize=(7.8, 6.7), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.4", label="Random review")
    for index, (name, frame) in enumerate(predictions.items()):
        ordered = frame.sort_values("probability", ascending=False)
        labels = ordered["label"].to_numpy(dtype=float)
        population = np.arange(1, len(labels) + 1) / len(labels)
        captured = np.cumsum(labels) / labels.sum()
        color = PALETTE[index % len(PALETTE)]
        twenty = int(np.searchsorted(population, 0.20))
        gain_20 = float(captured[min(twenty, len(captured) - 1)])
        axis.plot(population, captured, linewidth=2.2, color=color, label=f"{name} (top 20% captures {gain_20:.1%})")
        axis.scatter([population[twenty]], [captured[twenty]], color=color, s=30, zorder=3)
        axis.annotate(f"{gain_20:.1%}", (population[twenty], captured[twenty]), xytext=(6, 6), textcoords="offset points", color=color, fontsize=10)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Fraction of interactions reviewed", ylabel="Fraction of observed errors captured", title=f"Cumulative error-capture gain{suffix}")
    axis.grid(False)
    axis.legend(frameon=False, loc="lower right")
    if watermark:
        axis.text(0.5, -0.14, watermark, transform=axis.transAxes, ha="center", fontsize=10)
    registry["cumulative_gain"] = _save(fig, output_dir / "cumulative_error_capture", dpi)

    # Empirical CDFs reveal separation of predicted scores for errors and correct responses.
    fig, axes = plt.subplots(1, len(predictions), figsize=(7.0 * len(predictions), 5.8), sharex=True, sharey=True, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for (name, frame), axis in zip(predictions.items(), axes_array):
        for label, description, color in [(0, "Correct response", PALETTE[2]), (1, "Observed error", PALETTE[3])]:
            values = np.sort(frame.loc[frame["label"] == label, "probability"].to_numpy(dtype=float))
            cumulative = np.arange(1, len(values) + 1) / len(values)
            median = float(np.median(values))
            axis.plot(values, cumulative, linewidth=2.2, color=color, label=f"{description} (median={median:.3f})")
        axis.set_title(name)
        axis.set_xlabel("Predicted error probability")
        axis.grid(False)
        axis.legend(frameon=False, loc="lower right")
    axes_array[0].set_ylabel("Cumulative proportion")
    fig.suptitle(f"Prediction-score distributions{suffix}", fontsize=16)
    if watermark:
        fig.text(0.5, -0.02, watermark, ha="center", fontsize=10)
    registry["score_distribution"] = _save(fig, output_dir / "prediction_score_ecdf", dpi)

    # Confusion matrices with exact counts written in each cell.
    fig, axes = plt.subplots(1, len(predictions), figsize=(6.2 * len(predictions), 5.2), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for (name, frame), axis in zip(predictions.items(), axes_array):
        metrics = binary_classification_metrics(frame["label"], frame["probability"], threshold=0.5)
        matrix = np.asarray(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
        )
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                text_color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=text_color,
                )
        axis.set_xticks([0, 1], ["Predicted correct", "Predicted error"])
        axis.set_yticks([0, 1], ["Observed correct", "Observed error"])
        axis.set_title(f"{name}\nthreshold = 0.50")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Confusion matrices with exact counts{suffix}", fontsize=16)
    if watermark:
        fig.text(0.5, -0.02, watermark, ha="center", fontsize=10)
    registry["confusion_matrix"] = _save(fig, output_dir / "confusion_matrices", dpi)

    (output_dir / "diagnostic_visualizations.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry
