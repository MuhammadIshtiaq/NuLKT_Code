"""Annotated line graphs for evaluation metrics and discrimination curves."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import binary_classification_metrics
from .style import PALETTE, configure_publication_style


def _save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def _validate_predictions(predictions: Mapping[str, pd.DataFrame]) -> None:
    if not predictions:
        raise ValueError("At least one model prediction frame is required")
    for model, frame in predictions.items():
        missing = {"label", "probability"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{model} predictions are missing {sorted(missing)}")
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy(dtype=float)
        if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{model} requires non-empty binary labels")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{model} probabilities must be finite and within [0, 1]")


def generate_evaluation_line_graphs(
    predictions: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    title_suffix: str = "",
    watermark: str | None = None,
    dpi: int = 800,
) -> dict[str, list[str]]:
    """Generate separate scalar-metric plots plus valid ROC and PR curves."""

    _validate_predictions(predictions)
    configure_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" — {title_suffix}" if title_suffix else ""
    metric_keys = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Acc."),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("mcc", "MCC"),
    ]
    computed = {
        model: binary_classification_metrics(frame["label"], frame["probability"], threshold=0.5)
        for model, frame in predictions.items()
    }
    model_names = list(predictions)
    metric_paths: list[Path] = []
    prevalence = float(next(iter(predictions.values()))["label"].mean())
    for metric_key, metric_label in metric_keys:
        values = np.asarray([computed[model][metric_key] for model in model_names], dtype=float)
        positions = np.arange(len(model_names))
        fig, axis = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
        for index, (position, value) in enumerate(zip(positions, values)):
            color = PALETTE[index % len(PALETTE)]
            axis.scatter(position, value, s=90, color=color, edgecolor="black", linewidth=0.7, zorder=3)
            axis.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                color=color,
            )
        if metric_key == "auroc":
            axis.axhline(0.5, linestyle="--", linewidth=1.2, color="0.4", label="Chance = 0.500")
        elif metric_key == "auprc":
            axis.axhline(prevalence, linestyle="--", linewidth=1.2, color="0.4", label=f"Prevalence = {prevalence:.3f}")
        elif metric_key == "mcc":
            axis.axhline(0.0, linestyle="--", linewidth=1.2, color="0.4", label="No association = 0")
        axis.set_xticks(positions, model_names)
        axis.set_xlim(-0.5, len(model_names) - 0.5)
        axis.set_ylim((-1.05, 1.05) if metric_key == "mcc" else (-0.05, 1.05))
        axis.set_ylabel(metric_label)
        axis.set_xlabel("Model")
        axis.set_title(f"{metric_label}{suffix}")
        axis.grid(False)
        if metric_key in {"auroc", "auprc", "mcc"}:
            axis.legend(frameon=False, loc="lower right")
        if watermark:
            axis.text(0.5, -0.18, watermark, transform=axis.transAxes, ha="center", fontsize=10)
        metric_paths.extend(_save(fig, output_dir / f"metric_{metric_key}_dot", dpi))

    fig, axis = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#666666", label="Chance (AUC = 0.500)")
    for index, (model, frame) in enumerate(predictions.items()):
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy(dtype=float)
        false_positive, true_positive, _ = roc_curve(labels, probabilities)
        auc_value = roc_auc_score(labels, probabilities)
        color = PALETTE[index % len(PALETTE)]
        axis.plot(false_positive, true_positive, linewidth=2.3, color=color, label=f"{model} (AUC = {auc_value:.3f})")
        point = int(np.argmin(np.abs(false_positive - 0.5)))
        axis.annotate(
            f"AUC {auc_value:.3f}",
            (false_positive[point], true_positive[point]),
            xytext=(8, 6 + index * 12),
            textcoords="offset points",
            color=color,
            fontsize=10,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curves{suffix}")
    axis.grid(False)
    axis.legend(frameon=False, loc="lower right")
    if watermark:
        axis.text(0.5, -0.14, watermark, transform=axis.transAxes, ha="center", fontsize=10)
    roc_paths = _save(fig, output_dir / "roc_curve_line", dpi)

    fig, axis = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    prevalence = float(next(iter(predictions.values()))["label"].mean())
    axis.axhline(prevalence, linestyle="--", linewidth=1.2, color="#666666", label=f"Prevalence = {prevalence:.3f}")
    for index, (model, frame) in enumerate(predictions.items()):
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        average_precision = average_precision_score(labels, probabilities)
        color = PALETTE[index % len(PALETTE)]
        axis.plot(recall, precision, linewidth=2.3, color=color, label=f"{model} (AP = {average_precision:.3f})")
        point = int(np.argmin(np.abs(recall - 0.5)))
        axis.annotate(
            f"AP {average_precision:.3f}",
            (recall[point], precision[point]),
            xytext=(8, 6 + index * 12),
            textcoords="offset points",
            color=color,
            fontsize=10,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title=f"Precision–Recall curves{suffix}")
    axis.grid(False)
    axis.legend(frameon=False, loc="best")
    if watermark:
        axis.text(0.5, -0.14, watermark, transform=axis.transAxes, ha="center", fontsize=10)
    pr_paths = _save(fig, output_dir / "precision_recall_curve_line", dpi)

    registry = {
        "evaluation_metrics": [str(path) for path in metric_paths],
        "roc_curve": [str(path) for path in roc_paths],
        "precision_recall_curve": [str(path) for path in pr_paths],
    }
    (output_dir / "evaluation_line_graphs.json").write_text(
        __import__("json").dumps(registry, indent=2), encoding="utf-8"
    )
    return registry
