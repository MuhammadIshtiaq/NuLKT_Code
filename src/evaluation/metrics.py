"""Binary prediction metrics with explicit handling of undefined quantities.

Metrics are never imputed.  A metric that cannot be computed (for example,
AUROC in a single-class subgroup) is returned as ``None`` with a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_rate: float | None
    absolute_gap: float | None


def _arrays(y_true: Iterable[int], y_prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=np.int8).reshape(-1)
    p = np.asarray(list(y_prob), dtype=np.float64).reshape(-1)
    if y.size != p.size or y.size == 0:
        raise ValueError("y_true and y_prob must be non-empty and equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    return y, p


def reliability_bins(
    y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 15
) -> list[ReliabilityBin]:
    """Return equal-width reliability bins; empty bins remain explicit."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=True), n_bins - 1)
    result: list[ReliabilityBin] = []
    for idx in range(n_bins):
        mask = indices == idx
        if not np.any(mask):
            result.append(ReliabilityBin(float(edges[idx]), float(edges[idx + 1]), 0, None, None, None))
            continue
        mp, obs = float(p[mask].mean()), float(y[mask].mean())
        result.append(ReliabilityBin(float(edges[idx]), float(edges[idx + 1]), int(mask.sum()), mp, obs, abs(mp - obs)))
    return result


def expected_calibration_error(
    y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 15
) -> float:
    bins = reliability_bins(y_true, y_prob, n_bins)
    n = sum(item.count for item in bins)
    return float(sum((item.count / n) * (item.absolute_gap or 0.0) for item in bins))


def calibration_slope_intercept(
    y_true: Iterable[int], y_prob: Iterable[float]
) -> tuple[float | None, float | None, str | None]:
    """Fit y ~ intercept + slope * logit(p), returning an honest diagnostic."""
    y, p = _arrays(y_true, y_prob)
    if np.unique(y).size < 2:
        return None, None, "single-class labels"
    try:
        from sklearn.linear_model import LogisticRegression

        logits = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS)).reshape(-1, 1)
        model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000).fit(logits, y)
        return float(model.coef_[0, 0]), float(model.intercept_[0]), None
    except Exception as exc:  # pragma: no cover - backend-specific failure
        return None, None, f"calibration regression failed: {exc}"


def binary_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute the complete preregistered binary metric set.

    The threshold must have been selected on validation data.  This function
    deliberately does not optimize it.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    y, p = _arrays(y_true, y_prob)
    pred = (p >= threshold).astype(np.int8)
    warnings: list[str] = []
    auc: float | None
    auprc: float | None
    if np.unique(y).size < 2:
        auc = None
        auprc = None
        balanced_accuracy = None
        mcc = None
        warnings.append("AUROC, AUPRC, balanced accuracy, and MCC unavailable: labels contain one class")
    else:
        auc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
        balanced_accuracy = float(balanced_accuracy_score(y, pred))
        mcc = float(matthews_corrcoef(y, pred))
    slope, intercept, calibration_warning = calibration_slope_intercept(y, p)
    if calibration_warning:
        warnings.append(calibration_warning)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(y.size),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "auroc": auc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": mcc,
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, n_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "warnings": warnings,
    }


def select_thresholds(y_true: Iterable[int], y_prob: Iterable[float]) -> dict[str, float]:
    """Select decision thresholds on a validation set only."""
    y, p = _arrays(y_true, y_prob)
    if np.unique(y).size < 2:
        raise ValueError("threshold selection requires both classes")
    candidates = np.unique(np.r_[0.0, p, 1.0])
    rows = []
    for threshold in candidates:
        pred = p >= threshold
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        rows.append((float(threshold), f1, recall + specificity - 1, precision, recall))
    arr = np.asarray(rows)
    return {
        "f1_optimal": float(arr[np.argmax(arr[:, 1]), 0]),
        "youden_j": float(arr[np.argmax(arr[:, 2]), 0]),
        "precision_oriented": float(arr[np.argmax(arr[:, 3] + 1e-6 * arr[:, 4]), 0]),
        "recall_oriented": float(arr[np.argmax(arr[:, 4] + 1e-6 * arr[:, 3]), 0]),
    }
