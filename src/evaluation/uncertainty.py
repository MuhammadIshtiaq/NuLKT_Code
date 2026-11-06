"""Predictive uncertainty summaries and uncertainty-aware abstention."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_entropy(probability: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(probability), dtype=float)
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("probabilities must be finite and in [0, 1]")
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mc_dropout_summary(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize stochastic probabilities shaped [passes, observations]."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("samples must have shape [at least 2 passes, observations]")
    if np.any((values < 0) | (values > 1)) or not np.isfinite(values).all():
        raise ValueError("all samples must be probabilities")
    mean = values.mean(axis=0)
    predictive_entropy = binary_entropy(mean)
    expected_entropy = binary_entropy(values.ravel()).reshape(values.shape).mean(axis=0)
    return {
        "probability": mean,
        "standard_deviation": values.std(axis=0, ddof=1),
        "predictive_entropy": predictive_entropy,
        "mutual_information": np.maximum(predictive_entropy - expected_entropy, 0.0),
        "confidence": 1.0 - predictive_entropy,
    }


def abstention_flags(uncertainty: Iterable[float], threshold: float) -> np.ndarray:
    values = np.asarray(list(uncertainty), dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("uncertainty and threshold must be finite")
    return values >= threshold


def risk_coverage_curve(y_true: Iterable[int], y_prob: Iterable[float], uncertainty: Iterable[float]) -> dict[str, list[float]]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_prob), dtype=float)
    u = np.asarray(list(uncertainty), dtype=float)
    if not (len(y) == len(p) == len(u)) or len(y) == 0:
        raise ValueError("inputs must be non-empty and equal length")
    order = np.argsort(u)
    errors = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    retained = np.arange(1, len(y) + 1)
    return {"coverage": (retained / len(y)).tolist(), "selective_risk": (np.cumsum(errors) / retained).tolist()}

