"""Validation-only probability calibration and reliability summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import binary_classification_metrics, reliability_bins


@dataclass
class TemperatureScaler:
    temperature: float = 1.0
    fitted: bool = False

    def fit(self, y_validation: Iterable[int], probabilities: Iterable[float]) -> "TemperatureScaler":
        """Fit on validation predictions; callers must not pass test labels."""
        y = np.asarray(list(y_validation), dtype=np.float64)
        p = np.asarray(list(probabilities), dtype=np.float64)
        if y.size == 0 or y.size != p.size or np.unique(y).size < 2:
            raise ValueError("calibration requires equal-length validation arrays with both classes")
        eps = np.finfo(float).eps
        logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))

        def nll(log_temperature: float) -> float:
            temperature = np.exp(log_temperature)
            scaled = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
            return float(-np.mean(y * np.log(np.clip(scaled, eps, 1)) + (1 - y) * np.log(np.clip(1 - scaled, eps, 1))))

        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"temperature optimization failed: {result.message}")
        self.temperature = float(np.exp(result.x))
        self.fitted = True
        return self

    def transform(self, probabilities: Iterable[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fitted on validation data first")
        p = np.asarray(list(probabilities), dtype=np.float64)
        eps = np.finfo(float).eps
        logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
        return 1.0 / (1.0 + np.exp(-np.clip(logits / self.temperature, -40, 40)))


def calibration_comparison(y_true: Iterable[int], before: Iterable[float], after: Iterable[float], n_bins: int = 15) -> dict:
    y = list(y_true)
    before_values = list(before)
    after_values = list(after)
    pre = binary_classification_metrics(y, before_values, n_bins=n_bins)
    post = binary_classification_metrics(y, after_values, n_bins=n_bins)
    return {
        "pre": {k: pre[k] for k in ("brier_score", "ece", "calibration_slope", "calibration_intercept")},
        "post": {k: post[k] for k in ("brier_score", "ece", "calibration_slope", "calibration_intercept")},
        "reliability_pre": [item.__dict__ for item in reliability_bins(y, before_values, n_bins)],
        "reliability_post": [item.__dict__ for item in reliability_bins(y, after_values, n_bins)],
    }
