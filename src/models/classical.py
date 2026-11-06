"""Classical baseline registry with explicit optional-dependency reporting."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class MajorityClassifier:
    """Empirical class-prior predictor with a scikit-learn-like API."""

    def __init__(self) -> None:
        self.positive_probability_: float | None = None

    def fit(self, features: Any, targets: Any) -> "MajorityClassifier":
        y = np.asarray(targets, dtype=np.float64)
        if y.size == 0:
            raise ValueError("Cannot fit MajorityClassifier on an empty target array")
        self.positive_probability_ = float(y.mean())
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.positive_probability_ is None:
            raise RuntimeError("MajorityClassifier must be fitted first")
        count = len(features)
        positive = np.full(count, self.positive_probability_, dtype=np.float64)
        return np.column_stack((1 - positive, positive))

    def predict(self, features: Any) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(np.int64)


@dataclass(frozen=True)
class BaselineAvailability:
    name: str
    available: bool
    dependency: str
    reason: str


OPTIONAL_DEPENDENCIES = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


def availability() -> list[BaselineAvailability]:
    status = [
        BaselineAvailability("majority", True, "numpy", "built in"),
        BaselineAvailability("logistic_regression", importlib.util.find_spec("sklearn") is not None, "scikit-learn", "required dependency"),
        BaselineAvailability("random_forest", importlib.util.find_spec("sklearn") is not None, "scikit-learn", "required dependency"),
    ]
    for name, module in OPTIONAL_DEPENDENCIES.items():
        installed = importlib.util.find_spec(module) is not None
        status.append(BaselineAvailability(name, installed, module, "installed" if installed else "optional dependency not installed"))
    return status


def log_availability() -> list[BaselineAvailability]:
    statuses = availability()
    for item in statuses:
        log = LOGGER.info if item.available else LOGGER.warning
        log("Classical baseline %-20s available=%s dependency=%s (%s)", item.name, item.available, item.dependency, item.reason)
    return statuses


def build_classical_baseline(name: str, random_state: int = 42, **kwargs: Any) -> Any:
    """Construct a probability classifier without silently substituting models."""

    name = name.lower()
    if name == "majority":
        return MajorityClassifier()
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        defaults = {"n_estimators": 300, "n_jobs": -1, "class_weight": "balanced_subsample"}
        defaults.update(kwargs)
        return RandomForestClassifier(random_state=random_state, **defaults)
    if name == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("xgboost baseline requested but xgboost is not installed")
        from xgboost import XGBClassifier
        defaults = {"n_estimators": 300, "eval_metric": "logloss", "n_jobs": -1}
        defaults.update(kwargs)
        return XGBClassifier(random_state=random_state, **defaults)
    if name == "lightgbm":
        if importlib.util.find_spec("lightgbm") is None:
            raise ImportError("lightgbm baseline requested but lightgbm is not installed")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, **kwargs)
    if name == "catboost":
        if importlib.util.find_spec("catboost") is None:
            raise ImportError("catboost baseline requested but catboost is not installed")
        from catboost import CatBoostClassifier
        defaults = {"verbose": False, "allow_writing_files": False}
        defaults.update(kwargs)
        return CatBoostClassifier(random_seed=random_state, **defaults)
    raise ValueError(f"Unknown classical baseline {name!r}")


def run_classical_baseline(name: str, x_train: Any, y_train: Any, x_eval: Any,
                           random_state: int = 42, **kwargs: Any) -> dict[str, Any]:
    """Fit one baseline and return probabilities plus transparent metadata."""

    model = build_classical_baseline(name, random_state=random_state, **kwargs)
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_eval))[:, 1]
    return {
        "name": name,
        "model": model,
        "probabilities": probabilities,
        "n_train": int(len(y_train)),
        "n_eval": int(len(x_eval)),
        "random_state": random_state,
    }

