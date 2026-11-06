"""Numerically stable temporal features derived without outcomes."""

from __future__ import annotations

import numpy as np


def log_elapsed_seconds(seconds: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(seconds, dtype=np.float64)
    if (values < 0).any():
        raise ValueError("Elapsed time cannot be negative")
    return np.log1p(values).astype(np.float32)


def fixed_exponential_decay(
    seconds: np.ndarray | list[float], *, half_life_seconds: float
) -> np.ndarray:
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    values = np.asarray(seconds, dtype=np.float64)
    if (values < 0).any():
        raise ValueError("Elapsed time cannot be negative")
    return np.exp2(-values / half_life_seconds).astype(np.float32)
