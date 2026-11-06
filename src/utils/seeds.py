"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from its framework-provided seed."""
    try:
        import torch

        worker_seed = torch.initial_seed() % (2**32)
    except ImportError:
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
