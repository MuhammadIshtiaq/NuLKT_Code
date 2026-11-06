"""Atomic, reproducible PyTorch checkpoint management."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class CheckpointManager:
    """Save and restore models, optimizer state, and random-number generators."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float],
        scheduler: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.directory / f"{name}.pt"
        temporary = path.with_suffix(".tmp")
        payload = {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
            "extra": extra or {},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def load(
        self,
        path_or_name: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        path = Path(path_or_name)
        if not path.is_absolute() and path.parent == Path("."):
            path = self.directory / (path.name if path.suffix else f"{path.name}.pt")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if restore_rng and "rng" in checkpoint:
            random.setstate(checkpoint["rng"]["python"])
            np.random.set_state(checkpoint["rng"]["numpy"])
            torch.set_rng_state(checkpoint["rng"]["torch"])
            if torch.cuda.is_available() and checkpoint["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        return checkpoint

