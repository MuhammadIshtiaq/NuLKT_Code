"""Thin public experiment entry point for the end-to-end orchestrator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.models.factory import create_model
from .optimization import (
    create_ema,
    create_optimizer,
    create_scheduler,
    optimizer_steps_per_epoch,
)
from .trainer import Trainer, TrainerConfig

__all__ = ["create_optimizer", "run_neural_experiment"]


def _steps_per_epoch(loader: Iterable[dict[str, Tensor]], settings: Mapping[str, Any]) -> int | None:
    """Optimizer updates one epoch performs, or ``None`` when the loader has no length.

    Per-step schedules need this to size themselves; an iterable-style loader
    cannot supply it, so the caller must pass ``steps_per_epoch`` and the
    scheduler factory raises rather than guessing a horizon.
    """

    batches = settings.get("steps_per_epoch")
    if batches is None:
        try:
            batches = len(loader)  # type: ignore[arg-type]
        except TypeError:
            return None
    if not batches:
        return None
    return optimizer_steps_per_epoch(int(batches), int(settings.get("accumulation_steps", 1)))


def run_neural_experiment(
    model_name: str,
    vocab_sizes: Mapping[str, int],
    train_loader: Iterable[dict[str, Tensor]],
    validation_loader: Iterable[dict[str, Tensor]],
    model_settings: Mapping[str, Any],
    training_settings: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Build, train, and return paths/history without touching final-test labels.

    ``training_settings`` additionally accepts the opt-in optimization keys
    documented in :mod:`src.training.optimization` (``lr_schedule``, ``ema``, the
    extra optimizer names). Omitting all of them reproduces the flat-learning-rate
    AdamW run the published results were produced with.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = create_model(model_name, vocab_sizes, model_settings)
    optimizer = create_optimizer(model, training_settings)
    trainer_keys = set(TrainerConfig.__dataclass_fields__)
    trainer_config = TrainerConfig(**{k: v for k, v in training_settings.items() if k in trainer_keys})
    scheduler, interval = create_scheduler(
        optimizer,
        training_settings,
        steps_per_epoch=_steps_per_epoch(train_loader, training_settings),
        epochs=trainer_config.epochs,
    )
    trainer = Trainer(
        model, optimizer, trainer_config, output / "checkpoints", device=device,
        scheduler=scheduler, scheduler_interval=interval, ema=create_ema(model, training_settings),
    )
    history_path = output / "training_history.json"
    history = trainer.fit(train_loader, validation_loader, history_path)
    checkpoint_path = output / "checkpoints" / f"{trainer_config.checkpoint_name}.pt"
    # All downstream validation/test predictions must use the checkpoint selected
    # by validation loss, not the possibly degraded final early-stopping epoch.
    best_checkpoint = trainer.checkpoints.load(
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location=trainer.device,
        restore_rng=False,
    )
    return {
        "model": model,
        "trainer": trainer,
        "history": history,
        "history_path": history_path,
        "checkpoint_path": checkpoint_path,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_metrics": best_checkpoint["metrics"],
    }
