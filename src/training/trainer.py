"""Device-agnostic trainer with early stopping and resumable checkpoints."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.components import beta_evidential_loss
from src.utils.device import select_device
from .checkpointing import CheckpointManager
from .optimization import STEP_INTERVALS, ExponentialMovingAverage

LOGGER = logging.getLogger(__name__)


class _Progress:
    """Live batch progress that behaves sensibly whether or not stderr is a tty.

    Interactive runs get a tqdm bar. Redirected runs get a periodic one-line
    update, because a bar's carriage returns render a log file unreadable, and an
    unattended multi-hour run is exactly when the log matters most.
    """

    def __init__(self, iterable: Iterable[Any], description: str, interval_seconds: float = 30.0) -> None:
        self.description = description
        self.interval = interval_seconds
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.started = time.time()
        self.last_report = self.started
        self.bar = None
        self.iterator: Iterable[Any] = iterable
        interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
        if interactive:
            try:
                from tqdm.auto import tqdm

                self.bar = tqdm(iterable, desc=description, total=self.total, leave=False,
                                dynamic_ncols=True, unit="batch")
                self.iterator = self.bar
            except Exception:  # tqdm is optional; fall back to line reporting
                self.bar = None

    def update(self, completed: int, mean_loss: float) -> None:
        if self.bar is not None:
            self.bar.set_postfix(loss=f"{mean_loss:.4f}", refresh=False)
            return
        now = time.time()
        if now - self.last_report < self.interval:
            return
        self.last_report = now
        elapsed = now - self.started
        rate = completed / elapsed if elapsed > 0 else 0.0
        if self.total:
            share = 100.0 * completed / self.total
            remaining = (self.total - completed) / rate if rate > 0 else float("nan")
            print(
                f"  {self.description}: {completed}/{self.total} ({share:.1f}%) "
                f"loss={mean_loss:.4f} {rate:.1f} batch/s eta={remaining / 60:.1f} min",
                flush=True,
            )
        else:
            print(f"  {self.description}: {completed} batches loss={mean_loss:.4f}", flush=True)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _make_progress(iterable: Iterable[Any], description: str) -> _Progress:
    return _Progress(iterable, description)


@dataclass
class TrainerConfig:
    epochs: int = 30
    patience: int = 5
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    loss: str = "bce"
    positive_weight: float | None = None
    focal_gamma: float = 2.0
    prediction_weight: float = 1.0
    contrastive_weight: float = 0.1
    mastery_weight: float = 0.1
    evidential_weight: float = 0.1
    calibration_weight: float = 0.01
    routing_entropy_weight: float = 0.001
    checkpoint_name: str = "best"
    monitor: str = "loss"
    monitor_mode: str = "min"


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _clone_state(model: nn.Module) -> dict[str, Tensor]:
    # ``state_dict`` hands back live references; a snapshot that is meant to
    # survive a swap has to own its storage.
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


class Trainer:
    """Train any model returning at least ``logits`` in an output dictionary."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | torch.device | None = None,
        scheduler: Any = None,
        scheduler_interval: str = "epoch",
        ema: ExponentialMovingAverage | None = None,
    ) -> None:
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.config = config
        if config.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")
        self.scheduler = scheduler
        if scheduler_interval not in STEP_INTERVALS:
            raise ValueError(f"scheduler_interval must be one of {STEP_INTERVALS}, got {scheduler_interval!r}")
        # Per-epoch is the historical behaviour and stays the default; cosine and
        # one-cycle schedules are sized in optimizer updates and must say so.
        self.scheduler_interval = scheduler_interval
        self.ema = ema.to(self.device) if ema is not None else None
        self.checkpoints = CheckpointManager(checkpoint_dir)
        self.amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        LOGGER.info(
            "trainer device=%s mixed_precision=%s torch=%s",
            self.device,
            self.amp_enabled,
            torch.__version__,
        )

    def _prediction_loss(self, logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
        targets = targets.float()
        if self.config.loss == "bce":
            pos_weight = None if self.config.positive_weight is None else logits.new_tensor(self.config.positive_weight)
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        elif self.config.loss == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probability = torch.sigmoid(logits)
            p_t = probability * targets + (1 - probability) * (1 - targets)
            loss = (1 - p_t).pow(self.config.focal_gamma) * bce
        else:
            raise ValueError(f"Unknown prediction loss: {self.config.loss}")
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    def compute_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        valid = ~batch.get("padding_mask", torch.zeros_like(batch["responses"], dtype=torch.bool))
        # Positions carried only as attention context are attended to but must not
        # contribute to the loss, or overlapping windows would weight the earlier
        # part of each learner's trajectory more heavily than the rest.
        if "score_mask" in batch:
            valid = valid & batch["score_mask"].bool()
        prediction = self._prediction_loss(outputs["logits"], batch["responses"], valid)
        total = self.config.prediction_weight * prediction
        pieces = {"prediction": float(prediction.detach())}
        if hasattr(self.model, "contrastive_loss") and self.config.contrastive_weight:
            contrastive = self.model.contrastive_loss(outputs, batch)
            total = total + self.config.contrastive_weight * contrastive
            pieces["contrastive"] = float(contrastive.detach())
        if "mastery" in outputs and self.config.mastery_weight:
            mastery_target = 1.0 - batch["responses"].float()
            # Probability-space BCE is intentionally evaluated in FP32. PyTorch
            # rejects BCELoss under CUDA autocast because FP16 gradients through
            # sigmoid probabilities can overflow; the main prediction head uses
            # the autocast-safe BCEWithLogits formulation above.
            with torch.autocast(device_type=self.device.type, enabled=False):
                mastery = F.binary_cross_entropy(
                    outputs["mastery"].float(), mastery_target.float(), reduction="none"
                )
                mastery = (mastery * valid).sum() / valid.sum().clamp_min(1)
            total = total + self.config.mastery_weight * mastery
            pieces["mastery"] = float(mastery.detach())
        if "alpha" in outputs and self.config.evidential_weight:
            evidential = beta_evidential_loss(outputs["alpha"], outputs["beta"], batch["responses"], valid)
            total = total + self.config.evidential_weight * evidential
            pieces["evidential"] = float(evidential.detach())
        if self.config.calibration_weight:
            brier = ((torch.sigmoid(outputs["logits"]) - batch["responses"].float()).pow(2) * valid).sum() / valid.sum().clamp_min(1)
            total = total + self.config.calibration_weight * brier
            pieces["brier"] = float(brier.detach())
        if "routing_entropy" in outputs and self.config.routing_entropy_weight:
            # Negative entropy discourages premature routing collapse.
            routing = -(outputs["routing_entropy"] * valid).sum() / valid.sum().clamp_min(1)
            total = total + self.config.routing_entropy_weight * routing
            pieces["routing_regularizer"] = float(routing.detach())
        pieces["loss"] = float(total.detach())
        return total, pieces

    def _optimizer_step(self) -> None:
        """Apply one accumulated update, then advance whatever tracks updates."""

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # A GradScaler that saw non-finite gradients skips the update and backs the
        # scale off. Advancing the schedule or the weight average on a step the
        # parameters never took would desynchronise both from the actual training.
        if self.scaler.get_scale() < scale_before:
            return
        if self.ema is not None:
            self.ema.update(self.model)
        if self.scheduler is not None and self.scheduler_interval == "step":
            self.scheduler.step()

    def run_epoch(
        self,
        loader: Iterable[dict[str, Tensor]],
        training: bool,
        *,
        description: str = "",
    ) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        validation_labels: list[Tensor] = []
        validation_probabilities: list[Tensor] = []
        batches = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad if training else torch.no_grad
        # An interactive terminal gets a live bar; a redirected log gets periodic
        # one-line updates instead, because a bar's carriage returns turn a log
        # file into thousands of unreadable partial lines.
        progress = _make_progress(loader, description or ("train" if training else "validate"))
        with context():
            for index, raw_batch in enumerate(progress.iterator):
                batch = _to_device(raw_batch, self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    outputs = self.model(batch, return_diagnostics=False) if "return_diagnostics" in self.model.forward.__code__.co_varnames else self.model(batch)
                    loss, pieces = self.compute_loss(outputs, batch)
                    scaled_loss = loss / self.config.accumulation_steps
                if training:
                    self.scaler.scale(scaled_loss).backward()
                    if (index + 1) % self.config.accumulation_steps == 0:
                        self._optimizer_step()
                for key, value in pieces.items():
                    totals[key] = totals.get(key, 0.0) + value
                batches += 1
                if not training:
                    valid = ~batch.get(
                        "padding_mask", torch.zeros_like(batch["responses"], dtype=torch.bool)
                    )
                    if "score_mask" in batch:
                        valid = valid & batch["score_mask"].bool()
                    validation_labels.append(batch["responses"][valid].detach().cpu())
                    validation_probabilities.append(
                        torch.sigmoid(outputs["logits"])[valid].detach().float().cpu()
                    )
                progress.update(index + 1, totals["loss"] / batches)
        progress.close()
        if training and batches % self.config.accumulation_steps:
            self._optimizer_step()
        if not batches:
            raise ValueError("Data loader produced no batches")
        metrics = {key: value / batches for key, value in totals.items()}
        if not training and validation_labels:
            from sklearn.metrics import average_precision_score, roc_auc_score

            labels = torch.cat(validation_labels).numpy()
            probabilities = torch.cat(validation_probabilities).numpy()
            if len(set(labels.tolist())) > 1:
                metrics["auroc"] = float(roc_auc_score(labels, probabilities))
                metrics["auprc"] = float(average_precision_score(labels, probabilities))
        return metrics

    def fit(self, train_loader: Iterable[dict[str, Tensor]], validation_loader: Iterable[dict[str, Tensor]],
            history_path: str | Path | None = None) -> list[dict[str, Any]]:
        best = float("inf") if self.config.monitor_mode == "min" else -float("inf")
        stale = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, self.config.epochs + 1):
            label = f"epoch {epoch}/{self.config.epochs}"
            train_metrics = self.run_epoch(train_loader, training=True, description=f"{label} train")
            # With EMA enabled the averaged weights are what downstream code loads
            # from the checkpoint and predicts with, so they -- not the raw weights
            # -- are what gets validated, early-stopped on, and saved. Scoring the
            # raw weights while shipping the averaged ones would select the epoch on
            # a proxy for the model that never runs at inference.
            raw_state = _clone_state(self.model) if self.ema is not None else None
            if self.ema is not None:
                self.ema.copy_to(self.model)
            try:
                validation_metrics = self.run_epoch(validation_loader, training=False, description=f"{label} validate")
                record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
                history.append(record)
                score = validation_metrics[self.config.monitor]
                improved = score < best if self.config.monitor_mode == "min" else score > best
                LOGGER.info("epoch=%d train_loss=%.6f val_%s=%.6f", epoch, train_metrics["loss"], self.config.monitor, score)
                print(
                    f"  epoch {epoch}/{self.config.epochs} train_loss={train_metrics['loss']:.6f} "
                    f"val_{self.config.monitor}={score:.6f}"
                    + ("  <- best" if improved else f"  (no improvement, {stale + 1}/{self.config.patience})"),
                    flush=True,
                )
                if improved:
                    best, stale = score, 0
                    # The architecture configuration travels with the weights. Loading
                    # a checkpoint under different structural settings (for example a
                    # different attention-masking mode) silently changes what the model
                    # computes without any shape mismatch to catch it.
                    model_config = getattr(self.model, "config", None)
                    extra: dict[str, Any] = {"trainer_config": asdict(self.config)}
                    if model_config is not None and hasattr(model_config, "to_dict"):
                        extra["model_config"] = model_config.to_dict()
                    if self.ema is not None and raw_state is not None:
                        # ``model_state`` now holds the average; a resumed run has to
                        # keep training the weights that produced it, not the average.
                        ema_state = {k: v for k, v in self.ema.state_dict().items() if k != "shadow"}
                        extra["ema"] = {**ema_state, "raw_model_state": raw_state}
                    self.checkpoints.save(self.config.checkpoint_name, self.model, self.optimizer, epoch,
                                          validation_metrics, self.scheduler, extra)
                else:
                    stale += 1
            finally:
                if raw_state is not None:
                    self.model.load_state_dict(raw_state)
            # Per-step schedules already advanced inside the training epoch; stepping
            # them again here would run them through their horizon several times over.
            if self.scheduler is not None and self.scheduler_interval == "epoch":
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()
            if stale >= self.config.patience:
                LOGGER.info("Early stopping after epoch %d", epoch)
                break
        if history_path is not None:
            path = Path(history_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    def resume(self, checkpoint: str | Path) -> int:
        state = self.checkpoints.load(checkpoint, self.model, self.optimizer, self.scheduler, self.device)
        stored_ema = (state.get("extra") or {}).get("ema")
        if self.ema is not None and stored_ema is not None:
            # The saved ``model_state`` is the average itself, so it doubles as the
            # shadow and only the raw weights need restoring separately.
            self.ema.load_state_dict({**{k: v for k, v in stored_ema.items() if k != "raw_model_state"},
                                      "shadow": state["model_state"]})
            self.model.load_state_dict(stored_ema["raw_model_state"])
        return int(state["epoch"]) + 1
