"""Optimizers, learning-rate schedules, and weight averaging.

Everything in this module is opt-in. A training-settings mapping that carries
none of the keys documented below reproduces exactly what the published runs
used: AdamW at a flat learning rate, no schedule, no weight averaging. That
matters because the reported numbers were produced under those defaults and
must stay reproducible from the same stage definitions.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

SCHEDULE_NAMES = ("none", "cosine_warmup", "one_cycle", "plateau")
OPTIMIZER_NAMES = ("adam", "adamw", "sgd", "sgd_momentum", "rmsprop", "lion", "nadam", "radam")
STEP_INTERVALS = ("epoch", "step")


class Lion(Optimizer):
    """EvoLved Sign Momentum (Chen et al., 2023) with decoupled weight decay.

    The update is the ``sign`` of an interpolated momentum, so every coordinate
    moves by exactly ``lr``. Useful learning rates are therefore roughly 3-10x
    smaller than AdamW's at a correspondingly larger weight decay, which is why
    the factory never substitutes Lion for AdamW implicitly -- reusing an AdamW
    learning rate here diverges.

    Implemented in-repo rather than pulled from ``lion-pytorch`` because the
    algorithm is fifteen lines and a research artifact should not grow a
    dependency for that.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Lion learning rate must be positive, got {lr}")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Lion betas must lie in [0, 1), got {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Lion weight decay must be non-negative, got {weight_decay}")
        super().__init__(params, {"lr": lr, "betas": tuple(betas), "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: D102 - torch.optim contract
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                if decay:
                    # Decoupled: the decay shrinks the weights directly instead of
                    # entering the gradient, where sign() would erase its magnitude.
                    parameter.mul_(1.0 - lr * decay)
                update = momentum.mul(beta1).add_(gradient, alpha=1.0 - beta1).sign_()
                parameter.add_(update, alpha=-lr)
                momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
        return loss


def build_parameter_groups(
    model: nn.Module,
    weight_decay: float,
    exclude_bias_and_norm: bool = False,
) -> Any:
    """Return optimizer parameter groups, optionally sparing biases and norms.

    Decaying one-dimensional parameters -- biases and LayerNorm gains -- shrinks
    quantities that have no scale-invariance to exploit and measurably hurts
    transformers. Splitting them into a zero-decay group is the standard fix, but
    it changes the objective, so it stays off unless asked for. When off, the raw
    ``model.parameters()`` iterator is handed back untouched so the optimizer sees
    byte-identical arguments to the pre-existing code path.
    """

    if not exclude_bias_and_norm:
        return model.parameters()
    decayed: list[nn.Parameter] = []
    spared: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (spared if parameter.ndim <= 1 or name.endswith(".bias") else decayed).append(parameter)
    return [
        {"params": decayed, "weight_decay": float(weight_decay)},
        {"params": spared, "weight_decay": 0.0},
    ]


def create_optimizer(model: nn.Module, settings: Mapping[str, Any]) -> Optimizer:
    """Build the optimizer named by ``settings['optimizer']`` (default ``adamw``).

    Recognised keys: ``optimizer``, ``learning_rate``, ``weight_decay``,
    ``momentum`` (SGD/RMSprop), ``betas``, ``eps``, ``nesterov``, and
    ``no_decay_on_bias_and_norm``. ``betas``/``eps`` are only forwarded when the
    caller supplies them, so each optimizer keeps its own library defaults.
    """

    name = str(settings.get("optimizer", "adamw")).lower()
    lr = float(settings.get("learning_rate", 3e-4))
    weight_decay = float(settings.get("weight_decay", 0.0))
    parameters = build_parameter_groups(
        model, weight_decay, bool(settings.get("no_decay_on_bias_and_norm", False))
    )
    moments: dict[str, Any] = {}
    if "betas" in settings:
        moments["betas"] = tuple(float(value) for value in settings["betas"])
    if "eps" in settings:
        moments["eps"] = float(settings["eps"])
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **moments)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **moments)
    if name == "nadam":
        # NAdam's decoupled flag makes its weight decay comparable to AdamW's;
        # coupled L2 on a per-parameter-scaled update is a different penalty.
        return torch.optim.NAdam(
            parameters, lr=lr, weight_decay=weight_decay,
            decoupled_weight_decay=bool(settings.get("decoupled_weight_decay", True)), **moments,
        )
    if name == "radam":
        return torch.optim.RAdam(
            parameters, lr=lr, weight_decay=weight_decay,
            decoupled_weight_decay=bool(settings.get("decoupled_weight_decay", True)), **moments,
        )
    if name == "lion":
        return Lion(parameters, lr=lr, weight_decay=weight_decay, **moments)
    if name in {"sgd", "sgd_momentum"}:
        return torch.optim.SGD(
            parameters, lr=lr, momentum=float(settings.get("momentum", 0.9)),
            weight_decay=weight_decay, nesterov=bool(settings.get("nesterov", False)),
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters, lr=lr, weight_decay=weight_decay,
            momentum=float(settings.get("momentum", 0.0)),
            **{key: value for key, value in moments.items() if key == "eps"},
        )
    raise ValueError(f"Unknown optimizer {name!r}")


def optimizer_steps_per_epoch(batches_per_epoch: int, accumulation_steps: int = 1) -> int:
    """Optimizer updates one epoch performs under gradient accumulation.

    The trainer flushes a trailing partial accumulation window at the end of the
    epoch, so the count is a ceiling, not a floor. Getting this wrong silently
    mis-sizes every per-step schedule and makes OneCycleLR overrun its budget.
    """

    if batches_per_epoch <= 0:
        raise ValueError(f"batches_per_epoch must be positive, got {batches_per_epoch}")
    if accumulation_steps <= 0:
        raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}")
    return math.ceil(batches_per_epoch / accumulation_steps)


def cosine_warmup_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    """Multiplier for linear warmup into cosine decay at 0-based ``step``."""

    if warmup_steps > 0 and step < warmup_steps:
        # ``step + 1`` so the very first update trains at a non-zero rate; a
        # wasted all-zero step is a silent waste of the warmup budget.
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_scheduler(
    optimizer: Optimizer,
    settings: Mapping[str, Any],
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
) -> tuple[Any | None, str]:
    """Build the schedule named by ``settings['lr_schedule']`` and say when to step it.

    Returns ``(scheduler, interval)`` where ``interval`` is ``"step"`` (advance
    after every optimizer update) or ``"epoch"``. The two are not
    interchangeable: a per-step schedule advanced once per epoch would finish its
    warmup after the run is over.

    Recognised names and their keys:

    * ``none`` (default) -- no scheduler, the historical flat-rate behaviour.
    * ``cosine_warmup`` -- ``warmup_ratio`` (default 0.05) or an explicit
      ``warmup_steps``, plus ``min_lr_ratio`` (default 0.0) for the cosine floor.
    * ``one_cycle`` -- ``max_lr`` (default ``learning_rate``), ``warmup_ratio``
      as ``pct_start``, ``div_factor``, ``final_div_factor``.
    * ``plateau`` -- ``plateau_factor``, ``plateau_patience``, ``plateau_min_lr``;
      always per-epoch because the monitored score only exists per epoch.

    ``steps_per_epoch`` must be the number of *optimizer* updates per epoch (see
    :func:`optimizer_steps_per_epoch`), not the number of batches.
    """

    name = str(settings.get("lr_schedule", "none") or "none").lower()
    if name in {"none", "constant", "flat"}:
        return None, "epoch"
    if name not in SCHEDULE_NAMES:
        raise ValueError(f"Unknown lr_schedule {name!r}; expected one of {SCHEDULE_NAMES}")

    total_epochs = int(settings.get("epochs", 30) if epochs is None else epochs)
    if total_epochs <= 0:
        raise ValueError(f"lr_schedule {name!r} needs a positive epoch budget, got {total_epochs}")
    default_interval = "epoch" if name == "plateau" else "step"
    interval = str(settings.get("lr_schedule_interval", default_interval)).lower()
    if interval not in STEP_INTERVALS:
        raise ValueError(f"lr_schedule_interval must be one of {STEP_INTERVALS}, got {interval!r}")
    if name == "plateau" and interval != "epoch":
        # The plateau rule consumes the monitored validation score, which the
        # trainer only computes once an epoch; there is nothing to feed it mid-epoch.
        raise ValueError("lr_schedule 'plateau' can only step per epoch")
    if interval == "step" and not steps_per_epoch:
        raise ValueError(
            f"lr_schedule {name!r} steps per optimizer update and needs steps_per_epoch; "
            "pass it explicitly when the training loader has no length"
        )

    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # the trainer always treats a lower monitored score as better
                factor=float(settings.get("plateau_factor", 0.5)),
                patience=int(settings.get("plateau_patience", 2)),
                min_lr=float(settings.get("plateau_min_lr", 0.0)),
            ),
            "epoch",
        )

    total_steps = total_epochs * int(steps_per_epoch) if interval == "step" else total_epochs
    warmup_ratio = float(settings.get("warmup_ratio", 0.05))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must lie in [0, 1), got {warmup_ratio}")

    if name == "one_cycle":
        if total_steps < 3:
            raise ValueError(f"lr_schedule 'one_cycle' needs at least 3 steps, got {total_steps}")
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(settings.get("max_lr", settings.get("learning_rate", 3e-4))),
                total_steps=total_steps,
                # OneCycleLR divides by the length of each phase, so the warmup has
                # to own at least two steps or the schedule divides by zero.
                pct_start=min(max(warmup_ratio, 2.0 / total_steps), 0.9),
                div_factor=float(settings.get("div_factor", 25.0)),
                final_div_factor=float(settings.get("final_div_factor", 1e4)),
            ),
            interval,
        )

    warmup_steps = int(settings.get("warmup_steps", round(warmup_ratio * total_steps)))
    warmup_steps = max(0, min(warmup_steps, total_steps))
    min_ratio = float(settings.get("min_lr_ratio", 0.0))
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must lie in [0, 1], got {min_ratio}")
    return (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_warmup_factor(step, warmup_steps, total_steps, min_ratio),
        ),
        interval,
    )


class ExponentialMovingAverage:
    """Shadow copy of the model's parameters averaged over recent optimizer steps.

    Only parameters are averaged; buffers are copied verbatim from the live model
    so that swapping the average in leaves derived state (masks, counters)
    consistent with the architecture rather than blended across epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must lie in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.steps = 0
        self._parameter_names = {name for name, _ in model.named_parameters()}
        self.shadow = {
            key: value.detach().clone().to(device) if device is not None else value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def current_decay(self) -> float:
        """Decay actually used at this step, ramped in during the warmup window.

        A fixed 0.999 early on keeps the random initialisation in the average for
        thousands of steps. Ramping makes the average track the weights closely at
        first and only then start smoothing.
        """

        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self.steps) / (self.warmup_steps + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.current_decay()
        self.steps += 1
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            source = value.detach().to(shadow.device, dtype=shadow.dtype)
            if key in self._parameter_names and shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                shadow.copy_(source)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write the averaged weights into ``model`` in place."""

        model.load_state_dict({key: value.clone() for key, value in self.shadow.items()})

    def to(self, device: torch.device | str) -> ExponentialMovingAverage:
        self.shadow = {key: value.to(device) for key, value in self.shadow.items()}
        return self

    @contextlib.contextmanager
    def averaged_parameters(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily swap the average into ``model``, restoring the raw weights after."""

        raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield model
        finally:
            model.load_state_dict(raw)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "steps": self.steps,
            "shadow": {key: value.clone() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.steps = int(state["steps"])
        self.shadow = {
            key: value.detach().clone().to(self.shadow[key].device)
            for key, value in state["shadow"].items()
        }


def create_ema(model: nn.Module, settings: Mapping[str, Any]) -> ExponentialMovingAverage | None:
    """Return an EMA tracker when ``settings['ema']`` is true, otherwise ``None``.

    Keys: ``ema`` (default ``False``), ``ema_decay`` (default 0.999),
    ``ema_warmup_steps`` (default 0, meaning no ramp).
    """

    if not bool(settings.get("ema", False)):
        return None
    return ExponentialMovingAverage(
        model,
        decay=float(settings.get("ema_decay", 0.999)),
        warmup_steps=int(settings.get("ema_warmup_steps", 0)),
    )
