"""Validation-only temperature scaling for neural probabilities."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Single positive temperature fitted on held-out validation logits."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(self(logits))

    def fit(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None, max_iter: int = 50) -> float:
        """Fit on validation data only; callers must never pass final-test labels."""

        logits, targets = logits.detach(), targets.detach().float()
        if mask is not None:
            logits, targets = logits[mask.bool()], targets[mask.bool()]
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach())

