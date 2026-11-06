"""Model-size and inference-cost utilities for fair comparisons."""

from __future__ import annotations

import time
import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad or not trainable_only)


def parameter_match_ratio(candidate: nn.Module, reference: nn.Module) -> float:
    reference_count = count_parameters(reference)
    if reference_count == 0:
        raise ValueError("Reference model has no trainable parameters")
    return count_parameters(candidate) / reference_count


def benchmark_inference(model: nn.Module, batch: dict[str, Tensor], warmup: int = 5, repeats: int = 20) -> dict[str, float]:
    """Measure batch latency and throughput on the model's current device."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    model.eval()
    device = next(model.parameters()).device
    moved = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    with torch.inference_mode():
        for _ in range(warmup):
            model(moved)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(moved)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
    examples = int(next(iter(moved.values())).shape[0]) * repeats
    return {
        "parameters": float(count_parameters(model)),
        "mean_batch_latency_ms": seconds * 1000 / repeats,
        "examples_per_second": examples / seconds,
    }
