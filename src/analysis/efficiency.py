"""Complexity and efficiency measurement for the proposed model and baselines.

Table 13 of the publication contract has never had a source. A personalized
tutoring system has to run inside a classroom's latency and memory budget, so
parameter count, peak memory, and per-interaction inference latency are reported
alongside accuracy rather than treated as an afterthought.

Latency is measured with warm-up iterations discarded and, on CUDA, with explicit
synchronization, because an unsynchronized timing loop measures queue submission
rather than computation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.torch_dataset import vocab_sizes_for_model
from src.models.factory import create_model


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    embedding = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
    )
    return {
        "parameters_total": int(total),
        "parameters_trainable": int(trainable),
        "parameters_embedding": int(embedding),
        "parameters_non_embedding": int(total - embedding),
    }


def _synthetic_batch(
    vocab_sizes: dict[str, int],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Shape-correct inputs for timing; values are irrelevant to cost."""

    generator = torch.Generator(device="cpu").manual_seed(0)

    def ids(name: str) -> torch.Tensor:
        high = max(2, min(vocab_sizes.get(name, 2), 1000))
        return torch.randint(1, high, (batch_size, sequence_length), generator=generator).to(device)

    batch = {name: ids(name) for name in ("token", "lemma", "pos", "morph", "context", "exercise", "format", "learner")}
    batch["responses"] = torch.randint(0, 2, (batch_size, sequence_length), generator=generator).to(device)
    elapsed = torch.rand(batch_size, sequence_length, generator=generator).to(device) * 86_400
    batch["elapsed_seconds"] = elapsed
    batch["timestamps"] = torch.cumsum(elapsed, dim=1)
    batch["session"] = torch.randint(1, 5, (batch_size, sequence_length), generator=generator).to(device)
    batch["history_features"] = torch.rand(batch_size, sequence_length, 6, generator=generator).to(device)
    batch["padding_mask"] = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
    return batch


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, float]:
    model.eval()
    accepts_diagnostics = "return_diagnostics" in model.forward.__code__.co_varnames

    def _forward() -> None:
        model(batch, return_diagnostics=False) if accepts_diagnostics else model(batch)

    for _ in range(warmup):
        _forward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    series = pd.Series(durations)
    interactions = batch["responses"].numel()
    return {
        "batch_latency_ms_mean": float(series.mean() * 1000),
        "batch_latency_ms_median": float(series.median() * 1000),
        "batch_latency_ms_sd": float(series.std(ddof=1) * 1000) if len(series) > 1 else 0.0,
        "latency_us_per_interaction": float(series.median() * 1e6 / interactions),
        "interactions_per_second": float(interactions / series.median()),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else float("nan"),
    }


def run(
    root: Path,
    *,
    models: tuple[str, ...] = (
        "nulkt", "eip_ktformer", "mlp", "lstm", "gru", "deepfm", "dkt", "dkvmn",
        "sakt", "akt", "saint_plus", "hierarchical_transformer",
    ),
    sequence_length: int = 200,
    batch_size: int = 64,
    device: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Measure every model's size and inference cost under identical conditions."""

    from scripts.research_stages import _model_settings

    resolved = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_sizes = vocab_sizes_for_model(root)
    batch = _synthetic_batch(vocab_sizes, batch_size, sequence_length, resolved)

    rows: list[dict[str, Any]] = []
    for name in models:
        try:
            model = create_model(name, vocab_sizes, _model_settings(name, sequence_length)).to(resolved)
            record = {"model": name, "status": "available", **count_parameters(model)}
            record.update(measure_latency(model, batch, resolved))
            del model
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            record = {"model": name, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(record)
        print(f"efficiency={name} status={record['status']}", flush=True)

    frame = pd.DataFrame(rows)
    destination = root / "results" / "metrics" / "efficiency_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

    payload = {
        "device": str(resolved),
        "device_name": torch.cuda.get_device_name(0) if resolved.type == "cuda" else "cpu",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "interactions_per_batch": batch_size * sequence_length,
        "measurement_note": (
            "Latency uses discarded warm-up iterations and CUDA synchronization. Values are "
            "hardware- and build-specific and are not comparable across machines."
        ),
        "models": rows,
    }
    (root / "results" / "metrics" / "efficiency_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
