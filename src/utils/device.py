"""Explicit compute-device selection and reproducibility reporting."""

from __future__ import annotations

import os
from typing import Any

import torch


def select_device(requested: str | torch.device | None = None) -> torch.device:
    """Select CUDA when available, while preventing silent requested-device fallback."""

    configured = str(
        requested or os.environ.get("NULKT_DEVICE") or os.environ.get("EIP_DEVICE", "")
    ).strip().lower()
    require_cuda = (
        os.environ.get("NULKT_REQUIRE_CUDA")
        or os.environ.get("EIP_REQUIRE_CUDA", "0")
    ).strip().lower() in {"1", "true", "yes"}
    if configured:
        device = torch.device(configured)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but this Python has no CUDA-enabled PyTorch. "
                "Install a CUDA-enabled PyTorch build or request CPU explicitly."
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("NULKT_REQUIRE_CUDA is enabled, but CUDA was not selected")
    return device


def device_report() -> dict[str, Any]:
    """Return machine-readable runtime and accelerator evidence."""

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(select_device()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return report
