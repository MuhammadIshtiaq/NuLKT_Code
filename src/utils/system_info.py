"""Capture reproducibility-relevant software and hardware details."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_system_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(root),
        "packages": {},
    }
    for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]:
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    try:
        import psutil

        info["ram_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        info["ram_total_bytes"] = None
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "vram_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        info["torch"] = {"installed": False}
    return info


def write_system_info(root: Path, output: Path) -> dict[str, Any]:
    info = collect_system_info(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
