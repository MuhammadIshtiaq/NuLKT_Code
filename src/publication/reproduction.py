"""Build a provenance-preserving reproduction bundle from existing artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = [
    "src", "scripts", "configs", "run_all.py", "pyproject.toml", "environment.yml",
    "requirements.txt", "README.md", "RUNBOOK.md", "CITATION.cff", "LICENSE",
    "results/protocol", "results/statistical_tests", "results/metrics",
    "results/benchmark/ensemble_study.json",
    "results/benchmark/prompt_free_control_protocol.json",
    "results/benchmark/prompt_free_control.json",
    "results/data_summary.json", "results/preprocessing_summary.json",
    "results/figures/figure_registry.json", "results/figures/font_report.json",
    "data/metadata", "references", "logs",
]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "unavailable"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unavailable"
    except Exception:
        commit = "unavailable"
    hardware = {"cpu": platform.processor() or "unavailable", "ram_bytes": "unavailable", "cuda_available": False, "gpu": "unavailable"}
    try:
        import psutil

        hardware["ram_bytes"] = int(psutil.virtual_memory().total)
        hardware["logical_cpu_count"] = int(psutil.cpu_count(logical=True) or 0)
    except Exception as exc:
        hardware["psutil_status"] = f"unavailable: {type(exc).__name__}"
    try:
        import torch

        hardware["cuda_available"] = bool(torch.cuda.is_available())
        hardware["pytorch_cuda_build"] = torch.version.cuda
        hardware["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
            hardware["gpu_count"] = torch.cuda.device_count()
            hardware["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        hardware["torch_status"] = f"unavailable: {type(exc).__name__}"
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": hardware,
        "packages": packages,
        "git_commit": commit,
        "reproducibility_note": "Bitwise identity across hardware/software environments is not guaranteed.",
    }


def build_reproduction_bundle(project_root: Path) -> Path:
    results_root = (project_root / "results").resolve()
    bundle = results_root / "reproduction_bundle"
    if bundle.parent != results_root:
        raise RuntimeError(f"Unsafe reproduction-bundle target: {bundle}")
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    for relative in CANDIDATES:
        source = project_root / relative
        if not source.exists():
            copied.append({"source": relative, "status": "unavailable"})
            continue
        target = bundle / relative.replace("results/", "")
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
            for file in target.rglob("*"):
                if file.is_file():
                    copied.append({"source": str(source), "bundle_file": str(file.relative_to(bundle)), "sha256": _hash(file), "status": "available"})
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"source": relative, "bundle_file": str(target.relative_to(bundle)), "sha256": _hash(target), "status": "available"})
    checkpoint_dir = project_root / "checkpoints"
    model_hashes = []
    if checkpoint_dir.exists():
        for checkpoint in checkpoint_dir.rglob("*.pt"):
            model_hashes.append({"file": str(checkpoint.relative_to(project_root)), "sha256": _hash(checkpoint), "size_bytes": checkpoint.stat().st_size})
    (bundle / "model_hashes.json").write_text(
        json.dumps({"status": "available" if model_hashes else "unavailable", "models": model_hashes}, indent=2),
        encoding="utf-8",
    )
    (bundle / "environment.json").write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({"files": copied}, indent=2), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    build_reproduction_bundle(Path.cwd())
