"""Freeze and hash the final experimental protocol before test evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_PROTOCOL = {
    "status": "planned_not_executed",
    "dataset": {"name": "Duolingo SLAM 2018", "doi": "10.7910/DVN/8SWHNO", "tracks": "resolved from official metadata before execution"},
    "research_questions": [f"RQ{i}" for i in range(1, 12)],
    "hypotheses": [f"H{i}" for i in range(1, 9)],
    "protocols": ["official_benchmark_when_compatible", "chronological_70_15_15", "learner_disjoint_cold_start"],
    "primary_metrics": ["AUROC", "AUPRC"],
    "secondary_metrics": ["accuracy", "balanced_accuracy", "precision", "recall", "F1", "MCC", "log_loss", "Brier_score", "ECE"],
    "seeds": [42, 123, 2024, 2025, 2026],
    "alpha": 0.05,
    "multiple_comparison_correction": "Holm-Bonferroni",
    "primary_inference": "paired learner-clustered bootstrap",
    "secondary_tests": ["learner-level Wilcoxon signed-rank", "exact McNemar"],
    "confidence_interval": 0.95,
    "threshold_selection": "validation_only",
    "calibration_fit": "validation_only",
    "test_policy": "one final evaluation after configuration freeze",
    "causal_scope": "predictive decision support only; no causal instruction-effect claim",
}


def freeze_protocol(results_root: Path, protocol: dict | None = None, overwrite: bool = False) -> tuple[Path, str]:
    directory = results_root / "protocol"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "final_experimental_protocol.yaml"
    if path.exists() and not overwrite:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_path = directory / "final_experimental_protocol.yaml.sha256"
        if not hash_path.exists():
            hash_path = directory / "final_experimental_protocol.sha256"
        if not hash_path.exists():
            hash_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
        (directory / "protocol_manifest.json").write_text(
            json.dumps({"file": str(path), "sha256": digest, "status": "preexisting_frozen_protocol"}, indent=2),
            encoding="utf-8",
        )
        return path, digest
    payload = dict(DEFAULT_PROTOCOL if protocol is None else protocol)
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "final_experimental_protocol.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    (directory / "protocol_manifest.json").write_text(json.dumps({"file": str(path), "sha256": digest, "status": payload.get("status")}, indent=2), encoding="utf-8")
    return path, digest


if __name__ == "__main__":
    freeze_protocol(Path("results"))
