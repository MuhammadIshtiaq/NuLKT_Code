"""Post hoc TEST inference from completed, DEV-selected offline-safe NuLKT fits.

This script never trains, chooses an epoch, or uses TEST labels as model inputs.
TEST labels are read only to verify row identity and subsequently score predictions.
The resulting estimates remain post hoc because the historical TEST set has been
inspected during earlier research; they are not a fresh blind challenge result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_stages as stages  # noqa: E402
from src.analysis.inference import indexed_dataloader, load_proposed_model  # noqa: E402
from src.data.manifest import manifest_parts  # noqa: E402
from src.data.torch_dataset import load_processed_frame, vocab_sizes_for_model  # noqa: E402
from src.models.nulkt import NuLKTConfig  # noqa: E402

SEEDS = (42, 123, 2024)
SOURCE_PATHS = (
    "src/data/torch_dataset.py", "src/models/nulkt.py", "src/models/components.py",
    "scripts/research_stages.py", "src/training/trainer.py",
)
OUTPUT_COLUMNS = ("row_index", "label", "learner_id", "probability")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _check_data(root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = root / "data/processed/en_es/manifest.json"
    manifest = _read_json(manifest_path)
    preflight = _read_json(root / "offline_safe_preflight.json")
    if manifest.get("offline_safe_protocol") != "split-frozen-v1":
        raise RuntimeError("Expected split-frozen-v1 corpus")
    if preflight.get("status") != "PASS" or preflight.get("safe_part_hashes_verified") is not True:
        raise RuntimeError("Offline-safe preflight has not passed")
    if preflight.get("safe_preprocessing_hash") != manifest.get("preprocessing_hash"):
        raise RuntimeError("Preflight and corpus preprocessing hashes differ")
    if preflight.get("split_counts") != manifest.get("official_split_counts"):
        raise RuntimeError("Preflight and manifest split counts differ")
    parts = manifest_parts(root, manifest)
    hashes = manifest.get("offline_safe_part_sha256", {})
    if len(parts) != len(hashes) or not parts:
        raise RuntimeError("Incomplete corpus partition hash inventory")
    for part in parts:
        if not part.is_file() or sha(part) != hashes.get(part.name):
            raise RuntimeError(f"Corpus partition missing or changed: {part}")
    return manifest, sha(manifest_path)


def _check_completed_fit(root: Path, seed: int, manifest_hash: str,
                         manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if seed not in SEEDS:
        raise ValueError(f"Expected one of {SEEDS}, got {seed}")
    run_dir = root / "results/raw_runs/offline_safe_nulkt" / f"seed_{seed}"
    provenance = _read_json(run_dir / "safe_fit_provenance.json")
    if provenance.get("status") != "complete" or provenance.get("seed") != seed:
        raise RuntimeError(f"Seed {seed} fit is not complete")
    if provenance.get("model") != "nulkt" or provenance.get("data_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Seed {seed} fit model or corpus provenance mismatch")
    if provenance.get("model_overrides") != {"exercise_context": True}:
        raise RuntimeError("Unexpected NuLKT fit architecture overrides")
    if provenance.get("blind_dev_response_policy") != (
        "all valid DEV observed_responses=-1; labels retained only as targets"
    ):
        raise RuntimeError("DEV response masking provenance mismatch")
    if provenance.get("test_policy") != "not loaded during fit; post hoc after frozen combiner selection":
        raise RuntimeError("TEST partition policy provenance mismatch")
    if provenance.get("history_policy") != manifest.get("offline_safe_semantics"):
        raise RuntimeError("Frozen-history policy mismatch")
    recorded_sources = {key.replace("\\", "/"): value for key, value in provenance.get("source_sha256", {}).items()}
    if set(recorded_sources) != set(SOURCE_PATHS):
        raise RuntimeError("Incomplete fit source fingerprint")
    for name in SOURCE_PATHS:
        if sha(root / name) != recorded_sources[name]:
            raise RuntimeError(f"Fit source changed after training: {name}")
    dev_path = run_dir / "dev_predictions.csv"
    if not dev_path.is_file() or sha(dev_path) != provenance.get("dev_prediction_sha256"):
        raise RuntimeError("Completed DEV predictions missing or changed")
    dev = pd.read_csv(dev_path)
    if list(dev) != list(OUTPUT_COLUMNS) or len(dev) != manifest["official_split_counts"]["dev"]:
        raise RuntimeError("Completed DEV predictions have wrong columns or row count")
    if (dev.row_index.duplicated().any() or not np.isfinite(dev.probability).all()
            or not dev.probability.between(0, 1).all()):
        raise RuntimeError("Completed DEV predictions have duplicate rows or invalid probabilities")
    if len(dev) != provenance.get("dev_prediction_rows"):
        raise RuntimeError("Completed DEV prediction provenance count mismatch")
    checkpoint_path = run_dir / "checkpoints" / f"best_seed_{seed}.pt"
    history_path = run_dir / "training_history.json"
    if not checkpoint_path.is_file() or not history_path.is_file():
        raise RuntimeError("Completed fit lacks best checkpoint or training history")
    return checkpoint_path, provenance


def _check_checkpoint(root: Path, checkpoint: dict[str, Any], seed: int,
                      history_path: Path) -> None:
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 1:
        raise RuntimeError("Checkpoint has no selected epoch")
    if not isinstance(checkpoint.get("model_state"), dict):
        raise RuntimeError("Checkpoint has no model state")
    extra = checkpoint.get("extra") or {}
    recorded = extra.get("model_config")
    if not isinstance(recorded, dict):
        raise RuntimeError("Checkpoint lacks architecture configuration")
    expected = NuLKTConfig(
        vocab_sizes=vocab_sizes_for_model(root),
        **stages._model_settings("nulkt", 200, {"exercise_context": True}),
    ).to_dict()
    if recorded != expected:
        raise RuntimeError("Checkpoint model configuration differs from the completed fit")
    trainer = extra.get("trainer_config") or {}
    if trainer.get("checkpoint_name") != f"best_seed_{seed}":
        raise RuntimeError("Checkpoint name/seed mismatch")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise RuntimeError("Completed fit has empty training history")
    selected = min(history, key=lambda row: row["validation"]["loss"])
    if (selected["epoch"] != checkpoint["epoch"]
            or not np.isclose(selected["validation"]["loss"],
                              checkpoint.get("metrics", {}).get("loss", np.nan), rtol=0, atol=1e-8)):
        raise RuntimeError("Checkpoint does not match the DEV-selected best epoch")


@torch.inference_mode()
def predict_masked(model: torch.nn.Module, batches: Iterable[dict[str, torch.Tensor]],
                   device: torch.device) -> pd.DataFrame:
    """Infer only from batches whose scored and context responses are withheld."""
    model.eval()
    records: list[pd.DataFrame] = []
    for batch in batches:
        valid = ~batch["padding_mask"]
        if not torch.all(batch["observed_responses"][valid] == -1):
            raise RuntimeError("Held-out response appeared in a neural input")
        if torch.any(batch["score_mask"] & ~valid):
            raise RuntimeError("Score mask includes a padded position")
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(moved, return_diagnostics=False)
        selected = valid.to(device) & moved["score_mask"].bool()
        records.append(pd.DataFrame({
            "row_index": moved["row_index"][selected].cpu().numpy(),
            "label": moved["responses"][selected].cpu().numpy(),
            "learner_id": moved["learner"][selected].cpu().numpy(),
            "probability": torch.sigmoid(output["logits"])[selected].float().cpu().numpy(),
        }))
    if not records:
        raise RuntimeError("No held-out inference batches")
    return pd.concat(records, ignore_index=True)


def check_row_identity(predictions: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    if list(predictions) != list(OUTPUT_COLUMNS):
        raise RuntimeError("Prediction columns changed")
    if len(predictions) != len(expected) or predictions.row_index.duplicated().any():
        raise RuntimeError("TEST predictions do not cover each row exactly once")
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0, 1).all():
        raise RuntimeError("TEST probabilities are invalid")
    actual = predictions.sort_values("row_index", kind="stable").reset_index(drop=True)
    truth = expected.sort_values("row_index", kind="stable").reset_index(drop=True)
    if actual.row_index.to_numpy().tolist() != truth.row_index.to_numpy().tolist():
        raise RuntimeError("TEST prediction row indices differ from corpus rows")
    for column in ("label", "learner_id"):
        if not np.array_equal(actual[column].to_numpy(), truth[column].to_numpy()):
            raise RuntimeError(f"TEST prediction {column} identity differs from corpus")
    return actual


def global_row_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve corpus row IDs when a held-out partition is read by itself."""
    if "row_order" not in frame or frame.row_order.isna().any() or frame.row_order.duplicated().any():
        raise RuntimeError("Held-out frame lacks unique global row_order")
    return frame.set_index("row_order", drop=False).rename_axis(index=None)


def infer_seed(root: Path, seed: int, *, device_name: str | None = None) -> dict[str, Any]:
    manifest, manifest_hash = _check_data(root)
    selection_path = root / "results/offline_safe_evaluation/benchmark/dev_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Freeze the complete benchmark DEV combiner and threshold before TEST inference")
    selection = _read_json(selection_path)
    if (selection.get("status") != "selected" or selection.get("regime") != "benchmark"
            or selection.get("data_manifest_sha256") != manifest_hash):
        raise RuntimeError("Frozen benchmark DEV selection does not match the safe corpus")
    checkpoint_path, fit = _check_completed_fit(root, seed, manifest_hash, manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_dir = root / "results/raw_runs/offline_safe_nulkt" / f"seed_{seed}"
    _check_checkpoint(root, checkpoint, seed, run_dir / "training_history.json")
    model, device, loader_meta = load_proposed_model(
        root, seed=seed, tag="offline_safe_nulkt", sequence_length=200, device=device_name
    )
    if loader_meta.get("architecture_config_source") != "checkpoint":
        raise RuntimeError("NuLKT architecture was not restored from checkpoint")
    frame, _ = load_processed_frame(
        root, derive_context=False, split_filter=("official_split", {"test"}),
    )
    if len(frame) != manifest["official_split_counts"]["test"]:
        raise RuntimeError("TEST corpus row count differs from manifest")
    if frame.row_order.duplicated().any():
        raise RuntimeError("TEST corpus contains duplicate row_order")
    expected = frame[["row_order", "label", "user_id_id"]].rename(columns={
        "row_order": "row_index", "user_id_id": "learner_id",
    })
    # The indexed sequence loader emits its DataFrame index as row_index.  A
    # TEST-only Parquet read has a split-local RangeIndex, so explicitly retain
    # the corpus-global row_order before constructing any sequence windows.
    frame = global_row_index(frame)
    loader, _ = indexed_dataloader(
        root, split="test", protocol="official_split", frame=frame,
        sequence_length=200, batch_size=64, context_length=0,
    )
    prediction = check_row_identity(predict_masked(model, loader, device), expected)
    output = run_dir / "test_predictions_posthoc.csv"
    provenance_path = run_dir / "safe_test_inference_provenance.json"
    if output.exists() or provenance_path.exists():
        if not output.exists() or not provenance_path.exists():
            raise RuntimeError("Incomplete prior TEST inference artifact; inspect it before rerunning")
        prior = _read_json(provenance_path)
        if sha(output) != prior.get("test_prediction_sha256") or sha(checkpoint_path) != prior.get("checkpoint_sha256"):
            raise RuntimeError("Existing TEST inference artifact has changed")
        previous = pd.read_csv(output)
        if (not np.array_equal(prediction[["row_index", "label", "learner_id"]].to_numpy(),
                               previous[["row_index", "label", "learner_id"]].to_numpy())
                or not np.allclose(prediction.probability, previous.probability, rtol=0, atol=1e-7)):
            raise RuntimeError("Rerun TEST predictions differ from the existing artifact")
        return prior
    _atomic_csv(output, prediction)
    provenance = {
        "status": "complete", "protocol": "split-frozen-v1", "regime": "post_hoc_test",
        "seed": seed, "model": "nulkt", "selection_partition": "DEV only",
        "test_policy": "TEST labels excluded from every neural input; used for post hoc scoring only",
        "prior_test_inspection": True,
        "benchmark_dev_selection_sha256": sha(selection_path),
        "data_manifest_sha256": manifest_hash,
        "fit_provenance_sha256": sha(run_dir / "safe_fit_provenance.json"),
        "dev_prediction_sha256": fit["dev_prediction_sha256"],
        "checkpoint_sha256": sha(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "architecture_config_source": loader_meta["architecture_config_source"],
        "test_prediction_rows": len(prediction),
        "test_prediction_sha256": sha(output),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(provenance_path, json.dumps(provenance, indent=2))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    for seed in SEEDS if args.all else (args.seed,):
        print(json.dumps(infer_seed(ROOT, seed, device_name=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
