"""Executable validation-driven research stages used by ``run_all.py``.

The final test partition is touched only by ``run_proposed`` after the frozen
configuration has been selected. Search, optimizer, and ablation stages use
validation data exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.quick_experiment import CLASSICAL_FEATURES
from src.data.torch_dataset import load_processed_frame, prepare_dataloaders
from src.evaluation.calibration import TemperatureScaler, calibration_comparison
from src.evaluation.metrics import binary_classification_metrics
from src.models import create_model, run_classical_baseline
from src.training import CheckpointManager, run_neural_experiment
from src.utils.device import select_device
from src.utils.seeds import seed_everything


NEURAL_BASELINES = [
    "mlp",
    "lstm",
    "gru",
    "deepfm",
    "dkt",
    "dkvmn",
    "sakt",
    "akt",
    "saint_plus",
    "hierarchical_transformer",
]
CLASSICAL_BASELINES = [
    "majority",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]
MODEL_ABLATION_KEYS = {
    "personalization",
    "linguistic",
    "temporal",
    "hierarchical",
    "moe",
    "contrastive",
    "context",
    "trainable_temporal_decay",
}


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "data" / "processed" / "en_es" / "manifest.json").read_text(encoding="utf-8"))


def _require_publication_corpus(root: Path) -> None:
    manifest = _read_manifest(root)
    if manifest.get("debug_subset"):
        raise RuntimeError(
            "Publication experiment stage refused: the processed manifest is a DEBUG/QUICK subset. "
            "Run `python run_all.py --stage preprocess --full-data` or `python run_all.py --full`."
        )


def _corpus_fingerprint(root: Path) -> str:
    """Identify the exact processed corpus a run was produced from."""

    return str(_read_manifest(root)["preprocessing_hash"])


def _completed(run_dir: Path, fingerprint: str) -> bool:
    """Report whether ``run_dir`` holds artifacts built from this exact corpus.

    Stamping the corpus fingerprint prevents a resumed run from silently reusing
    artifacts produced from a different (for example DEBUG/QUICK) preprocessing
    pass, which would compare models trained on different data.
    """

    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return status.get("preprocessing_hash") == fingerprint and status.get("status") == "COMPLETE"


def _completed_protocol(
    run_dir: Path,
    fingerprint: str,
    protocol_hash: str,
    artifact_path: Path | None = None,
) -> bool:
    """Require both the corpus and the complete experimental protocol to match."""

    status_path = run_dir / "run_status.json"
    if not _completed(run_dir, fingerprint):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if status.get("protocol_hash") != protocol_hash:
        return False
    if artifact_path is not None:
        expected_hash = status.get("prediction_sha256")
        if not artifact_path.is_file() or not expected_hash:
            return False
        return _file_sha256(artifact_path) == expected_hash
    return True


def _mark_completed(run_dir: Path, fingerprint: str, **details: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "COMPLETE",
        "preprocessing_hash": fingerprint,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    (run_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _settings_hash(*settings: dict[str, Any]) -> str:
    """Stable identifier for model/training/data settings used by resumable arms."""
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    """Hash executable Python/config sources used by a resumable experiment."""
    digest = hashlib.sha256()
    for directory in (root / "src", root / "scripts", root / "configs"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(path.read_bytes())
    entry_point = root / "run_all.py"
    if entry_point.is_file():
        digest.update(entry_point.relative_to(root).as_posix().encode("utf-8"))
        digest.update(entry_point.read_bytes())
    return digest.hexdigest()


def _model_settings(name: str, sequence_length: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    # ``eip_ktformer`` remains accepted for historical configs/checkpoints;
    # NuLKT is the canonical model name for all new experiments.
    name = "nulkt" if name == "eip_ktformer" else name
    common = {"hidden_dim": 128, "dropout": 0.2}
    settings: dict[str, dict[str, Any]] = {
        "mlp": {**common, "embedding_dim": 32},
        "lstm": {**common, "embedding_dim": 32},
        "gru": {**common, "embedding_dim": 32},
        "deepfm": {**common, "embedding_dim": 16},
        "dkt": {**common, "embedding_dim": 64},
        "dkvmn": {"memory_size": 50, "key_dim": 64, "value_dim": 128},
        "sakt": {**common, "num_heads": 4, "max_sequence_length": sequence_length},
        "akt": {**common, "num_heads": 4},
        "saint_plus": {**common, "num_heads": 4, "num_layers": 2, "max_sequence_length": sequence_length},
        "hierarchical_transformer": {**common, "num_heads": 4, "num_layers": 2, "max_sequence_length": sequence_length},
        "nulkt": {
            **common,
            "embedding_dim": 64,
            "num_heads": 4,
            "local_layers": 2,
            "global_layers": 2,
            "num_experts": 5,
            "max_sequence_length": sequence_length,
            "time_embedding_dim": 16,
            "learner_embedding_dim": 32,
            "contrastive_max_samples": 2048,
            # Legitimate for this task: an exercise's surface form is given at
            # prediction time, only its labels are withheld.
            "within_exercise_bidirectional": True,
        },
    }[name]
    settings.update(overrides or {})
    return settings


# Set by run_all when --patience is given. Early stopping, not the epoch ceiling,
# is what actually ends these runs, so a longer budget only takes effect if
# patience rises with it.
PATIENCE_OVERRIDE: int | None = None


def _training_settings(epochs: int, seed: int, optimizer: str = "adamw", **overrides: Any) -> dict[str, Any]:
    settings = {
        "epochs": epochs,
        "patience": PATIENCE_OVERRIDE if PATIENCE_OVERRIDE is not None else 5,
        "optimizer": optimizer,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "mixed_precision": True,
        "contrastive_weight": 0.1,
        "checkpoint_name": f"best_seed_{seed}",
    }
    settings.update(overrides)
    return settings


def _predict(model: torch.nn.Module, loader, device: torch.device) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            output = model(moved, return_diagnostics=False) if "return_diagnostics" in model.forward.__code__.co_varnames else model(moved)
            valid = ~moved["padding_mask"]
            if "score_mask" in moved:
                valid = valid & moved["score_mask"].bool()
            records.append(
                pd.DataFrame(
                    {
                        "label": moved["responses"][valid].cpu().numpy(),
                        "probability": torch.sigmoid(output["logits"])[valid].cpu().numpy(),
                        "learner_id": moved["learner"][valid].cpu().numpy(),
                        "row_index": moved.get("row_index", moved["learner"])[valid].cpu().numpy(),
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def _train(
    root: Path,
    loaders,
    vocab_sizes: dict[str, int],
    name: str,
    tag: str,
    seed: int,
    epochs: int,
    model_overrides: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
    sequence_length: int = 200,
    output_root: Path | None = None,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    seed_everything(seed)
    run = run_neural_experiment(
        name,
        vocab_sizes,
        loaders["train"],
        loaders["validation"],
        _model_settings(name, sequence_length, model_overrides),
        _training_settings(epochs, seed, **(training_overrides or {})),
        (output_root or (root / "results")) / "raw_runs" / tag / f"seed_{seed}",
    )
    device = next(run["model"].parameters()).device
    predictions = _predict(run["model"], loaders["validation"], device)
    metrics = binary_classification_metrics(predictions["label"], predictions["probability"])
    return run["model"], predictions, metrics


def run_classical_baselines(root: Path, *, skip_completed: bool = True) -> pd.DataFrame:
    """Fit tabular baselines on the full corpus, recording unavailability honestly."""

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    destination = metrics_dir / "classical_results.csv"
    run_dir = root / "results" / "raw_runs" / "classical_baselines"
    if skip_completed and _completed(run_dir, fingerprint) and destination.is_file():
        print("classical baselines status=SKIPPED reason=completed for this corpus", flush=True)
        return pd.read_csv(destination)

    frame, _ = load_processed_frame(root)
    train = frame[frame["chronological_split"].astype(str) == "train"]
    validation = frame[frame["chronological_split"].astype(str) == "validation"]
    x_train = train[CLASSICAL_FEATURES].fillna(0).to_numpy(np.float32)
    x_validation = validation[CLASSICAL_FEATURES].fillna(0).to_numpy(np.float32)
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    for name in CLASSICAL_BASELINES:
        try:
            result = run_classical_baseline(name, x_train, train["label"], x_validation, random_state=42)
            metrics = binary_classification_metrics(validation["label"], result["probabilities"])
            pd.DataFrame(
                {
                    "label": validation["label"].to_numpy(),
                    "probability": np.asarray(result["probabilities"], dtype=float),
                    "learner_id": validation["user_id_id"].to_numpy(),
                }
            ).to_csv(run_dir / f"{name}_validation_predictions.csv", index=False)
            metric_rows.append({"model": name, "status": "available", **metrics})
            print(f"classical={name} status=COMPLETE auroc={metrics['auroc']:.6f}", flush=True)
        except ImportError as exc:
            metric_rows.append({"model": name, "status": "unavailable", "reason": str(exc)})
            print(f"classical={name} status=UNAVAILABLE reason={exc}", flush=True)

    results = pd.DataFrame(metric_rows)
    results.to_csv(destination, index=False)
    _mark_completed(
        run_dir,
        fingerprint,
        stage="classical_baselines",
        n_validation=int(len(validation)),
        models=CLASSICAL_BASELINES,
    )
    return results


def run_baselines(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    seed: int = 42,
    skip_completed: bool = True,
) -> None:
    """Train every baseline on the full corpus, resumably and fault-isolated.

    Each model is stamped with the processed-corpus fingerprint, so an interrupted
    multi-hour run resumes without retraining finished models and without ever
    reusing artifacts built from a different preprocessing pass. A single model
    failure is recorded and the suite continues rather than discarding the run.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_classical_baselines(root, skip_completed=skip_completed)

    loaders, vocab_sizes = prepare_dataloaders(
        root, max_length=sequence_length, batch_size=batch_size, seed=seed
    )
    failures: list[dict[str, Any]] = []
    for name in NEURAL_BASELINES:
        run_dir = root / "results" / "raw_runs" / f"baseline_{name}"
        if skip_completed and _completed(run_dir, fingerprint):
            print(f"baseline={name} status=SKIPPED reason=completed for this corpus", flush=True)
            continue
        print(f"baseline={name} status=STARTING epochs={epochs}", flush=True)
        try:
            _, predictions, metrics = _train(
                root, loaders, vocab_sizes, name, f"baseline_{name}", seed, epochs,
                sequence_length=sequence_length,
            )
        except Exception as exc:  # one failed baseline must not discard the suite
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            print(f"baseline={name} status=FAILED error={type(exc).__name__}: {exc}", flush=True)
            continue
        predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
        _mark_completed(
            run_dir, fingerprint, stage="neural_baseline", model=name, seed=seed,
            epochs=epochs, sequence_length=sequence_length, batch_size=batch_size,
            n_validation=int(len(predictions)), validation_auroc=float(metrics["auroc"]),
        )
        print(
            f"baseline={name} status=COMPLETE validation_auroc={metrics['auroc']:.6f} "
            f"validation_auprc={metrics['auprc']:.6f}",
            flush=True,
        )

    # Rebuild the aggregate table from durable artifacts so resumed runs include
    # models completed in earlier sessions.
    neural_rows: list[dict[str, Any]] = []
    for name in NEURAL_BASELINES:
        path = root / "results" / "raw_runs" / f"baseline_{name}" / "validation_predictions.csv"
        if not path.is_file():
            continue
        predictions = pd.read_csv(path)
        neural_rows.append({
            "model": name, "seed": seed,
            **binary_classification_metrics(predictions["label"], predictions["probability"]),
        })
    pd.DataFrame(neural_rows).to_csv(metrics_dir / "neural_baseline_results.csv", index=False)
    if failures:
        (root / "results" / "raw_runs" / "baseline_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )


def _evaluate_classical_on_test(root: Path, name: str, best_dir: Path, fingerprint: str) -> None:
    """Fit the selected tabular baseline on train and read the test partition once.

    Classical baselines are otherwise only ever scored on validation, so a
    boosted winner had no test-partition evidence to compare the proposed model
    against. Seeds are not swept: the boosted learners are deterministic given a
    fixed ``random_state``, and a single fit is what the selection refers to.
    """

    from src.analysis.inference import indexed_dataloader, scored_positions

    frame, _ = load_processed_frame(root)
    splits = frame["chronological_split"].astype(str)
    train = frame[splits == "train"]
    if not (splits == "test").any():
        raise RuntimeError("No test partition is present in the processed corpus")

    # A tabular model naturally emits rows in corpus order, but the paired tests
    # downstream compare it row-for-row against a sequence model, which emits in
    # window order. Rebuild that emission order from the same windowing the
    # neural loader uses - it depends only on the windows, not on any model - and
    # score the classical baseline in exactly that order, so the two prediction
    # files are aligned by construction rather than by coincidence.
    loader, frame = indexed_dataloader(
        root, split="test", protocol="chronological_split",
        sequence_length=200, batch_size=256, context_length=0, frame=frame,
    )
    emitted: list[np.ndarray] = []
    for batch in loader:
        emitted.append(batch["row_index"][scored_positions(batch)].numpy())
    order = np.concatenate(emitted)
    order = order[order >= 0]
    if len(np.unique(order)) != len(order):
        raise RuntimeError("Window reconstruction scored a corpus row more than once")

    tested = frame.loc[order]
    x_train = train[CLASSICAL_FEATURES].fillna(0).to_numpy(np.float32)
    x_test = tested[CLASSICAL_FEATURES].fillna(0).to_numpy(np.float32)
    result = run_classical_baseline(name, x_train, train["label"], x_test, random_state=42)
    predictions = pd.DataFrame(
        {
            "label": tested["label"].to_numpy(),
            "probability": np.asarray(result["probabilities"], dtype=float),
            "learner_id": tested["user_id_id"].to_numpy(),
            "row_index": np.asarray(order, dtype=np.int64),
        }
    )
    predictions.to_csv(best_dir / "test_predictions.csv", index=False)
    metrics = binary_classification_metrics(predictions["label"], predictions["probability"])
    (best_dir / "test_metrics.json").write_text(
        json.dumps({"model": name, "family": "classical", **metrics}, indent=2), encoding="utf-8"
    )
    _mark_completed(best_dir, fingerprint, stage="best_baseline", model=name, family="classical")
    print(f"best_baseline model={name} status=COMPLETE test_auroc={metrics['auroc']:.6f}", flush=True)


def run_best_baseline_test(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    seeds: tuple[int, ...] = (42, 123, 2024, 2025, 2026),
    skip_completed: bool = True,
) -> None:
    """Select the strongest baseline on validation, then evaluate it once on test.

    The proposed model is compared against a baseline trained under the identical
    seed protocol. Selection uses validation AUROC only; the final test partition
    is read exactly once per seed, after selection is frozen.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    metrics_dir = root / "results" / "metrics"
    table_path = metrics_dir / "neural_baseline_results.csv"
    if not table_path.is_file():
        raise RuntimeError("Run the baseline suite before selecting a best baseline")
    table = pd.read_csv(table_path)
    if table.empty:
        raise RuntimeError("No completed neural baselines are available for selection")

    # Selection must range over every baseline family, not just the neural ones.
    # Restricting the pool to neural models understates the comparison whenever a
    # boosted tabular model scores higher on validation, which it does on this
    # corpus - reporting the strongest neural model as "the best baseline" would
    # overstate the proposed model's margin.
    pool = table[["model", "auroc", "auprc"]].copy()
    pool["family"] = "neural"
    classical_path = metrics_dir / "classical_results.csv"
    if classical_path.is_file():
        classical = pd.read_csv(classical_path)
        if "status" in classical:
            classical = classical[classical["status"].astype(str) == "available"]
        # The majority-class predictor is a degenerate reference, not a competitor.
        classical = classical[classical["model"].astype(str) != "majority"]
        if not classical.empty:
            available = classical[["model", "auroc", "auprc"]].copy()
            available["family"] = "classical"
            pool = pd.concat([pool, available], ignore_index=True)

    pool = pool.sort_values("auroc", ascending=False).reset_index(drop=True)
    winner = str(pool.loc[0, "model"])
    winner_family = str(pool.loc[0, "family"])
    selection = {
        "selected_model": winner,
        "selected_family": winner_family,
        "selection_metric": "validation auroc",
        "selection_partition": "validation",
        "selection_pool": "all baseline families (neural and classical)",
        "candidates": pool.to_dict(orient="records"),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    best_dir = root / "results" / "raw_runs" / "best_baseline"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"best_baseline selected={winner} family={winner_family} by validation auroc", flush=True)

    if winner_family == "classical":
        _evaluate_classical_on_test(root, winner, best_dir, fingerprint)
        return

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = best_dir / f"seed_{seed}"
        if skip_completed and _completed(seed_dir, fingerprint):
            print(f"best_baseline seed={seed} status=SKIPPED", flush=True)
            continue
        loaders, vocab_sizes = prepare_dataloaders(
            root, max_length=sequence_length, batch_size=batch_size, seed=seed
        )
        model, validation, validation_metrics = _train(
            root, loaders, vocab_sizes, winner, "best_baseline_training", seed, epochs,
            sequence_length=sequence_length,
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        validation.to_csv(seed_dir / "validation_predictions.csv", index=False)
        if "test" in loaders:
            tested = _predict(model, loaders["test"], next(model.parameters()).device)
            tested.to_csv(seed_dir / "test_predictions.csv", index=False)
        _mark_completed(
            seed_dir, fingerprint, stage="best_baseline", model=winner, seed=seed,
            epochs=epochs, validation_auroc=float(validation_metrics["auroc"]),
        )
        print(f"best_baseline seed={seed} status=COMPLETE validation_auroc={validation_metrics['auroc']:.6f}", flush=True)

    for seed in seeds:
        seed_dir = best_dir / f"seed_{seed}"
        for split in ("validation", "test"):
            path = seed_dir / f"{split}_predictions.csv"
            if path.is_file():
                predictions = pd.read_csv(path)
                rows.append({
                    "model": winner, "seed": seed, "split": split,
                    **binary_classification_metrics(predictions["label"], predictions["probability"]),
                })
    pd.DataFrame(rows).to_csv(root / "results" / "metrics" / "best_baseline_seed_results.csv", index=False)

    # ``run_all.statistics`` reads the paired seed-42 artifact by convention.
    canonical = best_dir / "seed_42" / "test_predictions.csv"
    if canonical.is_file():
        pd.read_csv(canonical).to_csv(best_dir / "test_predictions.csv", index=False)


def run_proposed(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    seeds: list[int] | tuple[int, ...] = (42, 123, 2024, 2025, 2026),
    skip_completed: bool = True,
) -> None:
    _require_publication_corpus(root)
    for seed in seeds:
        run_dir = root / "results" / "raw_runs" / "proposed" / f"seed_{seed}"
        complete = all(
            (run_dir / name).exists()
            for name in ("training_history.json", "validation_predictions.csv", "test_predictions.csv")
        )
        if skip_completed and complete:
            print(f"seed={seed} status=SKIPPED reason=completed artifacts already exist", flush=True)
            continue
        print(f"seed={seed} status=STARTING device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
        # Recreate loaders so the declared seed controls both initialization and
        # training-order randomness rather than inheriting generator state from
        # the preceding seed.
        loaders, vocab_sizes = prepare_dataloaders(
            root, max_length=sequence_length, batch_size=batch_size, seed=seed
        )
        model, validation, validation_metrics = _train(root, loaders, vocab_sizes, "nulkt", "proposed", seed, epochs, sequence_length=sequence_length)
        validation.to_csv(run_dir / "validation_predictions.csv", index=False)
        if "test" in loaders:
            tested = _predict(model, loaders["test"], next(model.parameters()).device)
            tested.to_csv(run_dir / "test_predictions.csv", index=False)
        print(
            f"seed={seed} status=COMPLETE validation_auroc={validation_metrics['auroc']:.6f} "
            f"validation_auprc={validation_metrics['auprc']:.6f}",
            flush=True,
        )

    # Rebuild aggregate metrics from durable prediction artifacts. This makes
    # interrupted/restarted Spyder runs include earlier completed seeds without
    # evaluating the protected test set again.
    rows: list[dict[str, Any]] = []
    test_log: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = root / "results" / "raw_runs" / "proposed" / f"seed_{seed}"
        validation_path = run_dir / "validation_predictions.csv"
        test_path = run_dir / "test_predictions.csv"
        if validation_path.exists():
            validation = pd.read_csv(validation_path)
            rows.append({
                "model": "NuLKT", "seed": seed, "split": "validation",
                **binary_classification_metrics(validation["label"], validation["probability"]),
            })
        if test_path.exists():
            tested = pd.read_csv(test_path)
            rows.append({
                "model": "NuLKT", "seed": seed, "split": "test",
                **binary_classification_metrics(tested["label"], tested["probability"]),
            })
            test_log.append({
                "seed": seed,
                "configuration": "frozen configs/proposed.yaml",
                "evaluations": 1,
                "source": str(test_path),
            })
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "seed_results.csv", index=False)
    (root / "results" / "test_evaluation_log.json").write_text(json.dumps(test_log, indent=2), encoding="utf-8")


def run_official_neural(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    context_length: int = 0,
    seeds: tuple[int, ...] = (42, 123, 2024),
    skip_completed: bool = True,
    exercise_context: bool = True,
    lr_schedule: str = "cosine_warmup",
    ema: bool = True,
) -> None:
    """Train NuLKT under the official SLAM partition for leaderboard comparability.

    ``context_length`` defaults to 0, matching the configuration the chronological
    stage is validated against. Overlapping context windows were tried here first
    and destabilised training badly: validation loss rose from 0.98 to 1.78 while
    training loss fell, and the model predicted a 0.44 error rate against a true
    rate of 0.14. That path needs isolating before it is used again.

    The shared task designates ``dev`` for model selection and ``test`` for the
    single reported evaluation, so early stopping uses dev and test is predicted
    once per seed. Row indices are retained so the boosted arm can be aligned to
    these predictions without a positional merge.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    from src.analysis.inference import collect_predictions, indexed_dataloader
    from src.data.torch_dataset import vocab_sizes_for_model

    tag = "official_eip_ktformer"
    for seed in seeds:
        run_dir = root / "results" / "raw_runs" / tag / f"seed_{seed}"
        if skip_completed and _completed(run_dir, fingerprint):
            print(f"official_neural seed={seed} status=SKIPPED", flush=True)
            continue
        print(f"official_neural seed={seed} status=STARTING epochs={epochs}", flush=True)

        loaders: dict[str, Any] = {}
        frame = None
        # The official partition names dev/test; the loaders use the project's
        # train/validation/test vocabulary, so dev is loaded as validation.
        for split, alias in (("train", "train"), ("dev", "validation"), ("test", "test")):
            loader, frame = indexed_dataloader(
                root, split=split, protocol="official_split", sequence_length=sequence_length,
                batch_size=batch_size, frame=frame, context_length=context_length,
            )
            loaders[alias] = loader

        seed_everything(seed)
        # The leaderboard arm previously trained a bare configuration: no
        # exercise context, no schedule, no weight averaging. Each of those has
        # since been measured. Exercise context is worth +0.0195 test AUROC on
        # the chronological split, and the screen in configs/advanced.yaml
        # promoted the schedule/EMA pair on every gate. Withholding them here
        # meant the arm feeding the published ensemble was the weakest version
        # of the model rather than the best one.
        model_overrides = {"exercise_context": True} if exercise_context else {}
        training_overrides: dict[str, Any] = {}
        if lr_schedule and lr_schedule != "none":
            training_overrides["lr_schedule"] = lr_schedule
            training_overrides["warmup_ratio"] = 0.08
        if ema:
            training_overrides["ema"] = True
            training_overrides["ema_decay"] = 0.9995
        model, _, validation_metrics = _train(
            root, loaders, vocab_sizes_for_model(root), "nulkt", tag, seed, epochs,
            model_overrides=model_overrides,
            training_overrides=training_overrides,
            sequence_length=sequence_length,
        )
        device = next(model.parameters()).device
        run_dir.mkdir(parents=True, exist_ok=True)
        for alias, name in (("validation", "dev"), ("test", "test")):
            predictions = collect_predictions(model, loaders[alias], device, with_expert_weights=False)
            predictions["probability"] = predictions["logit_probability"]
            predictions[["row_index", "label", "probability", "learner_id"]].to_csv(
                run_dir / f"{name}_predictions.csv", index=False
            )
        _mark_completed(
            run_dir, fingerprint, stage="official_neural", model="nulkt", seed=seed,
            epochs=epochs, protocol="official_split", context_length=context_length,
            exercise_context=exercise_context, lr_schedule=lr_schedule, ema=ema,
            dev_auroc=float(validation_metrics["auroc"]),
        )
        print(f"official_neural seed={seed} status=COMPLETE dev_auroc={validation_metrics['auroc']:.6f}", flush=True)


def run_benchmark(root: Path, *, regime: str = "benchmark") -> None:
    """Fit the boosted arm on the official split and blend it with the neural arm."""

    from scripts.benchmark_official import run as run_official

    payload = run_official(root, regime=regime)
    print(json.dumps({k: v for k, v in payload.items() if k != "features"}, indent=2, default=float), flush=True)


def run_ablation(root: Path, *, epochs: int = 10, sequence_length: int = 200, batch_size: int = 64) -> None:
    _require_publication_corpus(root)
    config = yaml.safe_load((root / "configs" / "ablation.yaml").read_text(encoding="utf-8"))
    validate_ablation_config(config)
    loaders, vocab_sizes = prepare_dataloaders(root, max_length=sequence_length, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    for name, specification in config["ablations"].items():
        model_name = specification.get("model_override", "nulkt")
        model_name = "nulkt" if model_name == "eip_ktformer" else model_name
        model_overrides = {"ablations": specification.get("model_ablation", {})} if model_name == "nulkt" else {}
        _, _, metrics = _train(
            root, loaders, vocab_sizes, model_name, f"ablation_{name}", 42, epochs,
            model_overrides=model_overrides,
            training_overrides=specification.get("training_override", {}),
            sequence_length=sequence_length,
        )
        rows.append({"variant": name, **metrics})
    pd.DataFrame(rows).to_csv(root / "results" / "metrics" / "ablation_results.csv", index=False)


def validate_ablation_config(config: dict[str, Any]) -> None:
    """Reject misspelled or no-op ablation declarations before any GPU work."""

    variants = config.get("ablations")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("ablation config must define a non-empty 'ablations' mapping")
    allowed_fields = {"model_ablation", "model_override", "training_override"}
    for name, specification in variants.items():
        if not isinstance(specification, dict):
            raise ValueError(f"ablation {name!r} must be a mapping")
        unknown_fields = set(specification).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(f"ablation {name!r} has unknown fields: {sorted(unknown_fields)}")
        model_ablation = specification.get("model_ablation", {})
        if not isinstance(model_ablation, dict):
            raise ValueError(f"ablation {name!r} model_ablation must be a mapping")
        unknown_keys = set(model_ablation).difference(MODEL_ABLATION_KEYS)
        if unknown_keys:
            raise ValueError(
                f"ablation {name!r} has unsupported model keys: {sorted(unknown_keys)}"
            )
        if any(value is not True for value in model_ablation.values()):
            raise ValueError(f"ablation {name!r} model-ablation flags must be true")
        model_override = specification.get("model_override", "nulkt")
        model_override = "nulkt" if model_override == "eip_ktformer" else model_override
        if model_override not in {"nulkt", "lstm"}:
            raise ValueError(f"ablation {name!r} has unsupported model override {model_override!r}")
        if model_override != "nulkt" and model_ablation:
            raise ValueError(
                f"ablation {name!r} cannot combine a replacement model with model flags"
            )
        if not isinstance(specification.get("training_override", {}), dict):
            raise ValueError(f"ablation {name!r} training_override must be a mapping")


def summarize_matched_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed matched-budget ablations across random seeds."""

    required = {"variant", "seed", "budget_epochs"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"matched ablation rows are missing columns: {missing}")
    metric_names = [
        name
        for name in (
            "auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "mcc",
            "brier_score", "ece",
        )
        if name in frame.columns
    ]
    rows: list[dict[str, Any]] = []
    for variant, block in frame.groupby("variant", sort=False):
        record: dict[str, Any] = {
            "variant": variant,
            "seeds": int(block["seed"].nunique()),
            "seed_list": ", ".join(str(value) for value in sorted(block["seed"].unique())),
            "budget_epochs": int(block["budget_epochs"].iloc[0]),
            "selection_partition": "chronological validation",
        }
        for metric in metric_names:
            values = pd.to_numeric(block[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(record)
    return pd.DataFrame(rows)


def run_matched_ablation(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    seeds: tuple[int, ...] = (42, 123, 2024, 2025, 2026),
    skip_completed: bool = True,
) -> None:
    """Run every real ablation at the proposed model's budget over five seeds.

    Only validation predictions are produced. Each variant/seed has its own
    durable checkpoint, history, predictions, and corpus-fingerprint marker, so
    a multi-day run resumes at the first unfinished cell. The publication table
    is promoted only after every configured variant has every requested seed.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    config_path = root / "configs" / "ablation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_ablation_config(config)
    variants = config["ablations"]
    output_root = root / "results" / "raw_runs" / "matched_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_hash = _settings_hash(
        config,
        {
            "epochs": epochs,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "seeds": list(seeds),
            "selection_partition": "chronological validation",
            "source_hash": _source_hash(root),
        },
    )
    (output_root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    for seed in seeds:
        loaders, vocab_sizes = prepare_dataloaders(
            root, max_length=sequence_length, batch_size=batch_size, seed=seed
        )
        for name, specification in variants.items():
            run_dir = output_root / name / f"seed_{seed}"
            prediction_path = run_dir / "validation_predictions.csv"
            if (
                skip_completed
                and prediction_path.is_file()
                and _completed_protocol(
                    run_dir, fingerprint, protocol_hash, prediction_path
                )
            ):
                print(f"matched_ablation variant={name} seed={seed} status=SKIPPED", flush=True)
                continue
            model_name = specification.get("model_override", "nulkt")
            model_name = "nulkt" if model_name == "eip_ktformer" else model_name
            model_overrides = (
                {"ablations": specification.get("model_ablation", {})}
                if model_name == "nulkt"
                else {}
            )
            print(
                f"matched_ablation variant={name} seed={seed} status=STARTING "
                f"epochs={epochs}",
                flush=True,
            )
            model, predictions, metrics = _train(
                root,
                loaders,
                vocab_sizes,
                model_name,
                f"matched_ablation/{name}",
                seed,
                epochs,
                model_overrides=model_overrides,
                training_overrides=specification.get("training_override", {}),
                sequence_length=sequence_length,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)
            model_device = next(model.parameters()).device
            _mark_completed(
                run_dir,
                fingerprint,
                stage="matched_ablation",
                variant=name,
                model=model_name,
                seed=seed,
                epochs=epochs,
                sequence_length=sequence_length,
                batch_size=batch_size,
                validation_rows=int(len(predictions)),
                validation_auroc=float(metrics["auroc"]),
                training_device=str(model_device),
                cuda_device_name=(
                    torch.cuda.get_device_name(model_device)
                    if model_device.type == "cuda"
                    else None
                ),
                protocol_hash=protocol_hash,
                prediction_sha256=_file_sha256(prediction_path),
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"matched_ablation variant={name} seed={seed} status=COMPLETE "
                f"validation_auroc={metrics['auroc']:.6f}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for name in variants:
        for seed in seeds:
            path = output_root / name / f"seed_{seed}" / "validation_predictions.csv"
            run_dir = path.parent
            if not path.is_file() or not _completed_protocol(
                run_dir, fingerprint, protocol_hash, path
            ):
                continue
            predictions = pd.read_csv(path)
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "budget_epochs": epochs,
                    **binary_classification_metrics(
                        predictions["label"], predictions["probability"]
                    ),
                }
            )
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(metrics_dir / "five_seed_ablation_results.csv", index=False)
    summary = summarize_matched_ablation(raw)
    summary.to_csv(metrics_dir / "five_seed_ablation_summary.csv", index=False)

    expected = len(variants) * len(seeds)
    if len(raw) == expected and (summary["seeds"] == len(seeds)).all():
        # Table 8 consumes the matched-only path. Never replace it with a partial run.
        summary.to_csv(metrics_dir / "matched_ablation_results.csv", index=False)
        summary.to_csv(metrics_dir / "ablation_results.csv", index=False)
    else:
        print(
            f"matched_ablation status=PARTIAL completed={len(raw)}/{expected}; "
            "publication matched_ablation_results.csv was not replaced",
            flush=True,
        )


def run_optimizer_study(root: Path, *, epochs: int = 10, sequence_length: int = 200, batch_size: int = 64) -> None:
    _require_publication_corpus(root)
    loaders, vocab_sizes = prepare_dataloaders(root, max_length=sequence_length, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    for name in ["adam", "adamw", "sgd_momentum", "rmsprop"]:
        _, _, metrics = _train(root, loaders, vocab_sizes, "nulkt", f"optimizer_{name}", 42, epochs, training_overrides={"optimizer": name}, sequence_length=sequence_length)
        rows.append({"optimizer": name, **metrics})
    pd.DataFrame(rows).to_csv(root / "results" / "metrics" / "optimizer_results.csv", index=False)


def _hyperparameter_trials(config: dict[str, Any], max_trials: int) -> list[tuple[int, dict[str, Any]]]:
    """Enumerate the search grid as ``(trial_index, model_overrides)`` pairs.

    The index counts every combination the product yields, including ones dropped
    for an indivisible head count, so a trial keeps the same number — and the same
    ``raw_runs`` directory — across resumes even if the filter changes.
    """

    grid = itertools.product(config["hidden_dim"], config["embedding_dim"], config["num_heads"], config["dropout"])
    trials: list[tuple[int, dict[str, Any]]] = []
    for trial, (hidden, embedding, heads, dropout) in enumerate(grid):
        if trial >= max_trials:
            break
        if hidden % heads:
            continue
        trials.append((trial, {"hidden_dim": hidden, "embedding_dim": embedding, "num_heads": heads, "dropout": dropout}))
    return trials


def _metrics_from_finished_run(
    run_dir: Path,
    loaders,
    vocab_sizes: dict[str, int],
    model_settings: dict[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    """Recompute a finished trial's validation metrics from its best checkpoint.

    ``Trainer.fit`` writes ``training_history.json`` only once it returns, so the
    file marks training that ran to completion rather than one killed mid-epoch,
    and ``_train`` reports metrics from the best-validation checkpoint rather than
    the last epoch. Reloading that checkpoint therefore reproduces the very numbers
    a retrain would produce, which is what makes an interrupted search cheap to
    resume. Returns ``None`` whenever the artifacts are absent or were written under
    a different architecture, leaving the caller to train the trial properly.
    """

    checkpoint_path = run_dir / "checkpoints" / f"best_seed_{seed}.pt"
    if not (run_dir / "training_history.json").is_file() or not checkpoint_path.is_file():
        return None
    model = create_model("nulkt", vocab_sizes, model_settings)
    device = select_device(None)
    state = CheckpointManager(run_dir / "checkpoints").load(
        checkpoint_path, model, optimizer=None, scheduler=None, map_location=device, restore_rng=False
    )
    stored = state.get("extra", {}).get("model_config", {})
    divergent = {key: (stored.get(key), value) for key, value in model_settings.items() if key in stored and stored[key] != value}
    if divergent:
        print(f"hyperparameters run_dir={run_dir.parent.name} status=STALE differs={divergent}", flush=True)
        return None
    predictions = _predict(model.to(device), loaders["validation"], device)
    return binary_classification_metrics(predictions["label"], predictions["probability"])


def run_hyperparameters(
    root: Path,
    *,
    epochs: int = 5,
    max_trials: int = 12,
    sequence_length: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    skip_completed: bool = True,
) -> None:
    """Grid-search the proposed model on validation data, one resumable trial at a time.

    Every trial stamps its own completion, and the results table is rewritten after
    each one, so a crash mid-search costs a single trial instead of the whole stage.
    A GPU fault poisons the CUDA context for the rest of the process, so recovery is
    a fresh run of this stage rather than an in-process retry.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    config = yaml.safe_load((root / "configs" / "hyperparameter_search.yaml").read_text(encoding="utf-8"))["search_space"]
    destination = root / "results" / "metrics" / "hyperparameter_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # One set of loaders serves every trial: the grid varies architecture only, and
    # rebuilding an identical corpus view per trial cost minutes of every search.
    loaders, vocab_sizes = prepare_dataloaders(root, max_length=sequence_length, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    for trial, overrides in _hyperparameter_trials(config, max_trials):
        tag = f"hyperparameter_{trial:03d}"
        run_dir = root / "results" / "raw_runs" / tag / f"seed_{seed}"
        model_overrides = {**overrides, "local_layers": 2, "global_layers": 2, "max_sequence_length": sequence_length}
        model_settings = _model_settings("nulkt", sequence_length, model_overrides)
        metrics: dict[str, Any] | None = None
        if skip_completed:
            stamped = _completed(run_dir, fingerprint)
            metrics = _metrics_from_finished_run(run_dir, loaders, vocab_sizes, model_settings, seed)
            if metrics is not None:
                print(f"hyperparameters trial={trial} status={'SKIPPED' if stamped else 'RECOVERED'} {overrides}", flush=True)
        if metrics is None:
            print(f"hyperparameters trial={trial} status=STARTING {overrides}", flush=True)
            _, _, metrics = _train(
                root, loaders, vocab_sizes, "nulkt", tag, seed, epochs,
                model_overrides=model_overrides, sequence_length=sequence_length,
            )
            print(f"hyperparameters trial={trial} status=COMPLETE validation_auprc={metrics['auprc']:.6f}", flush=True)
        _mark_completed(
            run_dir, fingerprint, stage="hyperparameters", trial=trial, seed=seed, epochs=epochs,
            sequence_length=sequence_length, batch_size=batch_size, **overrides, metrics=metrics,
        )
        rows.append({"trial": trial, **overrides, **metrics})
        # Rewritten per trial so an interrupted search still leaves a usable table.
        pd.DataFrame(rows).to_csv(destination, index=False)


def run_calibration(root: Path) -> None:
    _require_publication_corpus(root)
    run_dir = root / "results" / "raw_runs" / "proposed" / "seed_42"
    validation = pd.read_csv(run_dir / "validation_predictions.csv")
    tested = pd.read_csv(run_dir / "test_predictions.csv")
    scaler = TemperatureScaler().fit(validation["label"], validation["probability"])
    calibrated = scaler.transform(tested["probability"])
    payload = calibration_comparison(tested["label"], tested["probability"], calibrated)
    payload["temperature"] = scaler.temperature
    (root / "results" / "metrics" / "calibration_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"temperature": scaler.temperature, **payload["pre"], **{f"post_{k}": v for k, v in payload["post"].items()}}]).to_csv(root / "results" / "metrics" / "calibration_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baselines", "best_baseline", "proposed", "official_neural", "benchmark", "ablation", "matched_ablation", "optimizer", "hyperparameters", "calibration", "enhanced", "advanced"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", default="safe_baseline")
    args = parser.parse_args()
    functions = {
        "baselines": run_baselines,
        "best_baseline": run_best_baseline_test,
        "proposed": run_proposed,
        "official_neural": run_official_neural,
        "benchmark": run_benchmark,
        "ablation": run_ablation,
        "matched_ablation": run_matched_ablation,
        "optimizer": run_optimizer_study,
        "hyperparameters": run_hyperparameters,
        "calibration": run_calibration,
        "enhanced": run_enhanced,
        "advanced": run_advanced,
    }
    kwargs = {} if args.epochs is None or args.stage == "calibration" else {"epochs": args.epochs}
    if args.stage == "advanced":
        kwargs["variant"] = args.variant
    functions[args.stage](args.root.resolve(), **kwargs)


def run_enhanced(
    root: Path,
    *,
    epochs: int = 30,
    sequence_length: int = 200,
    batch_size: int = 64,
    seeds: tuple[int, ...] = (42, 123, 2024),
    skip_completed: bool = True,
) -> None:
    """Train the proposed model with the exercise-context arm switched on.

    A separate arm rather than an ablation row: ``exercise_context`` adds a
    signal the published model never had, so it belongs beside the baseline
    configuration as a candidate improvement, not inside a table whose other
    rows all remove something.

    The motivating measurement is in ``results/analysis/feature_gap.json`` -
    gradient boosting draws roughly a third of its split gain from inputs the
    sequence model never receives, and prompt identity alone is half of that.
    """

    raise RuntimeError(
        "The legacy enhanced stage is archived because its completed artifacts used current-response "
        "latency and automatic test scoring. Use the validation-only advanced stage instead."
    )

    _require_publication_corpus(root)  # pragma: no cover - retained legacy implementation below
    overrides = {"exercise_context": True, "exercise_scalar_features": 4}
    for seed in seeds:
        run_dir = root / "results" / "raw_runs" / "enhanced" / f"seed_{seed}"
        complete = all(
            (run_dir / name).exists()
            for name in ("validation_predictions.csv", "test_predictions.csv")
        )
        if skip_completed and complete:
            print(f"enhanced seed={seed} status=SKIPPED", flush=True)
            continue
        print(f"enhanced seed={seed} status=STARTING", flush=True)
        loaders, vocab_sizes = prepare_dataloaders(
            root, max_length=sequence_length, batch_size=batch_size, seed=seed
        )
        model, validation, validation_metrics = _train(
            root, loaders, vocab_sizes, "nulkt", "enhanced", seed, epochs,
            model_overrides=overrides, sequence_length=sequence_length,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        validation.to_csv(run_dir / "validation_predictions.csv", index=False)
        if "test" in loaders:
            tested = _predict(model, loaders["test"], next(model.parameters()).device)
            tested.to_csv(run_dir / "test_predictions.csv", index=False)
        print(
            f"enhanced seed={seed} status=COMPLETE validation_auroc={validation_metrics['auroc']:.6f}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = root / "results" / "raw_runs" / "enhanced" / f"seed_{seed}"
        for split, name in (("validation", "validation_predictions.csv"), ("test", "test_predictions.csv")):
            path = run_dir / name
            if path.exists():
                frame = pd.read_csv(path)
                rows.append({
                    "model": "NuLKT+context", "seed": seed, "split": split,
                    **binary_classification_metrics(frame["label"], frame["probability"]),
                })
    if rows:
        destination = root / "results" / "metrics" / "enhanced_seed_results.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(f"enhanced wrote {destination}", flush=True)


def run_advanced(
    root: Path,
    *,
    epochs: int | None = None,
    sequence_length: int = 200,
    batch_size: int | None = None,
    seeds: tuple[int, ...] | None = None,
    variant: str = "safe_baseline",
    skip_completed: bool = True,
) -> None:
    """Validation-only screen of the leakage-safe, expert-designed candidate.

    This stage deliberately does not construct a test loader. Promotion to a
    five-seed locked test run is a separate decision after validation screening.
    """

    _require_publication_corpus(root)
    fingerprint = _corpus_fingerprint(root)
    specification = yaml.safe_load((root / "configs" / "advanced.yaml").read_text(encoding="utf-8"))
    output_root = (root / str(specification["output"]["root"])).resolve()
    if root.resolve() not in output_root.parents:
        raise RuntimeError(f"Advanced output root must stay inside the project: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(specification, sort_keys=False), encoding="utf-8"
    )
    model_overrides = dict(specification["model"])
    training_overrides = dict(specification["training"])
    variants = specification["screen"]["variants"]
    if variant not in variants:
        raise ValueError(f"Unknown advanced variant {variant!r}; expected one of {sorted(variants)}")
    variant_settings = variants[variant]
    model_overrides.update(variant_settings.get("model", {}))
    training_overrides.update(variant_settings.get("training", {}))
    yaml_epochs = int(training_overrides.pop("epochs", 40))
    configured_epochs = int(epochs) if epochs is not None else yaml_epochs
    configured_patience = int(training_overrides.get("patience", 10))
    if PATIENCE_OVERRIDE is not None:
        configured_patience = PATIENCE_OVERRIDE
    training_overrides["patience"] = configured_patience
    data_settings = dict(specification["data"])
    data_settings.update(variant_settings.get("data", {}))
    configured_batch_size = int(
        batch_size if batch_size is not None else data_settings.get("batch_size", 64)
    )
    data_settings.update(
        {"sequence_length": sequence_length, "batch_size": configured_batch_size}
    )
    if data_settings.get("include_test") is not False:
        raise RuntimeError("Advanced model selection must set include_test: false")
    selected_seeds = tuple(int(seed) for seed in (seeds or specification["screen"]["seeds"]))
    resolved_model = _model_settings("nulkt", sequence_length, model_overrides)
    resolved_training = _training_settings(
        configured_epochs, 0, **training_overrides
    )
    resolved_training.pop("checkpoint_name", None)
    source_hash = _source_hash(root)
    config_hash = _settings_hash(
        resolved_model, resolved_training, data_settings,
        {"source": source_hash, "variant": variant},
    )
    required_names = ("training_history.json", "validation_predictions.csv")
    for seed in selected_seeds:
        run_dir = output_root / "raw_runs" / "advanced" / variant / f"seed_{seed}"
        status_path = run_dir / "run_status.json"
        checkpoint = run_dir / "checkpoints" / f"best_seed_{seed}.pt"
        durable = all((run_dir / name).is_file() for name in required_names) and checkpoint.is_file()
        if skip_completed and status_path.is_file() and durable:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            artifacts = {
                "training_history.json": run_dir / "training_history.json",
                "validation_predictions.csv": run_dir / "validation_predictions.csv",
                "checkpoint": checkpoint,
            }
            artifact_hashes = status.get("artifact_hashes", {})
            artifacts_match = bool(artifact_hashes) and all(
                artifact_hashes.get(name) == _file_sha256(path)
                for name, path in artifacts.items()
            )
            if (
                status.get("status") == "COMPLETE"
                and status.get("preprocessing_hash") == fingerprint
                and status.get("config_hash") == config_hash
                and artifacts_match
            ):
                print(f"advanced seed={seed} status=SKIPPED reason=matching hash", flush=True)
                continue
        loaders, vocab_sizes = prepare_dataloaders(
            root,
            protocol=str(data_settings["protocol"]),
            max_length=sequence_length,
            context_length=int(data_settings["context_length"]),
            batch_size=configured_batch_size,
            seed=seed,
            feature_regime=str(data_settings["feature_regime"]),
            include_test=False,
        )
        model, validation, metrics = _train(
            root,
            loaders,
            vocab_sizes,
            "nulkt",
            f"advanced/{variant}",
            seed,
            configured_epochs,
            model_overrides=model_overrides,
            training_overrides=training_overrides,
            sequence_length=sequence_length,
            output_root=output_root,
        )
        del model
        run_dir.mkdir(parents=True, exist_ok=True)
        validation.to_csv(run_dir / "validation_predictions.csv", index=False)
        artifacts = {
            "training_history.json": run_dir / "training_history.json",
            "validation_predictions.csv": run_dir / "validation_predictions.csv",
            "checkpoint": run_dir / "checkpoints" / f"best_seed_{seed}.pt",
        }
        _mark_completed(
            run_dir,
            fingerprint,
            stage="advanced_validation_screen",
            variant=variant,
            seed=seed,
            config_hash=config_hash,
            source_hash=source_hash,
            model_settings=model_overrides,
            training_settings=training_overrides,
            data_settings=data_settings,
            artifact_hashes={name: _file_sha256(path) for name, path in artifacts.items()},
            validation_rows=int(len(validation)),
            validation_auroc=metrics["auroc"],
            validation_auprc=metrics["auprc"],
        )
    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        path = output_root / "raw_runs" / "advanced" / variant / f"seed_{seed}" / "validation_predictions.csv"
        if path.is_file():
            predictions = pd.read_csv(path)
            rows.append({
                "model": "NuLKT-Advanced", "variant": variant, "seed": seed,
                **binary_classification_metrics(predictions["label"], predictions["probability"]),
            })
    if not rows:
        raise RuntimeError("Advanced screen produced no validation predictions")
    destination = output_root / "metrics" / f"advanced_{variant}_validation_results.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"advanced wrote {destination}", flush=True)

    baseline_path = output_root / "metrics" / "advanced_safe_baseline_validation_results.csv"
    if variant != "safe_baseline" and baseline_path.is_file():
        baseline = pd.read_csv(baseline_path).set_index("seed")
        candidate = pd.DataFrame(rows).set_index("seed")
        common = baseline.index.intersection(candidate.index)
        if len(common):
            delta_auroc = float((candidate.loc[common, "auroc"] - baseline.loc[common, "auroc"]).mean())
            delta_auprc = float((candidate.loc[common, "auprc"] - baseline.loc[common, "auprc"]).mean())
            delta_brier = float(
                (candidate.loc[common, "brier_score"] - baseline.loc[common, "brier_score"]).mean()
            )
            delta_ece = float((candidate.loc[common, "ece"] - baseline.loc[common, "ece"]).mean())
            per_seed_positive = bool(
                ((candidate.loc[common, "auroc"] - baseline.loc[common, "auroc"]) > 0).all()
                and ((candidate.loc[common, "auprc"] - baseline.loc[common, "auprc"]) > 0).all()
            )
            thresholds = specification["screen"]
            promotion = {
                "variant": variant,
                "reference": "safe_baseline",
                "seeds": [int(seed) for seed in common],
                "mean_delta_auroc": delta_auroc,
                "mean_delta_auprc": delta_auprc,
                "mean_delta_brier": delta_brier,
                "mean_delta_ece": delta_ece,
                "positive_auroc_and_auprc_on_every_seed": per_seed_positive,
                "required_delta_auroc": float(thresholds["promotion_auroc_delta"]),
                "required_delta_auprc": float(thresholds["promotion_auprc_delta"]),
                "maximum_brier_delta": float(thresholds["max_brier_delta"]),
                "maximum_ece_delta": float(thresholds["max_ece_delta"]),
                "pass": bool(
                    delta_auroc >= float(thresholds["promotion_auroc_delta"])
                    and delta_auprc >= float(thresholds["promotion_auprc_delta"])
                    and delta_brier <= float(thresholds["max_brier_delta"])
                    and delta_ece <= float(thresholds["max_ece_delta"])
                    and per_seed_positive
                ),
            }
            promotion_path = output_root / "metrics" / f"advanced_{variant}_promotion.json"
            promotion_path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("Use scripts/reproduce_offline_safe.py; direct legacy stages do not enforce the split-frozen protocol.")
