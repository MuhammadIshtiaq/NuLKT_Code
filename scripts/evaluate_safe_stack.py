"""Select the seven original NuLKT/GBDT combiners on DEV, then score TEST.

Run --select-dev first. It reads no TEST labels, predictions, or metrics and
serializes five learner-grouped-fold selection of the original seven candidates,
the full-DEV fitted winner, and a grouped-OOF DEV F1 threshold.
Only --evaluate-test opens TEST predictions. The historical TEST was inspected
before this correction, so its results are retrospective, not fresh blind scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_safe_neural_test import SEEDS, _check_completed_fit, _check_data, sha  # noqa: E402
from scripts.run_safe_boosted import (  # noqa: E402
    CAT_SEEDS, LGB_SEEDS, MEMBERS, _protocol, _read_artifact,
)
from src.evaluation.ensemble import (  # noqa: E402
    _logit as _combiner_logit, _rank_uniform, _stack_matrix, build_combiners, cross_validate,
    grouped_oof_predictions, select_combiner,
)
from src.evaluation.metrics import binary_classification_metrics  # noqa: E402
from src.evaluation.statistics import learner_clustered_bootstrap  # noqa: E402

IDENTITY = ("row_index", "label", "learner_id")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _identity(frame: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    if len(frame) != expected_rows or not set([*IDENTITY, "probability"]) <= set(frame):
        raise RuntimeError("Prediction rows or columns do not match the official split")
    ordered = frame[[*IDENTITY, "probability"]].sort_values("row_index", kind="stable").reset_index(drop=True)
    if (ordered.row_index.duplicated().any() or not ordered.row_index.is_monotonic_increasing
            or not ordered.label.isin([0, 1]).all()
            or not np.isfinite(ordered.probability).all()
            or not ordered.probability.between(0, 1).all()):
        raise RuntimeError("Invalid prediction identity, targets, or probabilities")
    return ordered


def _aligned(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if not reference[list(IDENTITY)].equals(other[list(IDENTITY)]):
        raise RuntimeError(f"Row, label, or learner identity mismatch for {name}")


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def _freeze_state(name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Store all parameters needed to apply the selected DEV fit without refit."""
    if name in {"neural_only", "boosted_only"}:
        return {"type": "reference", "name": name}
    if name in {"probability_blend", "logit_blend", "rank_blend"}:
        weights = [float(value) for value in state["weights"]]
        if len(weights) != 2 or not np.isclose(sum(weights), 1.0):
            raise RuntimeError("Invalid fitted blend weights")
        return {"type": "blend", "name": name, "weights": weights}
    if name in {"logistic_stack", "logistic_stack_interactions"}:
        model = state["model"]
        coefficients = model.coef_.ravel().astype(float).tolist()
        if not np.array_equal(model.classes_, [0, 1]):
            raise RuntimeError("Unexpected logistic-stack class ordering")
        expected_features = 4 if name.endswith("interactions") else 2
        if len(coefficients) != expected_features:
            raise RuntimeError("Unexpected logistic-stack feature count")
        return {
            "type": "logistic", "name": name, "coefficients": coefficients,
            "intercept": float(model.intercept_[0]), "classes": [0, 1],
            "feature_order": (
                ["neural_logit", "boosted_logit", "logit_product", "absolute_logit_difference"]
                if expected_features == 4 else ["neural_logit", "boosted_logit"]
            ),
        }
    raise RuntimeError(f"Unknown selected combiner {name}")


def _apply_frozen(frozen: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    name = frozen["name"]
    if name in {"neural_only", "boosted_only"}:
        return frame[f"probability_{name.removesuffix('_only')}"].to_numpy(dtype=float)
    if frozen["type"] == "blend":
        weights = frozen["weights"]
        neural = frame.probability_neural.to_numpy(dtype=float)
        boosted = frame.probability_boosted.to_numpy(dtype=float)
        if name == "probability_blend":
            return weights[0] * neural + weights[1] * boosted
        if name == "logit_blend":
            return _sigmoid(weights[0] * _combiner_logit(neural) + weights[1] * _combiner_logit(boosted))
        if name == "rank_blend":
            return weights[0] * _rank_uniform(neural) + weights[1] * _rank_uniform(boosted)
    if frozen["type"] == "logistic":
        matrix = _stack_matrix(frame, name.endswith("interactions"))
        coefficients = np.asarray(frozen["coefficients"], dtype=float)
        if matrix.shape[1] != len(coefficients):
            raise RuntimeError("Frozen logistic feature matrix mismatch")
        return _sigmoid(matrix @ coefficients + float(frozen["intercept"]))
    raise RuntimeError(f"Unrecognized frozen combiner state {name}")


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Exact F1 argmax over {0, each distinct DEV probability, 1} in O(n log n)."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0 or np.unique(y).size < 2:
        raise ValueError("DEV labels must contain both classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite in [0,1]")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    unique, starts = np.unique(sorted_p, return_index=True)
    positives_before = np.r_[0, np.cumsum(sorted_y, dtype=np.int64)][starts]
    tp = int(y.sum()) - positives_before
    predicted = len(y) - starts
    f1 = 2 * tp / np.maximum(predicted + int(y.sum()), 1)
    candidates = list(zip(unique.tolist(), f1.tolist()))
    candidates.append((0.0, 2 * int(y.sum()) / (len(y) + int(y.sum()))))
    candidates.append((1.0, float(f1[unique == 1][0]) if np.any(unique == 1) else 0.0))
    return float(min(candidates, key=lambda item: (-item[1], item[0]))[0])


def _neural(root: Path, split: str, manifest: dict[str, Any], manifest_hash: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if split not in {"dev", "test"}:
        raise ValueError(split)
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    expected_count = manifest["official_split_counts"][split]
    for seed in SEEDS:
        checkpoint, fit = _check_completed_fit(root, seed, manifest_hash, manifest)
        if split == "dev":
            path = checkpoint.parent.parent / "dev_predictions.csv"
            expected_sha = fit["dev_prediction_sha256"]
        else:
            path = checkpoint.parent.parent / "test_predictions_posthoc.csv"
            provenance_path = checkpoint.parent.parent / "safe_test_inference_provenance.json"
            if not provenance_path.is_file():
                raise RuntimeError(f"Seed {seed} has no completed TEST inference provenance")
            inference = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (inference.get("status") != "complete" or inference.get("seed") != seed
                    or inference.get("data_manifest_sha256") != manifest_hash
                    or inference.get("checkpoint_sha256") != sha(checkpoint)
                    or inference.get("dev_prediction_sha256") != fit["dev_prediction_sha256"]):
                raise RuntimeError(f"Seed {seed} TEST inference provenance mismatch")
            expected_sha = inference["test_prediction_sha256"]
        if not path.is_file() or sha(path) != expected_sha:
            raise RuntimeError(f"Seed {seed} {split} prediction hash mismatch")
        frames.append(_identity(pd.read_csv(path), expected_count))
        hashes[str(seed)] = expected_sha
    for seed, frame in zip(SEEDS[1:], frames[1:], strict=True):
        _aligned(frames[0], frame, f"NuLKT seed {seed}")
    output = frames[0][list(IDENTITY)].copy()
    output["probability"] = _sigmoid(np.mean([_logit(frame.probability.to_numpy()) for frame in frames], axis=0))
    return output, hashes


def _boosted(root: Path, split: str, regime: str, expected_rows: int) -> tuple[pd.DataFrame, dict[str, str], str]:
    if split not in {"dev", "test"}:
        raise ValueError(split)
    protocol = _protocol(root, regime)
    output = root / "results/raw_runs" / f"offline_safe_gbdt_{regime}"
    phase = "dev" if split == "dev" else "refit"
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for member in MEMBERS:
        frame, meta = _read_artifact(output, member, phase, protocol)
        if len(frame) != expected_rows:
            raise RuntimeError(f"Boosted member {member}/{phase} row count mismatch")
        frame = frame.sort_values("row_index", kind="stable").reset_index(drop=True)
        frames[member] = frame
        hashes[member] = meta["prediction_sha256"]
        if split == "test":
            _, selected = _read_artifact(output, member, "dev", protocol)
            if member.startswith("cat_") and meta["selected_iterations"] != selected["selected_iterations"]:
                raise RuntimeError(f"{member} refit tree count differs from DEV selection")
    if split == "test":
        best = [_read_artifact(output, f"lgb_{seed}", "dev", protocol)[1]["selected_iterations"] for seed in LGB_SEEDS]
        expected_iteration = int(np.median(best))
        for seed in LGB_SEEDS:
            if _read_artifact(output, f"lgb_{seed}", "refit", protocol)[1]["selected_iterations"] != expected_iteration:
                raise RuntimeError("LightGBM refit tree count differs from DEV selection")
    first = frames[MEMBERS[0]]
    for member in MEMBERS[1:]:
        if not first[list(IDENTITY)].equals(frames[member][list(IDENTITY)]):
            raise RuntimeError(f"Boosted member identity mismatch: {member}")
    light = np.mean([frames[f"lgb_{seed}"].logit.to_numpy() for seed in LGB_SEEDS], axis=0)
    cat = np.mean([frames[f"cat_{seed}"].logit.to_numpy() for seed in CAT_SEEDS], axis=0)
    result = first[list(IDENTITY)].copy()
    result["probability"] = _sigmoid(0.4 * light + 0.6 * cat)
    return _identity(result, expected_rows), hashes, protocol["protocol_sha256"]


def select_dev(root: Path, regime: str) -> dict[str, Any]:
    output = root / "results/offline_safe_evaluation" / regime
    selection_path = output / "dev_selection.json"
    if selection_path.exists():
        raise RuntimeError("DEV selection already exists; it is immutable for this evaluation")
    manifest, manifest_hash = _check_data(root)
    n = manifest["official_split_counts"]["dev"]
    neural, neural_hashes = _neural(root, "dev", manifest, manifest_hash)
    boosted, boosted_hashes, boosted_protocol = _boosted(root, "dev", regime, n)
    _aligned(neural, boosted, "boosted DEV arm")
    dev = neural[list(IDENTITY)].copy()
    dev["probability_neural"] = neural.probability.to_numpy()
    dev["probability_boosted"] = boosted.probability.to_numpy()
    y = dev.label.to_numpy(dtype=np.int8)
    combiners = build_combiners()
    expected_candidates = {
        "neural_only", "boosted_only", "probability_blend", "logit_blend",
        "rank_blend", "logistic_stack", "logistic_stack_interactions",
    }
    if {combiner.name for combiner in combiners} != expected_candidates or len(combiners) != 7:
        raise RuntimeError("Original seven-candidate combiner family changed")
    cv_table = cross_validate(combiners, dev, n_splits=5, seed=42)
    selected_name, _ = select_combiner(cv_table)  # references remain eligible
    selected = next(candidate for candidate in combiners if candidate.name == selected_name)
    oof, fold_of_row = grouped_oof_predictions(selected, dev, n_splits=5)
    for fold in range(5):
        train_learners = set(dev.loc[fold_of_row != fold, "learner_id"])
        heldout_learners = set(dev.loc[fold_of_row == fold, "learner_id"])
        if train_learners & heldout_learners:
            raise RuntimeError("A learner appears on both sides of a DEV fold")
    threshold = f1_threshold(y, oof)
    fitted_state = selected.fit(dev, y)
    frozen = _freeze_state(selected_name, fitted_state)
    fitted = _apply_frozen(frozen, dev)
    original_fitted = selected.apply(fitted_state, dev)
    if not np.allclose(fitted, original_fitted, rtol=0, atol=1e-9):
        raise RuntimeError("Serialized combiner differs from original implementation")
    prediction = neural[list(IDENTITY)].copy()
    prediction["probability_neural"] = neural.probability.to_numpy()
    prediction["probability_boosted"] = boosted.probability.to_numpy()
    prediction["probability_oof"] = oof
    prediction["probability_fitted"] = fitted
    prediction["dev_fold"] = fold_of_row
    prediction_path = output / "dev_predictions.csv"
    _atomic_csv(prediction_path, prediction)
    selection = {
        "status": "selected", "regime": regime, "selection_partition": "official DEV only",
        "candidate_family": "original seven rules (two references, probability/logit/rank blends, two logistic stacks)",
        "selection_method": "highest mean AUROC over five learner-grouped DEV folds",
        "boosted_only_eligible": True, "neural_only_eligible": True,
        "ensemble_source_sha256": sha(root / "src/evaluation/ensemble.py"),
        "neural_seeds_required": list(SEEDS),
        "boosted_members_required": list(MEMBERS),
        "selected_combiner": selected_name, "frozen_combiner": frozen,
        "selected_is_combination": selected_name not in {"neural_only", "boosted_only"},
        "cross_validation": cv_table.to_dict(orient="records"),
        "threshold_strategy": "maximum grouped-OOF DEV F1; lowest threshold on ties",
        "threshold": threshold,
        "dev_oof_metrics": binary_classification_metrics(y, oof, threshold),
        "dev_fitted_metrics_descriptive": binary_classification_metrics(y, fitted, threshold),
        "dev_rows": n, "data_manifest_sha256": manifest_hash,
        "boosted_protocol_sha256": boosted_protocol,
        "neural_dev_prediction_sha256": neural_hashes,
        "boosted_member_dev_prediction_sha256": boosted_hashes,
        "dev_prediction_sha256": sha(prediction_path),
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "prior_test_inspection": True,
    }
    _atomic_json(selection_path, selection)
    return selection


def evaluate_test(root: Path, regime: str, n_bootstrap: int) -> dict[str, Any]:
    output = root / "results/offline_safe_evaluation" / regime
    selection_path = output / "dev_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("Select and serialize DEV blend/threshold before TEST evaluation")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selected" or selection.get("regime") != regime:
        raise RuntimeError("Invalid DEV selection")
    if selection.get("ensemble_source_sha256") != sha(root / "src/evaluation/ensemble.py"):
        raise RuntimeError("Combiner implementation changed after DEV selection")
    dev_path = output / "dev_predictions.csv"
    if not dev_path.is_file() or sha(dev_path) != selection.get("dev_prediction_sha256"):
        raise RuntimeError("DEV selection prediction artifact changed")
    manifest, manifest_hash = _check_data(root)
    if manifest_hash != selection["data_manifest_sha256"]:
        raise RuntimeError("Corpus changed after DEV selection")
    n = manifest["official_split_counts"]["test"]
    neural, neural_hashes = _neural(root, "test", manifest, manifest_hash)
    boosted, boosted_hashes, boosted_protocol = _boosted(root, "test", regime, n)
    if boosted_protocol != selection["boosted_protocol_sha256"]:
        raise RuntimeError("Boosted protocol changed after DEV selection")
    _aligned(neural, boosted, "boosted TEST arm")
    prediction = neural[list(IDENTITY)].copy()
    prediction["probability_neural"] = neural.probability.to_numpy()
    prediction["probability_boosted"] = boosted.probability.to_numpy()
    threshold = float(selection["threshold"])
    combined = _apply_frozen(selection["frozen_combiner"], prediction)
    prediction["probability"] = combined
    y = prediction.label.to_numpy(dtype=np.int8)
    metrics = {
        name: binary_classification_metrics(y, prediction[column].to_numpy(), threshold)
        for name, column in (("neural", "probability_neural"),
                             ("boosted", "probability_boosted"), ("selected_blend", "probability"))
    }
    comparisons: dict[str, Any] = {}
    if n_bootstrap:
        for comparator, column in (("boosted", "probability_boosted"), ("neural", "probability_neural")):
            comparisons[comparator] = {
                metric: learner_clustered_bootstrap(
                    y, combined, prediction[column].to_numpy(), prediction.learner_id.to_numpy(),
                    metric=metric, n_bootstrap=n_bootstrap, seed=42,
                ) for metric in ("auroc", "auprc", "brier_score")
            }
    test_path = output / "test_predictions_posthoc.csv"
    report_path = output / "test_evaluation_posthoc.json"
    if test_path.exists() or report_path.exists():
        raise RuntimeError("TEST report already exists; preserve immutable first evaluation")
    _atomic_csv(test_path, prediction)
    report = {
        "status": "complete", "regime": regime, "evaluation_partition": "official TEST",
        "interpretation": "post hoc split-frozen replay; historical TEST previously inspected",
        "selection_sha256": sha(selection_path), "data_manifest_sha256": manifest_hash,
        "neural_test_prediction_sha256": neural_hashes,
        "boosted_member_test_prediction_sha256": boosted_hashes,
        "test_prediction_sha256": sha(test_path), "test_rows": n,
        "selected_combiner": selection["selected_combiner"],
        "frozen_combiner": selection["frozen_combiner"], "frozen_threshold": threshold,
        "metrics": metrics, "paired_learner_cluster_bootstrap": comparisons,
        "bootstrap_replicates": n_bootstrap,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select-dev", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--regime", choices=("benchmark", "deployment"), required=True)
    parser.add_argument("--bootstraps", type=int, default=200,
                        help="Learner-clustered paired replicates for TEST comparisons (0 disables)")
    args = parser.parse_args()
    if args.bootstraps != 0 and args.bootstraps < 100:
        parser.error("--bootstraps must be 0 or at least 100")
    result = select_dev(ROOT, args.regime) if args.select_dev else evaluate_test(ROOT, args.regime, args.bootstraps)
    print(json.dumps(result, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
