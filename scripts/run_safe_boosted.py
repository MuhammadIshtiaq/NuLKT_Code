"""Resumable, split-frozen official boosted benchmark.

Each model/phase writes an independently verifiable prediction artifact. Run
`--phase all` to resume missing phases, or select one with `--member`.
Nothing in this script changes the original benchmark implementation.
"""
from __future__ import annotations

import argparse
import gc
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
from scripts.benchmark_official import (  # noqa: E402
    CATEGORICAL_FEATURES, binary_classification_metrics, engineer_features,
    feature_columns, select_device,
)
from src.data.manifest import manifest_parts  # noqa: E402

LGB_SEEDS = (42, 123, 2024)
CAT_SEEDS = (42, 123)
MEMBERS = tuple([f"lgb_{s}" for s in LGB_SEEDS] + [f"cat_{s}" for s in CAT_SEEDS])
PROTOCOL_VERSION = "offline-safe-resumable-boosted-v1"
LGB_SETTINGS = dict(
    n_estimators=3000, learning_rate=0.04, num_leaves=511, objective="binary",
    n_jobs=-1, verbose=-1, colsample_bytree=0.6, subsample=0.8,
    subsample_freq=1, min_child_samples=300,
)
CAT_SETTINGS = dict(
    iterations=3000, learning_rate=0.06, depth=8, l2_leaf_reg=6.0,
    eval_metric="AUC", verbose=False, allow_writing_files=False,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _protocol(root: Path, regime: str) -> dict[str, Any]:
    manifest_path = root / "data/processed/en_es/manifest.json"
    preflight_path = root / "offline_safe_preflight.json"
    if not manifest_path.is_file() or not preflight_path.is_file():
        raise RuntimeError("Offline-safe manifest and preflight are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if manifest.get("offline_safe_protocol") != "split-frozen-v1" or preflight.get("status") != "PASS":
        raise RuntimeError("Offline-safe data/preflight are not valid")
    if preflight.get("safe_preprocessing_hash") != manifest.get("preprocessing_hash"):
        raise RuntimeError("Preflight and corpus preprocessing hashes disagree")
    parts = sorted((root / "data/processed/en_es").glob("part-*.parquet"))
    expected_hashes = manifest.get("offline_safe_part_sha256", {})
    if len(parts) != len(manifest.get("parts", [])):
        raise RuntimeError("Processed part count mismatch")
    for path in parts:
        if sha(path) != expected_hashes.get(path.name):
            raise RuntimeError(f"Processed part SHA mismatch: {path}")
    core = dict(
        version=PROTOCOL_VERSION, regime=regime,
        data_manifest_sha256=sha(manifest_path),
        official_split_counts=manifest["official_split_counts"],
        benchmark_source_sha256=sha(root / "scripts/benchmark_official.py"),
        features=feature_columns(regime),
        categorical_features=[n for n in CATEGORICAL_FEATURES if n in feature_columns(regime)],
        lightgbm_settings=LGB_SETTINGS, lightgbm_seeds=LGB_SEEDS,
        catboost_settings=CAT_SETTINGS, catboost_seeds=CAT_SEEDS,
        early_stopping_rounds=80, refit_on_train_and_dev=True,
        lightgbm_refit_iterations="median of all three dev-selected best_iteration values",
        catboost_refit_iterations="per-member dev-selected tree_count",
        blend_logit_weights={"lightgbm": 0.4, "catboost": 0.6},
        test_policy="post hoc; no model or weight selection on test",
    )
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "protocol_sha256": hashlib.sha256(encoded).hexdigest()}


def _output(root: Path, regime: str) -> Path:
    return root / "results/raw_runs" / f"offline_safe_gbdt_{regime}"


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _identity(block: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({
        "row_index": block.index.to_numpy(),
        "label": block["label"].to_numpy(),
        "learner_id": block["user_id_id"].to_numpy(),
    })
    if result.row_index.duplicated().any() or result.empty:
        raise RuntimeError("Missing or duplicate split row identity")
    return result


def _parts(frame: pd.DataFrame, regime: str) -> dict[str, pd.DataFrame]:
    frame = engineer_features(frame)
    names = list(dict.fromkeys([*feature_columns(regime), "label", "user_id_id"]))
    return {
        split: frame.loc[frame.official_split.astype(str) == split, names].copy()
        for split in ("train", "dev", "test")
    }


def _load_narrow_frame(root: Path, regime: str, phase: str) -> pd.DataFrame:
    """Load frozen columns, excluding TEST labels entirely during DEV fitting."""
    manifest = json.loads((root / "data/processed/en_es/manifest.json").read_text(encoding="utf-8"))
    columns = list(dict.fromkeys([*feature_columns(regime), "official_split", "label",
                                  "user_id_id", "row_order"]))
    filters = [("official_split", "in", ["train", "dev"])] if phase == "dev" else None
    frames = [pd.read_parquet(path, columns=columns, filters=filters)
              for path in manifest_parts(root, manifest)]
    frame = pd.concat(frames, ignore_index=True)
    if phase == "dev" and frame.official_split.astype(str).eq("test").any():
        raise RuntimeError("TEST row entered boosted DEV fit memory")
    if frame.row_order.duplicated().any():
        raise RuntimeError("Duplicate global row_order in frozen corpus")
    return frame.set_index("row_order", drop=True)


def _artifact_dir(output: Path, member: str, phase: str) -> Path:
    return output / "members" / member / phase


def _read_artifact(output: Path, member: str, phase: str, protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _artifact_dir(output, member, phase)
    meta_path, predictions_path = directory / "metadata.json", directory / "predictions.csv"
    if not meta_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Missing {member}/{phase} artifact")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("status") != "complete" or meta.get("member") != member
        or meta.get("phase") != phase or meta.get("protocol_sha256") != protocol["protocol_sha256"]
        or meta.get("prediction_sha256") != sha(predictions_path)
    ):
        raise RuntimeError(f"Invalid {member}/{phase} metadata or SHA")
    predictions = pd.read_csv(predictions_path)
    required = ["row_index", "label", "learner_id", "logit"]
    if list(predictions.columns) != required or len(predictions) != meta.get("rows"):
        raise RuntimeError(f"Invalid {member}/{phase} prediction shape")
    if predictions.row_index.duplicated().any() or not np.isfinite(predictions.logit).all():
        raise RuntimeError(f"Invalid {member}/{phase} row identity or logits")
    expected_split = "dev" if phase == "dev" else "test"
    if len(predictions) != protocol["official_split_counts"][expected_split]:
        raise RuntimeError(f"Invalid {member}/{phase} split row count")
    if not isinstance(meta.get("selected_iterations"), int) or meta["selected_iterations"] < 1:
        raise RuntimeError(f"Invalid {member}/{phase} selected iteration count")
    if phase == "dev" and "feature_importance_sha256" in meta:
        importance = directory / "feature_importance.csv"
        if meta.get("feature_importance_sha256") != sha(importance):
            raise RuntimeError("LightGBM feature importance SHA mismatch")
    return predictions, meta


def _write_artifact(
    output: Path, member: str, phase: str, protocol: dict[str, Any],
    predictions: pd.DataFrame, selected_iterations: int,
    importance: pd.DataFrame | None = None,
) -> None:
    directory = _artifact_dir(output, member, phase)
    directory.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "status": "complete", "member": member, "phase": phase,
        "protocol_sha256": protocol["protocol_sha256"],
        "selected_iterations": selected_iterations,
        "rows": len(predictions), "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    if importance is not None:
        path = directory / "feature_importance.csv"
        _atomic_csv(path, importance)
        meta["feature_importance_sha256"] = sha(path)
    path = directory / "predictions.csv"
    _atomic_csv(path, predictions)
    meta["prediction_sha256"] = sha(path)
    _atomic_text(directory / "metadata.json", json.dumps(meta, indent=2))
    _read_artifact(output, member, phase, protocol)


def _already_done(output: Path, member: str, phase: str, protocol: dict[str, Any]) -> bool:
    try:
        _read_artifact(output, member, phase, protocol)
        return True
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def _matrix(block: pd.DataFrame, protocol: dict[str, Any], family: str) -> pd.DataFrame:
    result = block[protocol["features"]].copy()
    for column in protocol["categorical_features"]:
        result[column] = (
            result[column].astype("category") if family == "lgb"
            else result[column].astype("int64").astype("str")
        )
    return result


def _fit_phase(
    root: Path, regime: str, member: str, phase: str, protocol: dict[str, Any],
    *, frame: pd.DataFrame | None = None, overrides: dict[str, Any] | None = None,
) -> None:
    if member not in MEMBERS or phase not in ("dev", "refit"):
        raise ValueError("Expected a configured member and dev/refit phase")
    output = _output(root, regime)
    if _already_done(output, member, phase, protocol):
        print(f"verified; skip {member}/{phase}", flush=True)
        return
    if frame is None:
        frame = _load_narrow_frame(root, regime, phase)
    parts = _parts(frame, regime)
    del frame
    gc.collect()
    if phase == "refit":
        if member.startswith("lgb"):
            iterations = [
                _read_artifact(output, f"lgb_{seed}", "dev", protocol)[1]["selected_iterations"]
                for seed in LGB_SEEDS
            ]
            selected = int(np.median(iterations))
        else:
            selected = _read_artifact(output, member, "dev", protocol)[1]["selected_iterations"]
        fit = pd.concat([parts["train"], parts["dev"]], ignore_index=True)
        target = parts["test"]
    else:
        selected = 0
        fit, target = parts["train"], parts["dev"]
    seed = int(member.rsplit("_", 1)[1])
    family = member.split("_", 1)[0]
    importance = None
    print(f"fit {member}/{phase}: train={len(fit)} predict={len(target)}", flush=True)
    if family == "lgb":
        import lightgbm as lgb
        settings = {**LGB_SETTINGS, **(overrides or {}), "random_state": seed}
        if phase == "refit":
            settings["n_estimators"] = selected
        model = lgb.LGBMClassifier(**settings)
        x_fit = _matrix(fit, protocol, family)
        if phase == "dev":
            model.fit(
                x_fit, fit["label"],
                eval_set=[(_matrix(target, protocol, family), target["label"])],
                eval_metric="auc", categorical_feature=protocol["categorical_features"],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
            selected = int(model.best_iteration_ or settings["n_estimators"])
            if member == f"lgb_{LGB_SEEDS[0]}":
                importance = pd.DataFrame({
                    "feature": protocol["features"],
                    "gain": model.booster_.feature_importance("gain"),
                }).sort_values("gain", ascending=False)
        else:
            model.fit(x_fit, fit["label"], categorical_feature=protocol["categorical_features"])
        probability = model.predict_proba(_matrix(target, protocol, family))[:, 1]
    else:
        from catboost import CatBoostClassifier, Pool
        cuda = select_device().type == "cuda"
        accelerator = (
            {"task_type": "GPU", "devices": "0", "gpu_ram_part": 0.7}
            if cuda else {"task_type": "CPU", "thread_count": -1}
        )
        settings = {**CAT_SETTINGS, **(overrides or {}), "random_seed": seed, **accelerator}
        settings["iterations"] = selected if phase == "refit" else settings["iterations"]
        settings["early_stopping_rounds"] = 80 if phase == "dev" else None
        model = CatBoostClassifier(**settings)
        fit_pool = Pool(_matrix(fit, protocol, family), fit["label"],
                        cat_features=protocol["categorical_features"])
        if phase == "dev":
            model.fit(
                fit_pool,
                eval_set=Pool(_matrix(target, protocol, family), target["label"],
                              cat_features=protocol["categorical_features"]),
                use_best_model=True,
            )
            selected = int(model.tree_count_)
        else:
            model.fit(fit_pool)
        probability = model.predict_proba(
            Pool(_matrix(target, protocol, family),
                 cat_features=protocol["categorical_features"])
        )[:, 1]
    predictions = _identity(target)
    predictions["logit"] = _logit(probability)
    _write_artifact(output, member, phase, protocol, predictions, selected, importance)
    print(f"complete {member}/{phase}: selected_iterations={selected}", flush=True)


def _assemble(root: Path, regime: str, protocol: dict[str, Any]) -> dict[str, Any]:
    output = _output(root, regime)
    by_phase: dict[str, dict[str, pd.DataFrame]] = {}
    metas: dict[str, dict[str, Any]] = {}
    for phase in ("dev", "refit"):
        by_phase[phase] = {}
        for member in MEMBERS:
            frame, meta = _read_artifact(output, member, phase, protocol)
            by_phase[phase][member] = frame
            metas[f"{member}/{phase}"] = meta
    iterations = [metas[f"lgb_{seed}/dev"]["selected_iterations"] for seed in LGB_SEEDS]
    median_iteration = int(np.median(iterations))
    for member in MEMBERS:
        expected = median_iteration if member.startswith("lgb") else metas[f"{member}/dev"]["selected_iterations"]
        if metas[f"{member}/refit"]["selected_iterations"] != expected:
            raise RuntimeError(f"{member} refit iteration count disagrees with dev selection")
    prediction_outputs: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {}
    for phase, split in (("dev", "dev"), ("refit", "test")):
        first = by_phase[phase][MEMBERS[0]]
        identity = first[["row_index", "label", "learner_id"]].reset_index(drop=True)
        if not identity.row_index.is_monotonic_increasing:
            raise RuntimeError(f"{split} rows are not in original corpus order")
        for member in MEMBERS[1:]:
            other = by_phase[phase][member][["row_index", "label", "learner_id"]].reset_index(drop=True)
            if not identity.equals(other):
                raise RuntimeError(f"{split} rows/labels/learners misalign for {member}")
        light = np.mean([by_phase[phase][f"lgb_{s}"].logit.to_numpy() for s in LGB_SEEDS], axis=0)
        cat = np.mean([by_phase[phase][f"cat_{s}"].logit.to_numpy() for s in CAT_SEEDS], axis=0)
        identity["probability"] = _probability(0.4 * light + 0.6 * cat)
        prediction_outputs[split] = identity[["label", "probability", "learner_id", "row_index"]]
        metrics[split] = binary_classification_metrics(identity.label, identity.probability)
    importance_path = _artifact_dir(output, f"lgb_{LGB_SEEDS[0]}", "dev") / "feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.is_file() else pd.DataFrame(columns=["feature", "gain"])
    protocol_record = {
        "feature_regime": regime, "features": protocol["features"], "excluded_features": [],
        "feature_importance_status": "available" if importance_path.is_file() else "unavailable: earlier fitted member did not save this diagnostic; predictions unchanged",
        "lightgbm_selected_iterations": iterations,
        "catboost_selected_iterations": [
            metas[f"cat_{s}/dev"]["selected_iterations"] for s in CAT_SEEDS
        ],
        "ensemble_seeds": list(LGB_SEEDS), "refit_on_train_and_dev": True,
        "refit_policy": "dev-selected iteration counts; fixed-count train+dev refit",
        "boosted_family": "LightGBM(3) + CatBoost(2), 0.4/0.6",
        "catboost_accelerator": "GPU:0" if select_device().type == "cuda" else "CPU",
        "resumable_protocol_sha256": protocol["protocol_sha256"],
        "member_prediction_sha256": {
            key: meta["prediction_sha256"] for key, meta in metas.items()
        },
    }
    for split in ("dev", "test"):
        _atomic_csv(output / f"{split}_predictions.csv", prediction_outputs[split])
    _atomic_csv(output / "feature_importance.csv", importance)
    _atomic_text(output / "training_protocol.json", json.dumps(protocol_record, indent=2))
    provenance = {
        "status": "complete", "regime": regime, "root": str(root),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": protocol["data_manifest_sha256"],
        "source_sha256": protocol["benchmark_source_sha256"],
        "resumable_protocol_sha256": protocol["protocol_sha256"],
        "ensemble_seeds": list(LGB_SEEDS),
        "boosted_family": protocol_record["boosted_family"],
        "test_policy": protocol["test_policy"],
        "dev_prediction_sha256": sha(output / "dev_predictions.csv"),
        "test_prediction_sha256": sha(output / "test_predictions.csv"),
        "training_protocol_sha256": sha(output / "training_protocol.json"),
        "dev_metrics": metrics["dev"], "test_metrics_posthoc": metrics["test"],
    }
    _atomic_text(output / "safe_fit_provenance.json", json.dumps(provenance, indent=2, default=float))
    return {"status": "complete", "output": str(output), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("benchmark", "deployment"), required=True)
    parser.add_argument("--phase", choices=("status", "dev", "refit", "assemble", "all"), default="all")
    parser.add_argument("--member", choices=MEMBERS)
    args = parser.parse_args()
    if args.member and args.phase not in ("dev", "refit"):
        parser.error("--member is valid only with --phase dev or refit")
    protocol = _protocol(ROOT, args.regime)
    output = _output(ROOT, args.regime)
    if args.phase in ("refit", "assemble", "all"):
        selection_path = ROOT / "results/offline_safe_evaluation" / args.regime / "dev_selection.json"
        if not selection_path.is_file():
            raise RuntimeError(
                "Freeze the DEV-selected combiner and threshold before any TEST refit or assembly: "
                "run evaluate_safe_stack.py --select-dev after all boosted DEV members and neural DEV seeds"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (selection.get("status") != "selected" or selection.get("regime") != args.regime
                or selection.get("boosted_protocol_sha256") != protocol["protocol_sha256"]
                or selection.get("data_manifest_sha256") != protocol["data_manifest_sha256"]):
            raise RuntimeError("Frozen DEV selection does not match this boosted protocol and corpus")
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "resumable_protocol.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("protocol_sha256") != protocol["protocol_sha256"]:
            if any((output / "members").glob("*/dev/metadata.json")) or any(
                (output / "members").glob("*/refit/metadata.json")
            ):
                raise RuntimeError("Frozen protocol differs; use a separate output directory")
            _atomic_text(frozen_path, json.dumps(protocol, indent=2))
    else:
        _atomic_text(frozen_path, json.dumps(protocol, indent=2))
    if args.phase == "status":
        for phase in ("dev", "refit"):
            for member in MEMBERS:
                try:
                    _read_artifact(output, member, phase, protocol)
                    state = "verified"
                except FileNotFoundError:
                    state = "missing"
                except Exception as exc:
                    state = f"invalid: {exc}"
                print(f"{member}/{phase}: {state}")
        return
    phases = ("dev", "refit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "assemble":
            break
        for member in (args.member,) if args.member else MEMBERS:
            _fit_phase(ROOT, args.regime, member, phase, protocol)
    if args.phase in ("assemble", "all"):
        print(json.dumps(_assemble(ROOT, args.regime, protocol), indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
