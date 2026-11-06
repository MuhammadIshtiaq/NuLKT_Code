"""Official SLAM benchmark protocol: comparability with the published leaderboard.

Every published en_es result is measured on the shared task's own train/dev/test
partition. The project's main protocol is a per-learner chronological split, which
answers a different question and cannot be compared against those numbers in
either direction. This module adds the missing benchmark arm.

Two feature regimes are kept strictly separate, because they answer different
retrospective questions:

``deployment``
    Internal name for the response-time-excluded retrospective sensitivity.
    Response latency is unavailable by construction, but the released token
    and linguistic fields can depend on a closest reference answer chosen
    after the learner's response. This regime is not validated for prospective
    instructional use.

``benchmark``
    Adds ``response_time_seconds``, which the shared task supplies as part of the
    exercise record and which competing systems therefore had access to. It is
    included here for retrospective shared-task comparability; prior inspection
    of published TEST labels means revised TEST estimates remain post hoc.

The ensemble combines sequential and gradient-boosted models. Combination
rules and thresholds are selected on development data before corrected TEST
inference; the published TEST labels were inspected during earlier work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.torch_dataset import load_processed_frame
from src.evaluation.metrics import binary_classification_metrics
from src.utils.device import select_device

# Past-only tabular features. Identity and context columns are categorical codes
# fitted on the official training partition only.
DEPLOYMENT_FEATURES: list[str] = [
    "prior_interactions",
    "prior_error_rate",
    "recent_20_error_rate",
    "prior_token_exposures",
    "prior_token_error_rate",
    "log1p_elapsed_seconds",
    "part_of_speech_id",
    "morphology_id",
    "exercise_format_id",
    "token_id",
    "dependency_label_id",
    "session_type_id",
    "client_id",
    "session_position",
    "token_index",
    "user_id_id",
]
BENCHMARK_ONLY_FEATURES: list[str] = ["response_time_seconds"]
CATEGORICAL_FEATURES: list[str] = [
    "part_of_speech_id", "morphology_id", "exercise_format_id", "token_id",
    "dependency_label_id", "session_type_id", "client_id", "user_id_id",
    "preceding_token_id", "following_token_id", "prompt_id",
]


# Derived features, computed from the corpus rather than stored in it. These
# follow the feature set the winning 2018 entry reported, and measured +0.0124
# AUROC on the official development partition over the stored columns alone.
ENGINEERED_FEATURES: list[str] = [
    "sentence_length",
    "preceding_token_id",
    "following_token_id",
    "token_relative_position",
    "prompt_id",
    "hours_since_token_last_seen",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sentence-context and token-revisit features in place-safe fashion.

    Every feature is either a property of the presented exercise, which is fully
    observable before the learner answers, or a function of strictly earlier
    interactions. Neither leaks an outcome. ``following_token_id`` is the surface
    form of the next word in the same sentence, not its label.
    """

    if all(name in frame.columns for name in ENGINEERED_FEATURES):
        return frame
    working = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable").copy()

    by_exercise = working.groupby("exercise_id", sort=False)
    working["sentence_length"] = by_exercise["token_index"].transform("size")
    working["preceding_token_id"] = by_exercise["token_id"].shift(1).fillna(0).astype(np.int32)
    working["following_token_id"] = by_exercise["token_id"].shift(-1).fillna(0).astype(np.int32)
    working["token_relative_position"] = working["token_index"] / working["sentence_length"].clip(lower=1)
    working["prompt_id"] = (
        working["prompt"].astype("category").cat.codes.astype(np.int32)
        if "prompt" in working.columns
        else working["exercise_id"].astype("category").cat.codes.astype(np.int32)
    )

    previous_time = working.groupby(["user_id", "token_id"], sort=False)["timestamp_days"].shift(1)
    # -1 marks a first encounter, which trees split on cleanly.
    working["hours_since_token_last_seen"] = ((working["timestamp_days"] - previous_time) * 24.0).fillna(-1.0)
    return working.loc[frame.index]


def feature_columns(regime: str, excluded_features: tuple[str, ...] = ()) -> list[str]:
    if regime == "deployment":
        columns = DEPLOYMENT_FEATURES + ENGINEERED_FEATURES
    elif regime == "benchmark":
        columns = DEPLOYMENT_FEATURES + ENGINEERED_FEATURES + BENCHMARK_ONLY_FEATURES
    else:
        raise ValueError(f"Unknown feature regime {regime!r}; use 'deployment' or 'benchmark'")
    unknown = sorted(set(excluded_features).difference(columns))
    if unknown:
        raise ValueError(f"Cannot exclude features absent from {regime!r}: {unknown}")
    excluded = set(excluded_features)
    selected = [name for name in columns if name not in excluded]
    if not selected:
        raise ValueError("Feature exclusion removed every benchmark feature")
    return selected


def _partitions(
    frame: pd.DataFrame, feature_names: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    mapped = frame["official_split"].astype(str)
    if feature_names is None:
        selected = frame.columns.tolist()
    else:
        selected = list(dict.fromkeys([*feature_names, "label", "user_id_id"]))
    return {
        name: frame.loc[mapped == name, selected].copy()
        for name in ("train", "dev", "test")
    }


def train_gradient_boosting(
    root: Path,
    *,
    regime: str = "benchmark",
    n_estimators: int = 3000,
    learning_rate: float = 0.04,
    num_leaves: int = 511,
    min_child_samples: int = 300,
    colsample_bytree: float = 0.6,
    seed: int = 42,
    frame: pd.DataFrame | None = None,
    refit_on_train_and_dev: bool = True,
    ensemble_seeds: tuple[int, ...] = (42, 123, 2024),
    use_catboost: bool = True,
    excluded_features: tuple[str, ...] = (),
    output_tag: str | None = None,
) -> dict[str, Any]:
    """Fit gradient boosting on the official training partition.

    Early stopping uses dev, which the shared task designates for model selection;
    test is never consulted during fitting. Defaults are the configuration selected
    by ``scripts/tune_boosted_arm.py`` on dev, which beat the previous settings by
    0.0024 AUROC against a measured seed standard deviation of 0.0005. Seed
    ensembling adds a further 0.0015 and is likewise adopted only because it
    exceeded that noise floor.
    """

    import lightgbm as lgb

    if frame is None:
        frame, _ = load_processed_frame(root)
    frame = engineer_features(frame)
    columns = feature_columns(regime, excluded_features)
    parts = _partitions(frame, columns)
    del frame
    gc.collect()
    categorical = [name for name in CATEGORICAL_FEATURES if name in columns]

    def matrix(block: pd.DataFrame) -> pd.DataFrame:
        data = block[columns].copy()
        for name in categorical:
            data[name] = data[name].astype("category")
        return data

    settings = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        objective="binary", n_jobs=-1, verbose=-1, colsample_bytree=colsample_bytree,
        subsample=0.8, subsample_freq=1, min_child_samples=min_child_samples,
    )
    seeds = tuple(ensemble_seeds) or (seed,)
    catboost_uses_gpu = select_device().type == "cuda"
    catboost_accelerator: dict[str, Any] = (
        {
            "task_type": "GPU",
            "devices": "0",
            # Leave headroom for the Windows display driver and other CUDA
            # consumers on the 8 GB laptop GPU.
            "gpu_ram_part": 0.7,
        }
        if catboost_uses_gpu
        else {"task_type": "CPU", "thread_count": -1}
    )
    print(
        f"  catboost accelerator={'GPU:0' if catboost_uses_gpu else 'CPU'}",
        flush=True,
    )

    def _logit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped))

    # CatBoost is included as a second family rather than more LightGBM seeds.
    # Measured on dev: CatBoost alone 0.85520 against LightGBM 0.85157, their logit
    # correlation 0.9317, and the blend 0.85850 - a gain of 0.0040 over LightGBM
    # seeds alone, against a seed standard deviation of 0.00047. XGBoost was tested
    # and rejected: it added 0.00009, which is inside noise.
    def _catboost_member(
        member: int,
        fit_frame: pd.DataFrame,
        predict_frames: list[pd.DataFrame],
        *,
        iterations: int = 3000,
        validation_frame: pd.DataFrame | None = None,
    ) -> tuple[list[np.ndarray], int]:
        from catboost import CatBoostClassifier, Pool

        def cat_matrix(block):
            data = block[columns].copy()
            for name in categorical:
                data[name] = data[name].astype("int64").astype("str")
            return data

        model = CatBoostClassifier(
            iterations=iterations, learning_rate=0.06, depth=8, l2_leaf_reg=6.0,
            eval_metric="AUC", random_seed=member, verbose=False,
            early_stopping_rounds=80 if validation_frame is not None else None,
            allow_writing_files=False,
            **catboost_accelerator,
        )
        fit_pool = Pool(
            cat_matrix(fit_frame), fit_frame["label"], cat_features=categorical
        )
        if validation_frame is None:
            model.fit(fit_pool)
        else:
            model.fit(
                fit_pool,
                eval_set=Pool(
                    cat_matrix(validation_frame),
                    validation_frame["label"],
                    cat_features=categorical,
                ),
                use_best_model=True,
            )
        predictions = [
            model.predict_proba(Pool(cat_matrix(block), cat_features=categorical))[:, 1]
            for block in predict_frames
        ]
        return predictions, int(model.tree_count_)

    dev_logits: list[np.ndarray] = []
    iteration_counts: list[int] = []
    dev_models = []
    for member in seeds:
        candidate = lgb.LGBMClassifier(**settings, random_state=member)
        candidate.fit(
            matrix(parts["train"]), parts["train"]["label"],
            eval_set=[(matrix(parts["dev"]), parts["dev"]["label"])],
            eval_metric="auc",
            categorical_feature=categorical,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        dev_models.append(candidate)
        iteration_counts.append(int(candidate.best_iteration_ or n_estimators))
        dev_logits.append(_logit(candidate.predict_proba(matrix(parts["dev"]))[:, 1]))
        print(f"  boosted seed={member} iterations={iteration_counts[-1]}", flush=True)
    # Preserve the reporting data before releasing the fitted development
    # models.  The train+dev refit below does not retain those model objects,
    # but feature importance must still come from the dev-selected estimator.
    feature_importance_gain = dev_models[0].booster_.feature_importance("gain")
    selected_iterations = int(np.median(iteration_counts))
    if refit_on_train_and_dev:
        dev_models.clear()
        del candidate
        gc.collect()

    # The shared task permits fitting the final model on train+dev before
    # predicting test, and the leading entries did exactly that. The iteration
    # count is the one already selected on dev, so no additional selection is
    # performed on the enlarged set; dev predictions still come from the
    # train-only model so that blend weights are never fitted on their own
    # training data.
    catboost_dev: list[np.ndarray] = []
    catboost_test: list[np.ndarray] = []
    catboost_iteration_counts: list[int] = []
    if use_catboost:
        combined_fit = (
            pd.concat([parts["train"], parts["dev"]], ignore_index=True)
            if refit_on_train_and_dev
            else parts["train"]
        )
        for member in seeds[:2]:
            try:
                print(f"  catboost seed={member} dev_fit=STARTING", flush=True)
                dev_outputs, selected_catboost_iterations = _catboost_member(
                    member,
                    parts["train"],
                    [parts["dev"]],
                    validation_frame=parts["dev"],
                )
                dev_probability, = dev_outputs
                print(f"  catboost seed={member} test_refit=STARTING", flush=True)
                test_outputs, _ = _catboost_member(
                    member,
                    combined_fit,
                    [parts["test"]],
                    iterations=selected_catboost_iterations,
                )
                test_probability, = test_outputs
            except Exception as exc:
                print(f"  catboost seed={member} unavailable: {type(exc).__name__}: {exc}", flush=True)
                break
            catboost_dev.append(_logit(dev_probability))
            catboost_test.append(_logit(test_probability))
            catboost_iteration_counts.append(selected_catboost_iterations)
            print(
                f"  catboost seed={member} fitted "
                f"selected_iterations={selected_catboost_iterations}",
                flush=True,
            )

    test_logits: list[np.ndarray] = []
    if refit_on_train_and_dev:
        combined = pd.concat([parts["train"], parts["dev"]], ignore_index=True)
        for member in seeds:
            final = lgb.LGBMClassifier(**{**settings, "n_estimators": selected_iterations}, random_state=member)
            final.fit(matrix(combined), combined["label"], categorical_feature=categorical)
            test_logits.append(_logit(final.predict_proba(matrix(parts["test"]))[:, 1]))
    else:
        test_logits = [_logit(m.predict_proba(matrix(parts["test"]))[:, 1]) for m in dev_models]

    def _mean_probability(logits: list[np.ndarray]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))

    # Blend the two families with the weights selected on dev by the diverse-
    # ensemble study; falls back to LightGBM alone if CatBoost was unavailable.
    if catboost_dev and catboost_test:
        light_weight, cat_weight = 0.4, 0.6
        dev_blend = light_weight * np.mean(dev_logits, axis=0) + cat_weight * np.mean(catboost_dev, axis=0)
        test_blend = light_weight * np.mean(test_logits, axis=0) + cat_weight * np.mean(catboost_test, axis=0)
        ensembled = {"dev": 1.0 / (1.0 + np.exp(-dev_blend)), "test": 1.0 / (1.0 + np.exp(-test_blend))}
        boosted_family = f"LightGBM({len(dev_logits)}) + CatBoost({len(catboost_dev)}), 0.4/0.6"
    else:
        ensembled = {"dev": _mean_probability(dev_logits), "test": _mean_probability(test_logits)}
        boosted_family = f"LightGBM({len(dev_logits)}) only"

    output_dir = root / "results" / "raw_runs" / (
        output_tag or f"official_gbdt_{regime}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, pd.DataFrame] = {}
    for split in ("dev", "test"):
        block = parts[split]
        frame_out = pd.DataFrame(
            {
                "label": block["label"].to_numpy(),
                "probability": ensembled[split],
                "learner_id": block["user_id_id"].to_numpy(),
                "row_index": block.index.to_numpy(),
            }
        )
        frame_out.to_csv(output_dir / f"{split}_predictions.csv", index=False)
        predictions[split] = frame_out

    importance = pd.DataFrame(
        {"feature": columns, "gain": feature_importance_gain}
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    training_protocol = {
        "feature_regime": regime,
        "features": columns,
        "excluded_features": list(excluded_features),
        "lightgbm_selected_iterations": iteration_counts,
        "catboost_selected_iterations": catboost_iteration_counts,
        "ensemble_seeds": list(seeds),
        "refit_on_train_and_dev": refit_on_train_and_dev,
        "refit_policy": "dev-selected iteration counts; fixed-count train+dev refit",
        "boosted_family": boosted_family,
        "catboost_accelerator": "GPU:0" if catboost_uses_gpu else "CPU",
    }
    (output_dir / "training_protocol.json").write_text(
        json.dumps(training_protocol, indent=2), encoding="utf-8"
    )

    return {
        "regime": regime,
        "features": columns,
        "excluded_features": list(excluded_features),
        "best_iteration": selected_iterations,
        "refit_on_train_and_dev": refit_on_train_and_dev,
        "ensemble_seeds": list(seeds),
        "lightgbm_iteration_counts": iteration_counts,
        "catboost_iteration_counts": catboost_iteration_counts,
        "boosted_family": boosted_family,
        "hyperparameters": {k: v for k, v in settings.items() if k not in {"n_jobs", "verbose", "objective"}},
        "dev_metrics": binary_classification_metrics(
            predictions["dev"]["label"], predictions["dev"]["probability"]
        ),
        "predictions": predictions,
        "output_dir": str(output_dir),
    }


def run_prompt_free_control(
    root: Path,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    skip_completed: bool = True,
) -> dict[str, Any]:
    """Refit the boosted arm without prompt identity under a frozen protocol.

    The control changes exactly one input, ``prompt_id``. Hyperparameters and
    train/dev/test handling are inherited unchanged from the official benchmark;
    no control-specific tuning is performed. The protocol record is written
    before fitting so the held-out test comparison cannot redefine the control.
    """

    from src.evaluation.statistics import learner_clustered_bootstrap

    benchmark_dir = root / "results" / "benchmark"
    metrics_dir = root / "results" / "metrics"
    full_dir = root / "results" / "raw_runs" / "official_gbdt_benchmark"
    full_protocol_path = full_dir / "training_protocol.json"
    if not full_protocol_path.is_file():
        raise FileNotFoundError(
            "The prompt-free control requires the completed full-feature protocol: "
            f"{full_protocol_path}"
        )
    full_protocol = json.loads(full_protocol_path.read_text(encoding="utf-8"))
    expected_catboost_members = len(full_protocol.get("catboost_selected_iterations", []))
    if expected_catboost_members not in {0, 2}:
        raise RuntimeError(
            "The full-feature benchmark has an incomplete CatBoost family "
            f"({expected_catboost_members}/2 members); refusing an unmatched control"
        )
    use_catboost = expected_catboost_members == 2
    control_dir = root / "results" / "raw_runs" / "official_gbdt_benchmark_prompt_free"
    destination = benchmark_dir / "prompt_free_control.json"
    protocol_path = benchmark_dir / "prompt_free_control_protocol.json"
    metric_path = metrics_dir / "prompt_free_control.csv"
    required_outputs = (
        control_dir / "dev_predictions.csv",
        control_dir / "test_predictions.csv",
        metric_path,
    )
    manifest_path = root / "data" / "processed" / "en_es" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_core = {
        "status": "frozen_before_fit",
        "protocol": "official SLAM train/dev/test",
        "track": "en_es",
        "feature_regime": "benchmark",
        "included_features": feature_columns("benchmark", ("prompt_id",)),
        "excluded_features": ["prompt_id"],
        "changed_variables": 1,
        "hyperparameter_tuning": "none; inherit the frozen official boosted-arm settings",
        "selection_partition": "official dev",
        "test_policy": "one evaluation after the exclusion and model settings are frozen",
        "preprocessing_hash": manifest.get("preprocessing_hash"),
        "seed": seed,
        "full_feature_training_protocol_sha256": hashlib.sha256(
            full_protocol_path.read_bytes()
        ).hexdigest(),
        "model_settings": {
            "lightgbm": {
                "n_estimators": 3000,
                "learning_rate": 0.04,
                "num_leaves": 511,
                "min_child_samples": 300,
                "colsample_bytree": 0.6,
                "subsample": 0.8,
                "subsample_freq": 1,
                "early_stopping_rounds": 80,
            },
            "catboost": {
                "enabled_to_match_full_feature_arm": use_catboost,
                "iterations": 3000,
                "learning_rate": 0.06,
                "depth": 8,
                "l2_leaf_reg": 6.0,
                "early_stopping_rounds": 80,
                "refit_iterations": "per-seed tree count selected on train-to-dev fit",
                "refit_validation": "none; fixed tree count on train+dev",
            },
            "ensemble_seeds": [42, 123, 2024],
            "family_logit_weights": (
                {"lightgbm": 0.4, "catboost": 0.6}
                if use_catboost
                else {"lightgbm": 1.0}
            ),
            "refit_on_train_and_dev": True,
        },
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    protocol = {
        **protocol_core,
        "protocol_hash": protocol_hash,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if (
        skip_completed
        and destination.is_file()
        and protocol_path.is_file()
        and all(path.is_file() for path in required_outputs)
    ):
        stored_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        base_matches = (
            stored_protocol.get("preprocessing_hash") == protocol_core["preprocessing_hash"]
            and stored_protocol.get("excluded_features") == ["prompt_id"]
            and stored_protocol.get("changed_variables") == 1
            and stored_protocol.get("feature_regime") == "benchmark"
            and stored_protocol.get("seed") == seed
            and stored_protocol.get("protocol_hash") in (None, protocol_hash)
        )
        if base_matches:
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
                comparison = payload["prompt_free_minus_full_test_auroc"]
                comparison_ok = all(
                    key in comparison
                    for key in ("model_a_minus_b", "ci_lower", "ci_upper")
                )
                family_ok = (
                    len(
                        payload.get("model_metadata", {}).get(
                            "catboost_iteration_counts", []
                        )
                    )
                    == expected_catboost_members
                )
                metric_frame = pd.read_csv(metric_path)
                expected_pairs = {
                    ("full_features", "dev"),
                    ("full_features", "test"),
                    ("prompt_free", "dev"),
                    ("prompt_free", "test"),
                }
                metric_ok = (
                    {"model", "split", "auroc"}.issubset(metric_frame.columns)
                    and expected_pairs.issubset(
                        set(zip(metric_frame["model"], metric_frame["split"]))
                    )
                )
                prediction_ok = all(
                    {"row_index", "label", "learner_id", "probability"}.issubset(
                        pd.read_csv(path, nrows=1).columns
                    )
                    and path.stat().st_size > 0
                    for path in required_outputs[:2]
                )
            except (KeyError, ValueError, pd.errors.ParserError, json.JSONDecodeError):
                comparison_ok = family_ok = metric_ok = prediction_ok = False
            if not (comparison_ok and family_ok and metric_ok and prediction_ok):
                base_matches = False
        if base_matches:
            # Early versions froze the same defaults but omitted their expanded
            # representation. Complete that provenance without refitting.
            protocol["frozen_at_utc"] = stored_protocol.get(
                "frozen_at_utc", protocol["frozen_at_utc"]
            )
            protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
            payload["protocol"] = protocol
            destination.write_text(
                json.dumps(payload, indent=2, default=float), encoding="utf-8"
            )
            return payload

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    control = train_gradient_boosting(
        root,
        regime="benchmark",
        seed=seed,
        use_catboost=use_catboost,
        excluded_features=("prompt_id",),
        output_tag="official_gbdt_benchmark_prompt_free",
    )
    actual_catboost_members = len(control["catboost_iteration_counts"])
    if actual_catboost_members != expected_catboost_members:
        raise RuntimeError(
            "Prompt-free control did not match the full-feature model family: "
            f"expected {expected_catboost_members} CatBoost members, got "
            f"{actual_catboost_members}. The stage can be safely retried."
        )

    rows: list[dict[str, Any]] = []
    aligned: dict[str, pd.DataFrame] = {}
    for split in ("dev", "test"):
        full_path = full_dir / f"{split}_predictions.csv"
        if not full_path.is_file():
            raise FileNotFoundError(f"Frozen full-feature predictions are absent: {full_path}")
        full = pd.read_csv(full_path)
        reduced = control["predictions"][split]
        joined = full.merge(
            reduced,
            on="row_index",
            suffixes=("_full", "_prompt_free"),
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(full) or len(joined) != len(reduced):
            raise RuntimeError(f"{split}: full and prompt-free predictions are not aligned")
        if not (joined["label_full"] == joined["label_prompt_free"]).all():
            raise RuntimeError(f"{split}: full and prompt-free predictions disagree on labels")
        if not (joined["learner_id_full"] == joined["learner_id_prompt_free"]).all():
            raise RuntimeError(f"{split}: full and prompt-free predictions disagree on learners")
        aligned[split] = joined
        for model, predictions in (("full_features", full), ("prompt_free", reduced)):
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "excluded_features": "" if model == "full_features" else "prompt_id",
                    **binary_classification_metrics(
                        predictions["label"], predictions["probability"]
                    ),
                }
            )

    test_joined = aligned["test"]
    comparison = learner_clustered_bootstrap(
        test_joined["label_prompt_free"],
        test_joined["probability_prompt_free"],
        test_joined["probability_full"],
        test_joined["learner_id_prompt_free"],
        metric="auroc",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    table = pd.DataFrame(rows)
    table.to_csv(metric_path, index=False)
    payload = {
        "status": "available",
        "protocol": protocol,
        "features": control["features"],
        "model_metadata": {
            "boosted_family": control["boosted_family"],
            "ensemble_seeds": control["ensemble_seeds"],
            "lightgbm_iteration_counts": control["lightgbm_iteration_counts"],
            "catboost_iteration_counts": control["catboost_iteration_counts"],
            "hyperparameters": control["hyperparameters"],
        },
        "full_feature_metrics": {
            split: next(
                row for row in rows if row["model"] == "full_features" and row["split"] == split
            )
            for split in ("dev", "test")
        },
        "prompt_free_metrics": {
            split: next(
                row for row in rows if row["model"] == "prompt_free" and row["split"] == split
            )
            for split in ("dev", "test")
        },
        "prompt_free_minus_full_test_auroc": comparison,
    }
    destination.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def fit_blend_weight(
    labels: np.ndarray,
    neural: np.ndarray,
    boosted: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Choose a convex blend weight on dev by maximizing AUROC.

    A logit-space average is used rather than a probability average because the
    two models are calibrated differently; averaging probabilities would let the
    more confident model dominate for reasons unrelated to its accuracy.
    """

    from sklearn.metrics import roc_auc_score

    grid = np.linspace(0.0, 1.0, 101) if grid is None else grid

    def logit(p: np.ndarray) -> np.ndarray:
        return np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

    ln, lb = logit(neural), logit(boosted)
    scores = [(float(w), float(roc_auc_score(labels, w * ln + (1 - w) * lb))) for w in grid]
    best_weight, best_score = max(scores, key=lambda item: item[1])
    return {
        "neural_weight": best_weight,
        "boosted_weight": 1.0 - best_weight,
        "dev_auroc": best_score,
        "dev_auroc_neural_only": float(roc_auc_score(labels, ln)),
        "dev_auroc_boosted_only": float(roc_auc_score(labels, lb)),
        "selection_partition": "official dev",
        "combination": "convex average in logit space",
    }


def apply_blend(neural: np.ndarray, boosted: np.ndarray, weight: float) -> np.ndarray:
    def logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped))

    combined = weight * logit(neural) + (1 - weight) * logit(boosted)
    return 1.0 / (1.0 + np.exp(-combined))


def ensemble_neural_seeds(root: Path, tag: str, seeds: tuple[int, ...], split: str) -> pd.DataFrame | None:
    """Average available seed predictions in logit space, aligned on row index."""

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        path = root / "results" / "raw_runs" / tag / f"seed_{seed}" / f"{split}_predictions.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    reference = frames[0]
    if "row_index" not in reference.columns:
        # Older artifacts lack the join key; alignment cannot be verified, so a
        # silent positional merge is refused.
        if any(len(frame) != len(reference) or not frame["label"].equals(reference["label"]) for frame in frames):
            raise RuntimeError("Seed predictions are not aligned and carry no row_index to align on")
        stacked = np.stack([frame["probability"].to_numpy(dtype=float) for frame in frames])
        merged = reference.copy()
    else:
        merged = reference[["row_index", "label", "learner_id"]].copy()
        columns = []
        for index, frame in enumerate(frames):
            renamed = frame[["row_index", "probability"]].rename(columns={"probability": f"p{index}"})
            merged = merged.merge(renamed, on="row_index", how="inner", validate="one_to_one")
            columns.append(f"p{index}")
        stacked = merged[columns].to_numpy(dtype=float).T
        merged = merged[["row_index", "label", "learner_id"]]

    clipped = np.clip(stacked, 1e-7, 1 - 1e-7)
    mean_logit = np.log(clipped / (1 - clipped)).mean(axis=0)
    merged["probability"] = 1.0 / (1.0 + np.exp(-mean_logit))
    merged["n_seeds"] = len(frames)
    return merged


def run(
    root: Path,
    *,
    regime: str = "benchmark",
    neural_tag: str = "official_eip_ktformer",
    seeds: tuple[int, ...] = (42, 123, 2024, 2025, 2026),
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the boosted arm, blend with any available neural arm, evaluate once."""

    output_dir = root / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    boosted = train_gradient_boosting(root, regime=regime, seed=seed)
    payload: dict[str, Any] = {
        "protocol": "official SLAM train/dev/test",
        "track": "en_es",
        "feature_regime": regime,
        "features": boosted["features"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gradient_boosting": {
            "best_iteration": boosted["best_iteration"],
            "dev": boosted["dev_metrics"],
        },
    }

    test_boosted = boosted["predictions"]["test"]
    payload["gradient_boosting"]["test"] = binary_classification_metrics(
        test_boosted["label"], test_boosted["probability"]
    )

    neural_dev = ensemble_neural_seeds(root, neural_tag, seeds, "dev")
    neural_test = ensemble_neural_seeds(root, neural_tag, seeds, "test")
    if neural_dev is None or neural_test is None:
        payload["ensemble"] = {
            "status": "unavailable",
            "reason": (
                f"no neural predictions under results/raw_runs/{neural_tag}. "
                "Run the official-protocol neural stage first."
            ),
        }
    else:
        dev_boosted = boosted["predictions"]["dev"]
        dev = neural_dev.merge(
            dev_boosted[["row_index", "probability"]].rename(columns={"probability": "boosted"}),
            on="row_index", how="inner", validate="one_to_one",
        )
        test = neural_test.merge(
            test_boosted[["row_index", "probability"]].rename(columns={"probability": "boosted"}),
            on="row_index", how="inner", validate="one_to_one",
        )
        blend = fit_blend_weight(
            dev["label"].to_numpy(), dev["probability"].to_numpy(), dev["boosted"].to_numpy()
        )
        test["probability_ensemble"] = apply_blend(
            test["probability"].to_numpy(), test["boosted"].to_numpy(), blend["neural_weight"]
        )
        test.to_csv(output_dir / "official_test_ensemble_predictions.csv", index=False)
        payload["neural"] = {
            "seeds_used": int(neural_test["n_seeds"].iloc[0]),
            "test": binary_classification_metrics(test["label"], test["probability"]),
        }
        payload["ensemble"] = {
            "status": "available",
            "blend": blend,
            "test": binary_classification_metrics(test["label"], test["probability_ensemble"]),
        }

    (output_dir / f"official_benchmark_{regime}.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--regime", choices=["benchmark", "deployment"], default="benchmark")
    arguments = parser.parse_args()
    payload = run(arguments.root.resolve(), regime=arguments.regime)
    print(json.dumps({key: value for key, value in payload.items() if key != "features"}, indent=2, default=float))


if __name__ == "__main__":
    raise SystemExit("Use scripts/reproduce_offline_safe.py; direct legacy benchmark mode does not enforce the split-frozen preflight.")
