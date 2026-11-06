"""Export experiment and analysis artifacts into the publication table contract.

``src.publication.tables`` reads a fixed set of paths and prints
"Experiment unavailable/failed" for anything absent. The experiment stages and
the N1-N8 analyses write to their own directories, so without this bridge a
completed run would still report most tables as missing.

The bridge only reshapes evidence that already exists. It never invents a row,
never fills a metric, and leaves a table genuinely unavailable when the
experiment behind it has not been run - which is the behaviour the quality-control
report depends on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.evaluation.metrics import binary_classification_metrics


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write(frame: pd.DataFrame, destination: Path) -> bool:
    if frame.empty:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return True


def export_dataset_statistics(results_root: Path) -> bool:
    """Materialize the dataset summary used by Table 1 and reproduction checks."""

    summary = _load_json(results_root / "preprocessing_summary.json")
    if not summary or summary.get("debug_subset") is True:
        return False
    required = ("interactions", "exercises", "learners")
    if any(summary.get(name) is None for name in required):
        return False

    # ``data_summary.json`` is the stable public contract. Preserve the complete
    # preprocessing provenance rather than copying only the fields shown in the
    # compact table.
    (results_root / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = [
        {"category": "corpus", "statistic": name, "value": int(summary[name])}
        for name in required
    ]
    rows.extend(
        {
            "category": "official partition",
            "statistic": str(split),
            "value": int(count),
        }
        for split, count in summary.get("official_split_counts", {}).items()
    )
    label_counts = summary.get("label_counts", {})
    rows.extend(
        {
            "category": "label",
            "statistic": "correct" if str(label) == "0" else "error",
            "value": int(count),
        }
        for label, count in sorted(label_counts.items())
    )
    total = sum(int(value) for value in label_counts.values())
    if total:
        rows.append(
            {
                "category": "label",
                "statistic": "error rate",
                "value": int(label_counts.get("1", 0)) / total,
            }
        )
    return _write(
        pd.DataFrame(rows), results_root / "publication" / "dataset_statistics.csv"
    )


def export_deep_kt_results(results_root: Path) -> bool:
    """Table 6 - neural and knowledge-tracing baselines on the full corpus."""

    source = results_root / "metrics" / "neural_baseline_results.csv"
    if not source.is_file():
        return False
    frame = pd.read_csv(source)
    if frame.empty:
        return False
    frame.insert(1, "family", frame["model"].map(_baseline_family))
    return _write(frame, results_root / "metrics" / "deep_kt_results.csv")


def _baseline_family(name: str) -> str:
    return {
        "mlp": "feed-forward", "deepfm": "feature interaction",
        "lstm": "recurrent", "gru": "recurrent",
        "dkt": "knowledge tracing", "dkvmn": "knowledge tracing",
        "sakt": "attentive knowledge tracing", "akt": "attentive knowledge tracing",
        "saint_plus": "attentive knowledge tracing",
        "hierarchical_transformer": "hierarchical transformer",
    }.get(str(name), "unclassified")


def export_proposed_results(results_root: Path) -> bool:
    """Table 7 - proposed-model results aggregated across all specified seeds."""

    source = results_root / "metrics" / "seed_results.csv"
    if not source.is_file():
        return False
    frame = pd.read_csv(source)
    if frame.empty:
        return False
    metrics = [
        name for name in ("auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "mcc", "brier_score", "ece")
        if name in frame.columns
    ]
    rows: list[dict[str, Any]] = []
    for split, block in frame.groupby("split", observed=True):
        record: dict[str, Any] = {
            "model": "NuLKT",
            "split": split,
            "seeds": int(block["seed"].nunique()),
            "seed_list": ", ".join(str(value) for value in sorted(block["seed"].unique())),
        }
        for name in metrics:
            record[f"{name}_mean"] = float(block[name].mean())
            # A single-seed standard deviation is undefined, not zero.
            record[f"{name}_sd"] = float(block[name].std(ddof=1)) if len(block) > 1 else None
        rows.append(record)
    return _write(pd.DataFrame(rows), results_root / "metrics" / "proposed_results.csv")


def export_main_results(results_root: Path) -> bool:
    """Canonical official-test comparison derived from retained predictions.

    The long-form contract is intentionally small and stable: every row is one
    measured metric for one frozen arm. Publication prose, tables, and claim
    validation can therefore consume the same evidence rather than transcribing
    values independently from several JSON files.
    """

    predictions_path = results_root / "benchmark" / "ensemble_test_predictions.csv"
    study_path = results_root / "benchmark" / "ensemble_study.json"
    if not predictions_path.is_file() or not study_path.is_file():
        return False
    predictions = pd.read_csv(predictions_path)
    required = {
        "label", "learner_id", "probability", "probability_boosted", "probability_neural"
    }
    missing = sorted(required.difference(predictions.columns))
    if missing or predictions.empty:
        return False
    study = _load_json(study_path)
    if not study:
        return False
    selected = str(study.get("selection", {}).get("selected_combiner", ""))
    if not selected:
        return False
    selected_result = next(
        (
            row
            for row in study.get("test_results", [])
            if str(row.get("combiner")) == selected
        ),
        None,
    )
    if not selected_result:
        return False

    arms = (
        (
            "NuLKT + gradient boosting (dev-selected stack)",
            "probability",
            "selected",
        ),
        ("Gradient boosting", "probability_boosted", "reference"),
        ("NuLKT", "probability_neural", "reference"),
    )
    metric_names = (
        "auroc", "auprc", "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "mcc", "log_loss", "brier_score", "ece", "calibration_slope",
        "calibration_intercept",
    )
    rows: list[dict[str, Any]] = []
    for model, probability_column, role in arms:
        metrics = binary_classification_metrics(
            predictions["label"], predictions[probability_column]
        )
        for metric in metric_names:
            value = metrics.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "model": model,
                    "role": role,
                    "partition": "official test",
                    "evaluation_unit": "token",
                    "metric": metric,
                    "mean": float(value),
                    "standard_deviation": None,
                    "n": int(len(predictions)),
                    "threshold": float(metrics["threshold"]),
                    "selection_protocol": (
                        f"{selected}; learner-grouped five-fold CV on official dev"
                        if role == "selected"
                        else "reference arm; not selected on test"
                    ),
                    "source": str(predictions_path.relative_to(results_root)),
                }
            )

    combined_auroc = next(
        row["mean"]
        for row in rows
        if row["role"] == "selected" and row["metric"] == "auroc"
    )
    recorded_auroc = float(selected_result["test_auroc"])
    if abs(combined_auroc - recorded_auroc) > 1e-10:
        raise RuntimeError(
            "ensemble_test_predictions.csv does not match the selected test result: "
            f"{combined_auroc} != {recorded_auroc}"
        )
    return _write(pd.DataFrame(rows), results_root / "metrics" / "main_results.csv")


def export_canonical_ensemble_study(results_root: Path) -> bool:
    """Export only the frozen two-arm ensemble used by the headline result.

    Exploratory diversity studies may contain superseded checkpoints. They stay
    available as diagnostics but are deliberately excluded from Table 18.
    """

    source = results_root / "benchmark" / "ensemble_study.json"
    if not source.is_file():
        return False
    study = json.loads(source.read_text(encoding="utf-8"))
    selected = study.get("selection", {}).get("selected_combiner")
    cv_by_name = {
        row.get("combiner"): row for row in study.get("cross_validation", [])
    }
    rows: list[dict[str, Any]] = []
    for test_row in study.get("test_results", []):
        name = test_row.get("combiner")
        cv = cv_by_name.get(name, {})
        rows.append(
            {
                "combiner": name,
                "description": test_row.get("description"),
                "selected": name == selected,
                "cv_auroc_mean": cv.get("cv_auroc_mean"),
                "cv_auroc_sd": cv.get("cv_auroc_sd"),
                "cv_folds": cv.get("folds"),
                "test_auroc": test_row.get("test_auroc"),
                "test_auprc": test_row.get("test_auprc"),
                "test_brier": test_row.get("test_brier"),
                "test_ece": test_row.get("test_ece"),
                "source": "benchmark/ensemble_study.json",
            }
        )
    if not rows or sum(bool(row["selected"]) for row in rows) != 1:
        raise ValueError("Canonical ensemble study must contain exactly one selected combiner")
    selected_row = next(row for row in rows if row["selected"])
    if selected_row["test_auroc"] is None:
        raise ValueError("Selected canonical ensemble has no protected-test AUROC")
    return _write(
        pd.DataFrame(rows), results_root / "metrics" / "canonical_ensemble_results.csv"
    )


def synchronize_published_study_self_row(results_root: Path) -> bool:
    """Keep this project's literature-table row tied to canonical predictions."""

    table_path = results_root / "published_studies.csv"
    study_path = results_root / "benchmark" / "ensemble_study.json"
    if not table_path.is_file() or not study_path.is_file():
        return False
    frame = pd.read_csv(table_path)
    required = {"Study", "Model", "Published score", "URL/reference"}
    if not required.issubset(frame.columns):
        raise ValueError(f"published_studies.csv lacks columns: {sorted(required - set(frame))}")
    own = frame["Study"].astype(str).isin(
        ["This work (NuLKT project)", "This work (EIP-KTFormer project)"]
    )
    if int(own.sum()) != 1:
        raise ValueError("published_studies.csv must contain exactly one 'This work' row")
    frame.loc[own, "Study"] = "This work (NuLKT project)"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    boosted = next(
        (row for row in study.get("test_results", []) if row.get("combiner") == "boosted_only"),
        None,
    )
    if boosted is None or boosted.get("test_auroc") is None:
        raise ValueError("canonical ensemble study has no boosted-only protected-test result")
    frame.loc[own, "Model"] = "LightGBM + CatBoost, frozen benchmark ensemble"
    frame.loc[own, "Published score"] = float(boosted["test_auroc"])
    frame.loc[own, "URL/reference"] = "results/benchmark/ensemble_study.json"
    if "Score provenance" not in frame:
        frame["Score provenance"] = "as reported by cited study"
    frame.loc[own, "Score provenance"] = "reproduced in this repository"
    if "reason for non-comparability" in frame:
        frame.loc[own, "reason for non-comparability"] = (
            "Dataverse v4.0 postdates the 2018 challenge release; report the separately "
            "frozen prompt-free control with cross-paper comparisons."
        )
    return _write(frame, table_path)


def export_statistical_tests(results_root: Path) -> bool:
    """Table 12 - flatten the statistical programme into a tabular record."""

    directory = results_root / "statistical_tests"
    rows: list[dict[str, Any]] = []

    programme = _load_json(directory / "statistical_programme.json")
    if programme:
        comparison = programme.get("multi_model_comparison", {})
        for entry in comparison.get("comparisons", []):
            rows.append(
                {
                    "comparison": f"NuLKT vs {entry.get('competitor')}",
                    "partition": programme.get("partition"),
                    "metric": entry.get("metric"),
                    "difference": entry.get("model_a_minus_b"),
                    "ci_lower": entry.get("ci_lower"),
                    "ci_upper": entry.get("ci_upper"),
                    "raw_p_value": entry.get("raw_p_value"),
                    "adjusted_p_value": entry.get("holm_adjusted_p_value"),
                    "significant_after_correction": entry.get("reject_null_after_correction"),
                    "method": entry.get("method"),
                    "correction": comparison.get("correction"),
                }
            )
        equivalence = programme.get("equivalence")
        if isinstance(equivalence, dict) and "mean_difference" in equivalence:
            rows.append(
                {
                    "comparison": "NuLKT vs best baseline (equivalence)",
                    "partition": "test",
                    "metric": "auroc",
                    "difference": equivalence.get("mean_difference"),
                    "ci_lower": equivalence.get("ci_lower"),
                    "ci_upper": equivalence.get("ci_upper"),
                    "raw_p_value": equivalence.get("p_value"),
                    "adjusted_p_value": None,
                    "significant_after_correction": None,
                    "method": f"{equivalence.get('method')} (margin {equivalence.get('margin')})",
                    "correction": "not applicable",
                }
            )

    paired = _load_json(directory / "statistical_tests.json")
    if paired:
        for entry in paired.get("comparisons", []):
            rows.append(
                {
                    "comparison": "NuLKT vs best baseline (paired test partition)",
                    "partition": "test",
                    "metric": entry.get("metric"),
                    "difference": entry.get("model_a_minus_b"),
                    "ci_lower": entry.get("ci_lower"),
                    "ci_upper": entry.get("ci_upper"),
                    "raw_p_value": entry.get("raw_p_value"),
                    "adjusted_p_value": None,
                    "significant_after_correction": None,
                    "method": entry.get("method"),
                    "correction": "not applicable",
                }
            )
    return _write(pd.DataFrame(rows), directory / "statistical_tests.csv")


def export_cold_start_results(results_root: Path) -> bool:
    """Table 14 - cold-start protocol comparison and warm-up curve."""

    payload = _load_json(results_root / "analysis" / "cold_start" / "cold_start_analysis.json")
    if not payload or payload.get("status") != "available":
        return False
    rows: list[dict[str, Any]] = []
    comparison = payload.get("protocol_comparison")
    if comparison:
        for key, label in (
            ("chronological_personalized", "Chronological personalized"),
            ("learner_disjoint_cold_start", "Learner-disjoint cold start"),
        ):
            block = comparison.get(key, {})
            rows.append(
                {
                    "protocol": label,
                    "history_bin": "all",
                    "n": block.get("n"),
                    "learners": block.get("learners"),
                    "auroc": block.get("auroc"),
                    "auprc": block.get("auprc"),
                    "brier": block.get("brier"),
                }
            )
    for entry in payload.get("warm_up_curve", []):
        upper = entry.get("history_upper")
        rows.append(
            {
                "protocol": "Learner-disjoint cold start",
                "history_bin": f"{int(entry['history_lower'])}-{'inf' if upper is None or upper == float('inf') else int(upper)}",
                "n": entry.get("n"),
                "learners": None,
                "auroc": entry.get("auroc"),
                "auprc": entry.get("auprc"),
                "brier": entry.get("brier"),
            }
        )
    return _write(pd.DataFrame(rows), results_root / "metrics" / "cold_start_results.csv")


def export_personalization_depth(results_root: Path) -> bool:
    """Table 15 - how much learner identity actually changes predictions."""

    payload = _load_json(results_root / "analysis" / "personalization" / "personalization_analysis.json")
    if not payload:
        return False
    summary = payload.get("intervention_summary", {})
    observed = summary.get("observed", {})
    rows: list[dict[str, Any]] = []
    for condition, label in (
        ("permuted_identity", "Learner identity permuted across learners"),
        ("anonymous_identity", "Learner identity removed (unseen-learner state)"),
    ):
        block = summary.get(condition)
        if not block:
            continue
        rows.append(
            {
                "intervention": label,
                "n": observed.get("n"),
                "auroc": block.get("auroc"),
                "auroc_loss_vs_observed": block.get("auroc_loss"),
                "auprc": block.get("auprc"),
                "auprc_loss_vs_observed": block.get("auprc_loss"),
                "mean_absolute_probability_change": block.get("mean_absolute_delta"),
                "p90_absolute_probability_change": block.get("p90_absolute_delta"),
                "share_moved_over_0_05": block.get("share_moved_over_0_05"),
            }
        )
    if rows:
        rows.insert(
            0,
            {
                "intervention": "Observed (true learner identity)",
                "n": observed.get("n"),
                "auroc": observed.get("auroc"),
                "auroc_loss_vs_observed": 0.0,
                "auprc": observed.get("auprc"),
                "auprc_loss_vs_observed": 0.0,
                "mean_absolute_probability_change": 0.0,
                "p90_absolute_probability_change": 0.0,
                "share_moved_over_0_05": 0.0,
            },
        )
    return _write(pd.DataFrame(rows), results_root / "metrics" / "personalization_depth_results.csv")


def export_subgroup_results(results_root: Path) -> bool:
    """Table 16 - calibration and robustness across reported subgroups."""

    payload = _load_json(results_root / "analysis" / "subgroup_calibration" / "subgroup_calibration.json")
    if not payload:
        return False
    rows: list[dict[str, Any]] = []
    for attribute, result in payload.get("by_attribute", {}).items():
        if result.get("status") != "tested":
            continue
        for entry in result.get("subgroups", []):
            rows.append(
                {
                    "attribute": attribute,
                    "subgroup": entry.get("subgroup"),
                    "n": entry.get("n"),
                    "observed_error_rate": entry.get("observed_rate"),
                    "mean_predicted": entry.get("mean_predicted"),
                    "ece_global_temperature": entry.get("global_temperature_ece"),
                    "max_calibration_error": entry.get("global_temperature_max_error"),
                    "ece_group_temperature": entry.get("group_temperature_ece"),
                    "ece_reduction_from_group_temperature": entry.get("ece_reduction_from_group_temperature"),
                }
            )
    return _write(pd.DataFrame(rows), results_root / "metrics" / "subgroup_results.csv")


def export_selective_prediction(results_root: Path) -> bool:
    """Supplementary - uncertainty quality and the teacher review budget."""

    payload = _load_json(results_root / "analysis" / "selective_prediction" / "selective_prediction.json")
    if not payload:
        return False
    rows: list[dict[str, Any]] = []
    for name, block in payload.get("uncertainty_comparison", {}).get("summary", {}).items():
        rows.append(
            {
                "uncertainty_signal": name,
                "aurc": block.get("aurc"),
                "normalized_skill": block.get("normalized_skill"),
                "selective_accuracy_at_80_coverage": block.get("selective_accuracy_at_80_coverage"),
                "selective_error_at_80_coverage": block.get("selective_error_at_80_coverage"),
            }
        )
    written = _write(pd.DataFrame(rows), results_root / "metrics" / "uncertainty_quality_results.csv")
    budget = pd.DataFrame(payload.get("teacher_budget", []))
    return _write(budget, results_root / "metrics" / "teacher_budget_results.csv") or written


def export_forgetting_results(results_root: Path) -> bool:
    """Supplementary - identifiable temporal timescales and retention-law ranking."""

    payload = _load_json(results_root / "analysis" / "forgetting" / "forgetting_analysis.json")
    if not payload:
        return False
    kernel = payload.get("learned_kernel", {})
    comparison = payload.get("law_comparison", {})
    rows = [
        {
            "quantity": "Identifiable attention heads",
            "value": kernel.get("heads_identifiable"),
            "detail": f"{kernel.get('heads_at_initialization')} of {kernel.get('heads_total')} heads remained at initialization and are excluded",
        },
        {
            "quantity": "Median attention half-life (hours)",
            "value": kernel.get("half_life_hours_median"),
            "detail": f"range {kernel.get('half_life_hours_min')} to {kernel.get('half_life_hours_max')}",
        },
        {
            "quantity": "Half-life spread ratio",
            "value": kernel.get("half_life_spread_ratio"),
            "detail": "multi-timescale" if kernel.get("multi_timescale") else "single timescale",
        },
        {
            "quantity": "Best-fitting retention law",
            "value": comparison.get("best_law"),
            "detail": f"delta AIC to runner-up {comparison.get('delta_aic_to_runner_up')}; below 2 does not distinguish the laws",
        },
    ]
    return _write(pd.DataFrame(rows), results_root / "metrics" / "forgetting_results.csv")


EXPORTERS: dict[str, Callable[[Path], bool]] = {
    "dataset_statistics": export_dataset_statistics,
    "main_results": export_main_results,
    "canonical_ensemble": export_canonical_ensemble_study,
    "published_self_row": synchronize_published_study_self_row,
    "deep_kt_results": export_deep_kt_results,
    "proposed_results": export_proposed_results,
    "statistical_tests": export_statistical_tests,
    "cold_start_results": export_cold_start_results,
    "personalization_depth_results": export_personalization_depth,
    "subgroup_results": export_subgroup_results,
    "selective_prediction": export_selective_prediction,
    "forgetting_results": export_forgetting_results,
}


def export_publication_metrics(results_root: Path) -> dict[str, str]:
    """Run every exporter, recording which publication sources became available."""

    status: dict[str, str] = {}
    for name, exporter in EXPORTERS.items():
        try:
            status[name] = "written" if exporter(results_root) else "unavailable"
        except Exception as exc:  # a broken exporter must not hide the others
            status[name] = f"failed: {type(exc).__name__}: {exc}"
    destination = results_root / "publication" / "bridge_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
