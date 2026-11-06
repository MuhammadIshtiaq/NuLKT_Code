"""Generate the required publication tables from machine-readable sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class TableSpec:
    number: int
    slug: str
    title: str
    source: str


TABLES = [
    TableSpec(1, "dataset_statistics", "Dataset statistics", "publication/dataset_statistics.csv"),
    TableSpec(2, "feature_definitions", "Feature definitions", "publication/feature_definitions.csv"),
    TableSpec(3, "model_architecture_hyperparameters", "Model architecture and hyperparameters", "publication/model_architecture_hyperparameters.csv"),
    TableSpec(4, "published_study_comparison", "Published-study comparison", "published_studies.csv"),
    TableSpec(5, "classical_ml_results", "Classical machine-learning validation results", "metrics/classical_validation_results.csv"),
    TableSpec(6, "deep_kt_results", "Deep-learning and knowledge-tracing baseline results", "metrics/deep_kt_results.csv"),
    TableSpec(7, "proposed_model_results", "Main official-partition results", "metrics/main_results.csv"),
    TableSpec(8, "ablation_study", "Five-seed matched-budget ablation study", "metrics/matched_ablation_results.csv"),
    TableSpec(9, "incremental_components", "Incremental component study", "metrics/incremental_results.csv"),
    TableSpec(10, "optimizer_scheduler", "Optimizer and scheduler analysis", "metrics/optimizer_results.csv"),
    TableSpec(11, "calibration_analysis", "Calibration analysis", "metrics/calibration_results.csv"),
    TableSpec(12, "statistical_tests", "Statistical tests", "statistical_tests/statistical_tests.csv"),
    TableSpec(13, "complexity_efficiency", "Complexity and efficiency", "metrics/efficiency_results.csv"),
    TableSpec(14, "cold_start_results", "Cold-start results", "metrics/cold_start_results.csv"),
    TableSpec(15, "personalization_depth", "Personalization-depth results", "metrics/personalization_depth_results.csv"),
    TableSpec(16, "robustness_subgroups", "Robustness and subgroup analysis", "metrics/subgroup_results.csv"),
    TableSpec(17, "exercise_level_sensitivity", "Exercise-level target sensitivity", "metrics/exercise_level_results.csv"),
    TableSpec(18, "ensemble_diversity", "Canonical two-arm ensemble study", "metrics/canonical_ensemble_results.csv"),
    TableSpec(19, "prompt_free_control", "Prompt-identity control", "metrics/prompt_free_control.csv"),
]


FEATURE_DEFINITIONS = [
    ("token", "categorical", "Target surface-token identity", "current prompt", "Available before response; train-fitted vocabulary"),
    ("lemma", "categorical", "Lemma slot (tied to surface token because SLAM has no lemma column)", "current prompt", "No invented lemma annotation"),
    ("part_of_speech", "categorical", "Universal POS annotation", "current prompt", "Train-fitted vocabulary"),
    ("morphology", "categorical", "Sorted morphological feature signature", "current prompt", "Train-fitted vocabulary"),
    ("dependency_label", "categorical", "Syntactic/context relation", "current prompt", "Train-fitted vocabulary"),
    ("exercise_format", "categorical", "Exercise presentation/response format", "current prompt", "Train-fitted vocabulary"),
    ("learner", "categorical", "Learner identity for personalized protocol", "historical identifier", "Unknown ID for cold-start learners"),
    ("prior_interactions", "numeric", "Number of earlier learner interactions", "interactions < t", "Shifted cumulative statistic"),
    ("prior_error_rate", "numeric", "Earlier learner error proportion", "interactions < t", "Current target excluded"),
    ("recent_20_error_rate", "numeric", "Error rate over at most 20 earlier exercises", "interactions < t", "Same-exercise outcomes excluded"),
    ("prior_token_exposures", "numeric", "Earlier exposures to the target token", "interactions < t", "Current exposure excluded"),
    ("prior_token_error_rate", "numeric", "Earlier target-token error proportion", "interactions < t", "Current target excluded"),
    ("elapsed_since_prior_exercise_seconds", "numeric", "Time since previous completed exercise", "timestamps < t", "Computed after chronological sorting"),
    ("response_time_seconds", "numeric", "Post-response latency retained for benchmark compatibility", "post-response", "Excluded from pre-response recommendation models"),
]


def _materialize_specification_sources(results_root: Path) -> None:
    publication = results_root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    feature_path = publication / "feature_definitions.csv"
    pd.DataFrame(FEATURE_DEFINITIONS, columns=["Feature", "Type", "Definition", "Information timing", "Leakage control"]).to_csv(feature_path, index=False)
    architecture_path = publication / "model_architecture_hyperparameters.csv"
    config_path = results_root.parent / "configs" / "proposed.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        rows = []
        for section, values in config.items():
            if not isinstance(values, dict):
                rows.append({"Section": "root", "Parameter": section, "Value": values, "Source": str(config_path)})
                continue
            for name, value in values.items():
                rows.append({"Section": section, "Parameter": name, "Value": json.dumps(value) if isinstance(value, (dict, list)) else value, "Source": str(config_path)})
        pd.DataFrame(rows).to_csv(architecture_path, index=False)


def _read_source(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame([{"Status": "Experiment unavailable/failed", "Evidence": f"Source not found: {path.name}"}]), "unavailable"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame([{"Status": "Experiment unavailable/failed", "Evidence": f"Unreadable source: {exc}"}]), "unavailable"
    if frame.empty:
        return pd.DataFrame([{"Status": "Experiment unavailable/failed", "Evidence": "Source contains no result rows"}]), "unavailable"
    return frame, "available"


def generate_tables(results_root: Path) -> dict:
    _materialize_specification_sources(results_root)
    output = results_root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict] = {}
    workbook_path = output / "publication_tables.xlsx"
    excel_available = True
    try:
        writer = pd.ExcelWriter(workbook_path, engine="openpyxl")
    except Exception:
        writer = None
        excel_available = False
    try:
        for spec in TABLES:
            source = results_root / spec.source
            frame, status = _read_source(source)
            stem = output / f"table_{spec.number:02d}_{spec.slug}"
            frame.to_csv(stem.with_suffix(".csv"), index=False)
            latex = frame.to_latex(index=False, escape=True, caption=spec.title, label=f"tab:{spec.slug}", na_rep="NR")
            stem.with_suffix(".tex").write_text(latex, encoding="utf-8")
            if writer is not None:
                frame.to_excel(writer, sheet_name=f"T{spec.number}_{spec.slug}"[:31], index=False)
            registry[str(spec.number)] = {"title": spec.title, "status": status, "source": str(source), "csv": str(stem.with_suffix('.csv')), "latex": str(stem.with_suffix('.tex'))}
    finally:
        if writer is not None:
            writer.close()
    registry["xlsx"] = {"status": "available" if excel_available else "unavailable", "path": str(workbook_path) if excel_available else None}
    (output / "table_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    generate_tables(Path("results"))
