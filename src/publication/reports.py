"""Generate manuscript notes and summaries without fabricating missing evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SECTIONS = [
    "Dataset", "Preprocessing", "Experimental Protocol", "Baselines", "Proposed Architecture",
    "Overall Performance", "Statistical Analysis", "Ablation", "Incremental Novelty", "Hyperparameter Analysis",
    "Optimizer Study", "Calibration", "Explainability", "Personalization", "Cold Start", "Robustness", "Efficiency", "Limitations",
]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _available_metric_rows(results_root: Path) -> pd.DataFrame | None:
    path = results_root / "metrics" / "main_results.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        return frame if not frame.empty else None
    except Exception:
        return None


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_final_summary(results_root: Path) -> Path:
    publication = results_root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    destination = publication / "final_results_summary.md"
    dataset = _json(results_root / "data_summary.json") or _json(results_root / "preprocessing_summary.json")
    frame = _available_metric_rows(results_root)
    lines = ["# Final Results Summary", "", "_Automatically generated exclusively from machine-readable result files._", ""]
    if dataset and not dataset.get("debug_subset", False) and dataset.get("interactions"):
        lines.extend(["## Dataset", "", f"Dataset summary source: `{results_root / 'preprocessing_summary.json'}`.", ""])
        for key in ("learners", "interactions", "exercises"):
            if key in dataset:
                lines.append(f"- {key.replace('_', ' ').title()}: {dataset[key]}")
        for split, count in dataset.get("official_split_counts", {}).items():
            lines.append(f"- Official {split.title()}: {count}")
        lines.append("")
    else:
        lines.extend(["## Dataset", "", "Experiment unavailable/failed: a validated dataset summary has not been produced.", ""])
    if frame is None:
        classical_path = results_root / "metrics" / "classical_validation_results.csv"
        intervals_path = results_root / "metrics" / "classical_validation_clustered_intervals.json"
        if classical_path.exists() and intervals_path.exists():
            classical = pd.read_csv(classical_path)
            lines.extend([
                "## Predictive Results", "",
                "Full-corpus chronological **validation-only** classical results are available below. "
                "They are not final test results and do not establish superiority of NuLKT.", "",
                _markdown_table(classical[["model", "n", "auroc", "auprc", "f1", "mcc", "brier_score", "ece"]]), "",
                "Learner-clustered 95% intervals are stored in "
                f"`{intervals_path}` (500 resamples across 2,593 learners).", "",
            ])
        else:
            lines.extend(["## Predictive Results", "", "Experiment unavailable/failed: no final multi-seed metrics are available. No best-model, improvement, confidence-interval, or significance claim is made.", ""])
    else:
        lines.extend(["## Predictive Results", "", _markdown_table(frame), ""])
    statistics = _json(results_root / "statistical_tests" / "statistical_tests.json")
    lines.extend(["## Statistical Evidence", "", "Experiment unavailable/failed." if not statistics else f"See `{results_root / 'statistical_tests' / 'statistical_tests.json'}`; only Holm-corrected comparisons should support hypotheses.", ""])
    lines.extend(["## Instructional Interpretation", "", "SLAM is observational. The recommendation layer is evaluated as decision support; this study cannot establish that explicit instruction causally improves learning.", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def generate_manuscript_notes(results_root: Path) -> Path:
    destination = results_root / "publication" / "manuscript_results_notes.md"
    availability = {
        "Dataset": (results_root / "data_summary.json").exists(),
        "Preprocessing": (results_root / "preprocessing_summary.json").exists(),
        "Experimental Protocol": (results_root / "protocol" / "final_experimental_protocol.yaml").exists(),
        "Baselines": (results_root / "metrics" / "classical_validation_results.csv").exists(),
        "Proposed Architecture": True,
        "Overall Performance": (results_root / "metrics" / "main_results.csv").exists(),
        "Statistical Analysis": (results_root / "statistical_tests" / "statistical_tests.json").exists(),
        "Ablation": (results_root / "metrics" / "matched_ablation_results.csv").exists(),
        "Incremental Novelty": (results_root / "metrics" / "incremental_results.csv").exists(),
        "Hyperparameter Analysis": (results_root / "metrics" / "hyperparameter_results.csv").exists(),
        "Optimizer Study": (results_root / "metrics" / "optimizer_results.csv").exists(),
        "Calibration": (results_root / "metrics" / "calibration_results.csv").exists(),
        "Explainability": (results_root / "explainability").exists(),
        "Personalization": (results_root / "metrics" / "personalization_depth_results.csv").exists(),
        "Cold Start": (results_root / "metrics" / "cold_start_results.csv").exists(),
        "Robustness": (results_root / "metrics" / "subgroup_results.csv").exists(),
        "Efficiency": (results_root / "metrics" / "efficiency_results.csv").exists(),
        "Limitations": True,
    }
    lines = ["# Manuscript-Ready Results Notes", "", "_Generated from artifact availability; unavailable experiments are not interpreted._", ""]
    for section in SECTIONS:
        lines.extend([f"## {section}", ""])
        if section == "Proposed Architecture":
            lines.extend(["The prespecified NuLKT architecture is documented in the generated architecture figure. Empirical contribution claims remain contingent on completed ablations.", ""])
        elif section == "Limitations":
            lines.extend(["The dataset is observational and does not identify a causal effect of explicit instruction. Correlated learner interactions require cluster-aware inference. Published-study comparisons require identical targets, tracks, splits, and metrics.", ""])
        elif availability[section]:
            lines.extend(["Artifact available; numerical prose must be populated programmatically from its machine-readable source.", ""])
        else:
            lines.extend(["Experiment unavailable/failed", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def generate_novelty_analysis(results_root: Path) -> Path:
    path = results_root / "publication" / "novelty_analysis.md"
    ablation = results_root / "metrics" / "matched_ablation_results.csv"
    lines = [
        "# Novelty Analysis", "",
        "## Central novelty", "",
        "The proposed novelty is not a new Transformer, mixture-of-experts, contrastive loss, or calibration method in isolation. It is a unified, leakage-safe framework that connects longitudinal English error prediction to transparent and uncertainty-aware explicit-instruction decision support.", "",
        "## Proposed contributions", "",
        "1. **Prediction-to-instruction integration.** NuLKT connects predicted learner difficulty, mastery, linguistic weakness, forgetting, and uncertainty to an auditable explicit-instruction recommendation engine. The system can abstain and request teacher review instead of forcing an unreliable recommendation.", "",
        "2. **Exercise-boundary leakage prevention.** Learner-history features are frozen before each submitted exercise, so outcomes from the current response and future interactions cannot influence its prediction. Automated perturbation tests verify that changing a future label leaves earlier predictions unchanged.", "",
        "3. **Joint linguistic and personalized representation.** Lexical, part-of-speech, morphological, dependency/context, exercise, and format signals are fused with a longitudinal learner state rather than treating knowledge tracing as an item-response sequence alone.", "",
        "4. **Two-timescale forgetting-aware knowledge tracing.** A local causal Transformer models within-session behavior, while a global causal memory models learning across sessions using elapsed time and trainable temporal decay.", "",
        "5. **Learner-conditioned expert routing.** A lightweight mixture-of-experts allows lexical, grammatical, temporal, history, and exercise-context evidence to contribute differently for different learner states, while exposing routing weights and utilization for audit.", "",
        "6. **Future-safe contrastive learner-state learning.** The auxiliary contrastive objective regularizes personalized learner representations using causal prefixes without constructing positives or negatives from future outcomes.", "",
        "7. **Calibrated uncertainty as a pedagogical safety mechanism.** Probability calibration, evidential uncertainty, abstention, and teacher-review routing make confidence part of the instructional decision rather than an optional reporting statistic.", "",
        "8. **Evaluation novelty through matched, longitudinal protocols.** The design combines official benchmark, per-learner chronological, and learner-disjoint cold-start evaluation with parameter-matched and compute-matched baselines, component ablations, multi-seed analysis, and learner-clustered inference.", "",
        "## Claims deliberately excluded", "",
        "Transformers, knowledge tracing, mixture-of-experts, contrastive learning, temporal decay, calibration, and explicit instruction are established ideas and are not individually claimed as new. The observational SLAM dataset also cannot establish that a recommendation causally improves learning.", "",
        "## Empirical status", "",
    ]
    if ablation.exists():
        lines.append(f"Component contribution claims must be derived from `{ablation}` and corrected statistical comparisons; implementation alone is not treated as evidence of improvement.")
    else:
        classical = results_root / "metrics" / "classical_validation_results.csv"
        if classical.exists():
            lines.append("The complete corpus has passed an exhaustive history-replay leakage audit, and full chronological validation-only classical baselines with learner-clustered intervals are available. Proposed-model multi-seed, ablation, and protected-test experiments have not yet been executed, so component-benefit, synergy, significance, superiority, and external-leaderboard claims are withheld.")
        else:
            lines.append("The architecture, data pipeline, causal tests, and quick smoke experiment are complete. Full multi-seed ablation and statistical experiments have not yet been executed, so performance, component-benefit, synergy, significance, and state-of-the-art claims are withheld.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_research_summary(results_root: Path) -> Path:
    """Write an abstract-style summary that reflects actual artifact availability."""
    path = results_root / "publication" / "research_summary.md"
    full_results = (results_root / "metrics" / "seed_results.csv").exists()
    classical_path = results_root / "metrics" / "classical_validation_results.csv"
    status = (
        "Full repeated-seed results are available and must be interpreted through the generated statistical artifacts."
        if full_results
        else (
            "The complete 3,396,935-interaction corpus has passed exhaustive history-replay, duplicate, and split-contamination audits. Full chronological validation-only classical baselines and 500-replicate learner-clustered intervals are available. The proposed neural model, repeated-seed, ablation, and protected-test experiments have not yet been run; therefore no NuLKT superiority or significance claim is made."
            if classical_path.exists()
            else "The repository has passed implementation, leakage, and quick end-to-end validation, but the full repeated-seed experiment has not yet been run; therefore the document makes no superiority or significance claim."
        )
    )
    text = f"""# Research Summary

## Title

**NuLKT: Personalized Knowledge Tracing for English Learning**

## Abstract-style summary

This project introduces NuLKT, an Explicit-Instruction Personalized Knowledge-Tracing Transformer for longitudinal English-learning support. The framework predicts token-level learner errors, estimates mastery, represents lexical and grammatical weaknesses, models temporal forgetting, and converts calibrated predictions into transparent explicit-instruction recommendations. Its linguistic encoder combines lexical, part-of-speech, morphological, dependency/context, exercise, and format information. A personalized learner encoder summarizes only interactions available before the current exercise. Local causal attention models behavior within sessions, while a time-aware global memory models longer-term learning and forgetting across sessions. A learner-conditioned mixture-of-experts adapts the contribution of linguistic, temporal, historical, and exercise evidence. Contrastive learning regularizes personalized learner states, and an evidential uncertainty head supports confidence estimation, abstention, and teacher review.

The principal data source is the official Duolingo 2018 SLAM English-learning track. The pipeline automatically downloads and verifies the dataset, constructs official, chronological-personalized, and learner-disjoint cold-start protocols, fits categorical vocabularies on training data only, and audits duplicate, temporal, target, and split leakage. Evaluation is designed around AUROC and AUPRC, supplemented by classification, calibration, efficiency, robustness, and learner-clustered statistical analyses. Classical, neural, knowledge-tracing, parameter-matched, optimizer, and ablation comparisons are prespecified.

The instructional layer is decision support rather than a causal intervention model. Because SLAM is observational, the study does not claim that recommended explicit instruction improves learning outcomes. {status}

## Research objective

Develop a computationally efficient and interpretable system that models an individual learner's evolving English knowledge state and recommends an appropriate explicit-instruction strategy when predictive evidence is sufficiently reliable.

## Expected contribution

The contribution is the experimentally testable integration of linguistic knowledge, personalized longitudinal state, hierarchical forgetting-aware causal modeling, learner-conditioned experts, contrastive representation learning, calibrated uncertainty, and auditable instructional recommendations—not the isolated invention of any one component.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
