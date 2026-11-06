"""Publication figures for the extended experimental programme (N1-N8).

Each generator returns ``None`` when its machine-readable source is absent, so an
unrun analysis leaves the corresponding figure registered as unavailable instead
of producing an empty or fabricated plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import PALETTE, save_figure


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def forgetting_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N1 - observed retention against lag, beside identifiable attention timescales."""

    analysis_dir = results_root / "analysis" / "forgetting"
    curve_path = analysis_dir / "empirical_retention_curve.csv"
    kernel_path = analysis_dir / "learned_temporal_kernel.csv"
    payload = _load_json(analysis_dir / "forgetting_analysis.json")
    if not curve_path.is_file() or not kernel_path.is_file() or payload is None:
        return None
    curve = pd.read_csv(curve_path)
    kernel = pd.read_csv(kernel_path)
    if curve.empty or kernel.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(
        curve["mean_lag_hours"], curve["accuracy"], yerr=curve["standard_error"],
        fmt="o", color=PALETTE[0], markersize=4, capsize=2, linewidth=1, label="Observed accuracy",
    )
    laws = payload.get("law_comparison", {}).get("laws", {})
    grid = np.geomspace(max(curve["mean_lag_hours"].min(), 1e-3), curve["mean_lag_hours"].max(), 200)
    formulas = {
        "exponential": lambda t, a, b, c: c + a * np.exp(-b * t),
        "power_law": lambda t, a, b, c: c + a * np.power(1.0 + t, -b),
        "logarithmic": lambda t, a, b, c: c + a / (1.0 + b * np.log1p(t)),
    }
    for index, (name, entry) in enumerate(laws.items()):
        if entry.get("status") != "fitted" or name not in formulas:
            continue
        parameters = entry["parameters"]
        fitted = formulas[name](grid, parameters["amplitude"], parameters["rate"], parameters["asymptote"])
        ax.plot(grid, fitted, color=PALETTE[index + 1], linewidth=1.6, label=f"{name.replace('_', ' ')} fit")
    ax.set_xscale("log")
    ax.set_xlabel("Lag since previous encounter with the same token (hours)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Empirical retention")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    identifiable = kernel[kernel["identifiable"]] if "identifiable" in kernel else kernel
    if identifiable.empty:
        ax.text(0.5, 0.5, "No identifiable temporal heads", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [f"{row.level[:1].upper()}{row.layer}h{row.head}" for row in identifiable.itertuples()]
        ax.bar(labels, identifiable["attention_half_life_hours"], color=PALETTE[2], edgecolor="black", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_ylabel("Attention half-life (hours, log scale)")
        ax.set_xlabel("Identifiable attention head")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        excluded = int(len(kernel) - len(identifiable))
        ax.set_title(f"Learned timescales ({excluded} unidentifiable heads excluded)")
    return [str(path) for path in save_figure(fig, output_dir / "forgetting_curve", include_svg=True)]


def risk_coverage_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N4 - selective risk against coverage for competing uncertainty signals."""

    analysis_dir = results_root / "analysis" / "selective_prediction"
    curves = sorted(analysis_dir.glob("risk_coverage_*.csv"))
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for index, path in enumerate(curves):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        name = path.stem.replace("risk_coverage_", "").replace("_", " ")
        style = "--" if name in {"random control", "oracle"} else "-"
        ax.plot(
            frame["coverage"], frame["selective_error"],
            linestyle=style, linewidth=1.8, color=PALETTE[index % len(PALETTE)], label=name,
        )
    ax.set_xlabel("Coverage (fraction of interactions retained)")
    ax.set_ylabel("Selective error rate")
    ax.set_title("Risk-coverage: abstaining on the least certain interactions")
    ax.legend(frameon=False, fontsize=8)
    return [str(path) for path in save_figure(fig, output_dir / "uncertainty_error", include_svg=True)]


def teacher_budget_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N4 - error mass surfaced within a bounded teacher review budget."""

    source = results_root / "analysis" / "selective_prediction" / "teacher_budget_curve.csv"
    if not source.is_file():
        return None
    frame = pd.read_csv(source)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    budget = frame["review_budget"] * 100
    ax.plot(budget, frame["error_recall"] * 100, marker="o", color=PALETTE[0], linewidth=1.8, label="Errors captured")
    ax.plot(budget, frame["review_precision"] * 100, marker="s", color=PALETTE[1], linewidth=1.8, label="Review precision")
    ax.plot(budget, budget, linestyle=":", color="grey", linewidth=1.4, label="Random review")
    ax.set_xlabel("Teacher review budget (% of interactions reviewed)")
    ax.set_ylabel("Percent")
    ax.set_title("Triage yield under a bounded review budget")
    ax.legend(frameon=False, fontsize=9)
    return [str(path) for path in save_figure(fig, output_dir / "teacher_review_budget", include_svg=True)]


def expert_utilization_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N3 - expert utilization and the learner archetypes routing induces."""

    analysis_dir = results_root / "analysis" / "archetypes"
    payload = _load_json(analysis_dir / "archetype_analysis.json")
    signature_path = analysis_dir / "learner_archetypes.csv"
    if payload is None or payload.get("status") == "unavailable" or not signature_path.is_file():
        return None
    utilization = payload.get("mean_expert_utilization", {})
    assigned = pd.read_csv(signature_path)
    if not utilization or assigned.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ax = axes[0]
    names = list(utilization)
    ax.bar(names, [utilization[name] for name in names], color=PALETTE[0], edgecolor="black", linewidth=0.6)
    ax.axhline(1 / len(names), linestyle="--", color="grey", linewidth=1.2, label="Uniform routing")
    ax.set_ylabel("Mean routing weight")
    ax.set_xlabel("Expert")
    ax.set_title("Expert utilization")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    expert_columns = [name for name in assigned.columns if name.startswith("expert_")]
    if "archetype" in assigned and expert_columns:
        profiles = assigned.groupby("archetype")[expert_columns].mean()
        width = 0.8 / max(len(profiles), 1)
        positions = np.arange(len(expert_columns))
        for index, (archetype, row) in enumerate(profiles.iterrows()):
            count = int((assigned["archetype"] == archetype).sum())
            ax.bar(
                positions + index * width, row.to_numpy(), width=width,
                color=PALETTE[index % len(PALETTE)], edgecolor="black", linewidth=0.5,
                label=f"Archetype {archetype} (n={count})",
            )
        ax.set_xticks(positions + width * (len(profiles) - 1) / 2)
        ax.set_xticklabels(expert_columns, rotation=30, fontsize=8)
        ax.set_ylabel("Mean routing weight")
        ax.set_title("Routing signature by discovered archetype")
        ax.legend(frameon=False, fontsize=8)
    return [str(path) for path in save_figure(fig, output_dir / "moe_expert_utilization", include_svg=True)]


def cold_start_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N5 - how quickly the model becomes useful for an unseen learner."""

    source = results_root / "analysis" / "cold_start" / "warm_up_curve.csv"
    if not source.is_file():
        return None
    curve = pd.read_csv(source)
    if curve.empty:
        return None
    payload = _load_json(results_root / "analysis" / "cold_start" / "cold_start_analysis.json") or {}
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    labels = [
        f"{int(row.history_lower)}-{'inf' if not np.isfinite(row.history_upper) else int(row.history_upper)}"
        for row in curve.itertuples()
    ]
    ax.plot(labels, curve["auroc"], marker="o", color=PALETTE[0], linewidth=1.8, label="Cold-start AUROC")
    comparison = payload.get("protocol_comparison", {})
    warm = comparison.get("chronological_personalized", {}).get("auroc")
    if warm is not None:
        ax.axhline(warm, linestyle="--", color=PALETTE[3], linewidth=1.5, label="Chronological personalized AUROC")
    ax.set_xlabel("Past-only interactions accumulated by the new learner")
    ax.set_ylabel("AUROC")
    ax.set_title("Cold-start warm-up: adaptation without a learned learner identity")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(frameon=False, fontsize=9)
    return [str(path) for path in save_figure(fig, output_dir / "cold_start_personalized", include_svg=True)]


def subgroup_calibration_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N8 - calibration error spread across reported subgroups."""

    payload = _load_json(results_root / "analysis" / "subgroup_calibration" / "subgroup_calibration.json")
    if payload is None:
        return None
    tested = {
        attribute: result for attribute, result in payload.get("by_attribute", {}).items()
        if result.get("status") == "tested" and result.get("subgroups")
    }
    if not tested:
        return None
    overall = payload.get("overall", {}).get("globally_calibrated", {}).get("ece")

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    offset = 0
    ticks: list[float] = []
    labels: list[str] = []
    for index, (attribute, result) in enumerate(tested.items()):
        frame = pd.DataFrame(result["subgroups"]).sort_values("ece_global_temperature", ascending=False)
        positions = np.arange(len(frame)) + offset
        ax.bar(
            positions, frame["ece_global_temperature"], color=PALETTE[index % len(PALETTE)],
            edgecolor="black", linewidth=0.5, label=attribute,
        )
        ticks.extend(positions.tolist())
        labels.extend(frame["subgroup"].astype(str).tolist())
        offset += len(frame) + 1
    if overall is not None:
        ax.axhline(overall, linestyle="--", color="grey", linewidth=1.4, label="Overall ECE")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=75, fontsize=6)
    ax.set_ylabel("Expected calibration error (global temperature)")
    ax.set_title("Calibration is not uniform across subgroups")
    ax.legend(frameon=False, fontsize=8)
    return [str(path) for path in save_figure(fig, output_dir / "subgroup_robustness", include_svg=True)]


def personalization_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N2 - discrimination under counterfactual learner-identity interventions."""

    payload = _load_json(results_root / "analysis" / "personalization" / "personalization_analysis.json")
    if payload is None:
        return None
    summary = payload.get("intervention_summary", {})
    observed = summary.get("observed", {})
    if not observed:
        return None
    conditions = [
        ("True learner identity", observed.get("auroc"), 0.0),
        ("Permuted identity", summary.get("permuted_identity", {}).get("auroc"),
         summary.get("permuted_identity", {}).get("mean_absolute_delta")),
        ("Identity removed", summary.get("anonymous_identity", {}).get("auroc"),
         summary.get("anonymous_identity", {}).get("mean_absolute_delta")),
    ]
    conditions = [row for row in conditions if row[1] is not None]
    if len(conditions) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    names = [row[0] for row in conditions]
    ax = axes[0]
    ax.bar(names, [row[1] for row in conditions], color=PALETTE[0], edgecolor="black", linewidth=0.6)
    lowest = min(row[1] for row in conditions)
    highest = max(row[1] for row in conditions)
    margin = max((highest - lowest) * 3, 0.01)
    ax.set_ylim(lowest - margin, highest + margin)
    ax.set_ylabel("AUROC")
    ax.set_title("Discrimination under identity intervention")
    ax.tick_params(axis="x", rotation=15, labelsize=8)

    ax = axes[1]
    ax.bar(names, [row[2] for row in conditions], color=PALETTE[1], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean absolute change in predicted probability")
    ax.set_title("Magnitude of prediction change")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    return [str(path) for path in save_figure(fig, output_dir / "personalization_intervention", include_svg=True)]


def linguistic_transfer_figure(results_root: Path, output_dir: Path) -> list[str] | None:
    """N6 - error difficulty by part of speech and pre-registered transfer tests."""

    analysis_dir = results_root / "analysis" / "linguistic_profile"
    pos_path = analysis_dir / "difficulty_by_part_of_speech.csv"
    payload = _load_json(analysis_dir / "linguistic_profile.json")
    if not pos_path.is_file() or payload is None:
        return None
    pos = pd.read_csv(pos_path)
    if pos.empty:
        return None

    hypotheses = [
        row for row in payload.get("transfer_analysis", {}).get("hypotheses", [])
        if row.get("status") == "tested"
    ]
    panels = 2 if hypotheses else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.0 * panels, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.bar(pos["category"].astype(str), pos["error_rate"], color=PALETTE[0], edgecolor="black", linewidth=0.6)
    corpus_rate = float(payload.get("corpus_error_rate", np.nan))
    if np.isfinite(corpus_rate):
        ax.axhline(corpus_rate, linestyle="--", color="grey", linewidth=1.4, label="Corpus error rate")
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Observed error rate")
    ax.set_xlabel("Universal part-of-speech tag")
    ax.set_title("Difficulty by part of speech")
    ax.tick_params(axis="x", rotation=60, labelsize=7)

    if hypotheses:
        ax = axes[1]
        names = [row["name"].replace("_", " ") for row in hypotheses]
        differences = [row["difference"] for row in hypotheses]
        errors = [
            [row["difference"] - row["ci_lower"] for row in hypotheses],
            [row["ci_upper"] - row["difference"] for row in hypotheses],
        ]
        colors = [PALETTE[2] if row["prediction_supported"] else PALETTE[3] for row in hypotheses]
        positions = np.arange(len(names))
        ax.barh(positions, differences, xerr=errors, color=colors, edgecolor="black", linewidth=0.6, capsize=3)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Error-rate difference from corpus mean")
        ax.set_title("Pre-registered transfer predictions\n(green supported, red not supported)")
    return [str(path) for path in save_figure(fig, output_dir / "linguistic_transfer", include_svg=True)]


ANALYSIS_FIGURES = {
    "forgetting_curve": forgetting_figure,
    "uncertainty_error": risk_coverage_figure,
    "teacher_review_budget": teacher_budget_figure,
    "moe_expert_utilization": expert_utilization_figure,
    "cold_start_personalized": cold_start_figure,
    "subgroup_robustness": subgroup_calibration_figure,
    "personalization_intervention": personalization_figure,
    "linguistic_transfer": linguistic_transfer_figure,
}


def generate_analysis_figures(results_root: Path, figure_dir: Path) -> dict[str, dict]:
    """Render every analysis figure whose source exists; report the rest honestly."""

    output_dir = figure_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict] = {}
    for name, generator in ANALYSIS_FIGURES.items():
        try:
            files = generator(results_root, output_dir)
        except Exception as exc:
            registry[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            plt.close("all")
            continue
        registry[name] = (
            {"status": "available", "files": files, "source": "results/analysis"}
            if files
            else {"status": "unavailable", "reason": "required analysis output absent"}
        )
        plt.close("all")
    return registry
