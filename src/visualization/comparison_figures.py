"""Figures for the head-to-head comparisons that decide the paper's claims.

These are deliberately separate from ``publication_figures``: that module renders
whatever machine-readable sources happen to exist, while this one renders the
three comparisons a reader will actually interrogate — how the arms rank on the
official partition, whether combining them recovers anything, and how the
proposed model stands against the *strongest* baseline rather than the strongest
neural baseline.

Every function returns ``None`` when its source is absent, so a partial results
tree renders a partial figure set instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import configure_publication_style, save_figure

# Slots 1-3 of the validated categorical palette. Three series is the all-pairs
# cap for this ordering, which is exactly what these comparisons need.
NEURAL = "#2a78d6"
BOOSTED = "#eb6834"
COMBINED = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"


def _json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _style_axes(ax) -> None:
    ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelcolor="#52514e")


def _roc_points(labels: np.ndarray, scores: np.ndarray, n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """ROC on a fixed grid so a 500k-row curve stays a small vector path."""
    order = np.argsort(-scores)
    y = np.asarray(labels, dtype=float)[order]
    tps, fps = np.cumsum(y), np.cumsum(1.0 - y)
    positive, negative = tps[-1], fps[-1]
    if positive == 0 or negative == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tpr, fpr = tps / positive, fps / negative
    index = np.unique(np.linspace(0, len(tpr) - 1, n_points).astype(int))
    return np.concatenate([[0.0], fpr[index]]), np.concatenate([[0.0], tpr[index]])


def official_partition_ranking(results_root: Path, output: Path) -> list[str] | None:
    """The headline table as a chart: every arm on the official test partition."""
    payload = _json(results_root / "benchmark" / "ensemble_study.json")
    if not payload:
        return None
    rows = sorted(payload["test_results"], key=lambda r: r["test_auroc"])
    labels = [r["combiner"].replace("_", " ") for r in rows]
    values = [r["test_auroc"] for r in rows]
    colors = []
    for row in rows:
        if row["combiner"] == "neural_only":
            colors.append(NEURAL)
        elif row["combiner"] == "boosted_only":
            colors.append(BOOSTED)
        else:
            colors.append(COMBINED)

    # A dot plot, not bars: the interesting spread here is a few thousandths of
    # AUROC, which needs a non-zero axis, and a bar drawn on a non-zero axis
    # misrepresents its own length. Dots carry position only, so the zoomed
    # range is honest.
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    y = np.arange(len(labels))
    floor = min(values) - 0.004
    reference = next((r["test_auroc"] for r in rows if r["combiner"] == "boosted_only"), None)
    if reference is not None:
        ax.axvline(reference, color=BOOSTED, linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)
    for position, value, color in zip(y, values, colors):
        ax.plot([floor, value], [position, position], color=color, linewidth=1.4, alpha=0.35, zorder=2)
        ax.plot(value, position, "o", markersize=11, color=color, zorder=3)
        ax.text(value + 0.0006, position, f"{value:.4f}", va="center", ha="left", fontsize=12, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(floor, max(values) + 0.005)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("Test AUROC (official SLAM partition)")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def ensemble_selection(results_root: Path, output: Path) -> list[str] | None:
    """Dev cross-validation beside test outcome — does selection transfer?"""
    payload = _json(results_root / "benchmark" / "ensemble_study.json")
    if not payload:
        return None
    cv = {row["combiner"]: row for row in payload["cross_validation"]}
    test = {row["combiner"]: row for row in payload["test_results"]}
    names = [name for name in cv if name in test]
    names.sort(key=lambda n: cv[n]["cv_auroc_mean"])
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    ax.errorbar(
        x, [cv[n]["cv_auroc_mean"] for n in names], yerr=[cv[n]["cv_auroc_sd"] for n in names],
        fmt="o", markersize=9, color=NEURAL, ecolor=NEURAL, elinewidth=1.6, capsize=4,
        label="Dev, learner-grouped 5-fold CV (±1 SD)", zorder=3,
    )
    ax.plot(
        x, [test[n]["test_auroc"] for n in names], "s", markersize=9, color=BOOSTED,
        label="Test (read once)", zorder=3,
    )
    selected = payload["selection"]["selected_combiner"]
    if selected in names:
        ax.axvline(names.index(selected), color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
        ax.text(
            names.index(selected), ax.get_ylim()[1], " selected", fontsize=11,
            color="#52514e", va="top", ha="left",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=11)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def official_roc(results_root: Path, output: Path) -> list[str] | None:
    """ROC for both arms and the combiner, on the official test partition."""
    path = results_root / "benchmark" / "ensemble_test_predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    labels = frame["label"].to_numpy()
    series = [
        ("NuLKT", frame["probability_neural"].to_numpy(), NEURAL),
        ("Gradient boosting", frame["probability_boosted"].to_numpy(), BOOSTED),
        ("Selected combiner", frame["probability"].to_numpy(), COMBINED),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    for name, scores, color in series:
        fpr, tpr = _roc_points(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=name, zorder=3)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def proposed_versus_best_baseline(results_root: Path, output: Path) -> list[str] | None:
    """The corrected chronological-split comparison, with the pool made explicit."""
    selection = _json(results_root / "raw_runs" / "best_baseline" / "selection.json")
    proposed_path = results_root / "metrics" / "proposed_results.csv"
    if not selection or not proposed_path.is_file():
        return None
    candidates = pd.DataFrame(selection["candidates"])
    if candidates.empty or "family" not in candidates:
        return None
    proposed = pd.read_csv(proposed_path)
    test_row = proposed[proposed["split"].astype(str) == "test"]
    if test_row.empty:
        return None
    proposed_auroc = float(test_row.iloc[0]["auroc_mean"])

    candidates = candidates.sort_values("auroc", ascending=True)
    labels = list(candidates["model"].astype(str)) + ["NuLKT"]
    values = list(candidates["auroc"].astype(float)) + [proposed_auroc]
    colors = [BOOSTED if family == "classical" else MUTED for family in candidates["family"]]
    colors.append(NEURAL)

    fig, ax = plt.subplots(figsize=(9.0, 6.4), constrained_layout=True)
    y = np.arange(len(labels))
    floor = min(values) - 0.012
    for position, value, color in zip(y, values, colors):
        ax.plot([floor, value], [position, position], color=color, linewidth=1.4, alpha=0.35, zorder=2)
        ax.plot(value, position, "o", markersize=10, color=color, zorder=3)
        ax.text(value + 0.0025, position, f"{value:.4f}", va="center", ha="left", fontsize=11, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(floor, max(values) + 0.022)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel("AUROC (validation for baselines, test for NuLKT)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NEURAL),
        plt.Rectangle((0, 0), 1, 1, color=BOOSTED),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
    ]
    ax.legend(handles, ["Proposed", "Classical (boosted)", "Neural baseline"], frameon=False, loc="lower right")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def feature_gap_composition(results_root: Path, output: Path) -> list[str] | None:
    """Whose evidence is the boosted model actually using, and can we reach it?"""
    payload = _json(results_root / "analysis" / "feature_gap.json")
    if not payload or payload.get("status") != "available":
        return None
    frame = pd.DataFrame(payload["features"]).sort_values("gain_fraction", ascending=False)
    top = frame.head(12).iloc[::-1]

    def _colour(row) -> str:
        if row["reaches_sequence_model"]:
            return NEURAL
        return COMBINED if row["added_by_exercise_context"] else BOOSTED

    colours = [_colour(row) for _, row in top.iterrows()]
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    positions = np.arange(len(top))
    ax.barh(positions, top["gain_fraction"] * 100, color=colours, height=0.66)
    for position, value in zip(positions, top["gain_fraction"] * 100):
        ax.text(value + 0.3, position, f"{value:.1f}%", va="center", ha="left", fontsize=11, color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Share of gradient-boosting split gain (%)")
    ax.set_xlim(0, float(top["gain_fraction"].max() * 100) + 4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NEURAL, COMBINED, BOOSTED)]
    ax.legend(
        handles,
        ["Already reaches the model", "Added by exercise-context arm", "Still unreachable"],
        frameon=False, loc="lower right",
    )
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def score_distributions(results_root: Path, output: Path) -> list[str] | None:
    """Where each arm places its probability mass, separated by true outcome."""
    path = results_root / "benchmark" / "ensemble_test_predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    arms = [
        ("NuLKT", "probability_neural", NEURAL),
        ("Gradient boosting", "probability_boosted", BOOSTED),
        ("Selected combiner", "probability", COMBINED),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), sharey=True, constrained_layout=True)
    edges = np.linspace(0, 1, 41)
    centres = (edges[:-1] + edges[1:]) / 2
    for ax, (name, column, colour) in zip(axes, arms):
        for label, style, alpha in ((0, "-", 0.75), (1, "--", 0.95)):
            subset = frame.loc[frame["label"] == label, column]
            density, _ = np.histogram(subset, bins=edges, density=True)
            ax.plot(
                centres, density, style, color=colour, linewidth=2.0, alpha=alpha,
                label="correct" if label == 0 else "error",
            )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Predicted error probability")
        ax.set_xlim(0, 1)
        _style_axes(ax)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def cumulative_gain(results_root: Path, output: Path) -> list[str] | None:
    """Operational view: errors caught against the review budget spent."""
    path = results_root / "benchmark" / "ensemble_test_predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    labels = frame["label"].to_numpy()
    total = labels.sum()
    if total == 0:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle=":", label="Random review")
    for name, column, colour in (
        ("NuLKT", "probability_neural", NEURAL),
        ("Gradient boosting", "probability_boosted", BOOSTED),
        ("Selected combiner", "probability", COMBINED),
    ):
        order = np.argsort(-frame[column].to_numpy())
        captured = np.cumsum(labels[order]) / total
        reviewed = np.arange(1, len(order) + 1) / len(order)
        index = np.unique(np.linspace(0, len(order) - 1, 512).astype(int))
        ax.plot(reviewed[index], captured[index], color=colour, linewidth=2.0, label=name)
    ax.set_xlabel("Fraction of interactions reviewed")
    ax.set_ylabel("Fraction of errors captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def ablation_tornado(results_root: Path, output: Path) -> list[str] | None:
    """Signed deviation from the full model - which components actually carry it."""
    source = results_root / "metrics" / "matched_ablation_results.csv"
    if not source.is_file():
        return None
    frame = pd.read_csv(source)
    if "variant" not in frame or "auroc_mean" not in frame or frame.empty:
        return None
    full = frame[frame["variant"].astype(str).str.startswith("A0")]
    if full.empty:
        return None
    reference = float(full.iloc[0]["auroc_mean"])
    others = frame[~frame["variant"].astype(str).str.startswith("A0")].copy()
    others["delta"] = others["auroc_mean"] - reference
    others = others.sort_values("delta")
    labels = [str(v).split("_", 1)[-1].replace("_", " ") for v in others["variant"]]

    fig, ax = plt.subplots(figsize=(9.6, 6.4), constrained_layout=True)
    positions = np.arange(len(others))
    # A component whose removal *raises* AUROC is the finding worth seeing, so it
    # takes the contrasting hue instead of being buried in a uniform ramp.
    colours = [BOOSTED if delta > 0 else NEURAL for delta in others["delta"]]
    ax.barh(positions, others["delta"], color=colours, height=0.64)
    ax.axvline(0, color="#c3c2b7", linewidth=1.0)
    for position, delta in zip(positions, others["delta"]):
        offset = 0.0012 if delta > 0 else -0.0012
        ax.text(
            delta + offset, position, f"{delta:+.4f}", va="center",
            ha="left" if delta > 0 else "right", fontsize=10, color=INK,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=11)
    span = float(np.abs(others["delta"]).max())
    ax.set_xlim(-span * 1.3, span * 0.75)
    ax.set_xlabel("AUROC change when the component is removed")
    _style_axes(ax)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def seed_stability_strip(results_root: Path, output: Path) -> list[str] | None:
    """Per-seed spread set against the strongest baseline on a shared axis."""
    seed_path = results_root / "metrics" / "seed_results.csv"
    if not seed_path.is_file():
        return None
    frame = pd.read_csv(seed_path)
    tested = frame[frame["split"].astype(str) == "test"] if "split" in frame else frame
    if tested.empty:
        return None
    values = tested["auroc"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(values))
    ax.plot(values, jitter, "o", markersize=11, color=NEURAL, alpha=0.85, label="NuLKT seeds")
    ax.plot([values.mean()], [0], "|", markersize=34, color=NEURAL, markeredgewidth=2.4)

    baseline_metrics = _json(results_root / "raw_runs" / "best_baseline" / "test_metrics.json")
    if baseline_metrics and "auroc" in baseline_metrics:
        reference = float(baseline_metrics["auroc"])
        ax.axvline(reference, color=BOOSTED, linewidth=1.6, linestyle="--")
        ax.text(
            reference, 0.15, f" {baseline_metrics.get('model', 'best baseline')} {reference:.4f}",
            color=BOOSTED, fontsize=11, va="bottom", ha="left",
        )
    ax.set_yticks([])
    ax.set_ylim(-0.24, 0.30)
    ax.set_xlabel("Test AUROC")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def generate_comparison_figures(results_root: Path) -> dict[str, dict[str, Any]]:
    """Render every comparison whose source exists; report the rest as absent."""
    configure_publication_style()
    figure_dir = results_root / "figures" / "comparison"
    figure_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "official_partition_ranking": official_partition_ranking,
        "ensemble_selection": ensemble_selection,
        "official_roc": official_roc,
        "proposed_versus_best_baseline": proposed_versus_best_baseline,
        "feature_gap_composition": feature_gap_composition,
        "score_distributions": score_distributions,
        "cumulative_gain": cumulative_gain,
        "ablation_tornado": ablation_tornado,
        "seed_stability_strip": seed_stability_strip,
    }
    registry: dict[str, dict[str, Any]] = {}
    for name, builder in builders.items():
        files = builder(results_root, figure_dir / name)
        registry[name] = (
            {"status": "available", "files": files}
            if files
            else {"status": "unavailable", "reason": "required machine-readable source absent"}
        )
    (figure_dir / "comparison_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry
