"""Reproducible descriptive statistics and publication-style SLAM EDA figures."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.manifest import manifest_parts
from src.visualization.style import active_dpi


LOGGER = logging.getLogger(__name__)


def _root(value: Path | None) -> Path:
    return value.resolve() if value else Path(__file__).resolve().parents[2]


def _load(root: Path, quick: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "data" / "processed" / "en_es" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the preprocessing stage before EDA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest_parts(root, manifest)
    if quick:
        paths = paths[:1]
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if quick and len(frame) > 100_000:
        frame = frame.iloc[:100_000].copy()
    return frame, manifest


def _wilson_interval(errors: pd.Series, interactions: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided 95% Wilson score intervals for binomial proportions."""

    z = 1.959963984540054
    successes = errors.to_numpy(dtype=float)
    totals = interactions.to_numpy(dtype=float)
    lower = np.full(len(totals), np.nan)
    upper = np.full(len(totals), np.nan)
    valid = totals > 0
    proportion = successes[valid] / totals[valid]
    denominator = 1 + z**2 / totals[valid]
    centre = (proportion + z**2 / (2 * totals[valid])) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / totals[valid] + z**2 / (4 * totals[valid] ** 2)
    ) / denominator
    lower[valid] = np.maximum(0.0, centre - half_width)
    upper[valid] = np.minimum(1.0, centre + half_width)
    return lower, upper


def _summarize_binary(frame: pd.DataFrame, column: str, output_name: str) -> pd.DataFrame:
    output = (
        frame.groupby(column, dropna=False, observed=True)["label"]
        .agg(interactions="size", errors="sum", error_rate="mean")
        .reset_index()
        .rename(columns={column: output_name})
        .sort_values("interactions", ascending=False)
    )
    output["ci95_lower"], output["ci95_upper"] = _wilson_interval(
        output["errors"], output["interactions"]
    )
    return output


def _summarize_binned(
    frame: pd.DataFrame,
    value_column: str,
    output_name: str,
    bins: list[float] | np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    categories = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)
    working = pd.DataFrame({output_name: categories, "label": frame["label"].to_numpy()})
    output = (
        working.groupby(output_name, observed=False)["label"]
        .agg(interactions="size", errors="sum", error_rate="mean")
        .reset_index()
    )
    output["ci95_lower"], output["ci95_upper"] = _wilson_interval(
        output["errors"], output["interactions"]
    )
    return output


def compute_eda_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate descriptive and uncertainty-aware summaries from interactions."""

    learner_lengths = frame.groupby("user_id", observed=True).size().rename("interactions")
    session_lengths = frame.groupby(["user_id", "session_id"], observed=True).size().rename("interactions")
    exercise_lengths = frame.groupby("exercise_id", observed=True).size().rename("tokens")
    stats = pd.DataFrame(
        {
            "statistic": [
                "learners", "interactions", "exercises", "sessions", "errors", "error_rate",
                "mean_interactions_per_learner", "median_interactions_per_learner",
                "mean_tokens_per_exercise", "median_tokens_per_exercise",
            ],
            "value": [
                frame["user_id"].nunique(), len(frame), frame["exercise_id"].nunique(),
                frame[["user_id", "session_id"]].drop_duplicates().shape[0], int(frame["label"].sum()),
                float(frame["label"].mean()), float(learner_lengths.mean()), float(learner_lengths.median()),
                float(exercise_lengths.mean()), float(exercise_lengths.median()),
            ],
        }
    )

    morphology_keys = (
        frame["morphology"].fillna("").astype(str).str.split("|").explode().str.split("=", n=1).str[0]
    )
    morphology_distribution = (
        morphology_keys[morphology_keys.ne("")]
        .value_counts()
        .rename_axis("morphological_attribute")
        .reset_index(name="interactions")
    )

    elapsed_table = _summarize_binned(
        frame.assign(
            elapsed_since_prior_exercise_seconds=pd.to_numeric(
                frame["elapsed_since_prior_exercise_seconds"], errors="coerce"
            ).clip(lower=0)
        ),
        "elapsed_since_prior_exercise_seconds",
        "elapsed_bin",
        np.array([0, 60, 300, 1_800, 3_600, 21_600, 86_400, 604_800, 2_592_000, np.inf]),
        ["0–1m", "1–5m", "5–30m", "30–60m", "1–6h", "6–24h", "1–7d", "7–30d", ">30d"],
    )
    history_table = _summarize_binned(
        frame, "prior_interactions", "history_bin",
        [-0.5, 0.5, 9.5, 49.5, 199.5, 999.5, np.inf],
        ["0", "1–9", "10–49", "50–199", "200–999", "≥1,000"],
    )
    session_position_table = _summarize_binned(
        frame, "session_position", "session_position_bin",
        [-0.5, 1.5, 5.5, 10.5, 20.5, 50.5, np.inf],
        ["0–1", "2–5", "6–10", "11–20", "21–50", ">50"],
    )

    token_summary = _summarize_binary(frame, "token", "token")
    token_summary["frequency_band"] = pd.cut(
        token_summary["interactions"],
        bins=[0, 2, 5, 10, 25, 50, 100, 250, 500, 1_000, np.inf],
        labels=["1", "2–4", "5–9", "10–24", "25–49", "50–99", "100–249", "250–499", "500–999", "≥1,000"],
        include_lowest=True,
        right=False,
    )
    lexical_frequency_bands = (
        token_summary.groupby("frequency_band", observed=False)
        .agg(types=("token", "size"), interactions=("interactions", "sum"), errors=("errors", "sum"))
        .reset_index()
    )
    lexical_frequency_bands["error_rate"] = (
        lexical_frequency_bands["errors"] / lexical_frequency_bands["interactions"].replace(0, np.nan)
    )
    lexical_frequency_bands["ci95_lower"], lexical_frequency_bands["ci95_upper"] = _wilson_interval(
        lexical_frequency_bands["errors"], lexical_frequency_bands["interactions"]
    )

    class_distribution = frame["label"].value_counts().sort_index().rename_axis("label").reset_index(name="interactions")
    class_distribution["class"] = class_distribution["label"].map({0: "correct", 1: "error"})
    class_distribution["proportion"] = class_distribution["interactions"] / len(frame)

    return {
        "dataset_statistics": stats,
        "class_distribution": class_distribution,
        "official_split_distribution": frame["official_split"].value_counts().rename_axis("official_split").reset_index(name="interactions"),
        "learner_sequence_lengths": learner_lengths.reset_index(),
        "session_lengths": session_lengths.reset_index(),
        "exercise_format_distribution": _summarize_binary(frame, "exercise_format", "exercise_format"),
        "pos_distribution": _summarize_binary(frame, "part_of_speech", "part_of_speech"),
        "morphology_distribution": morphology_distribution,
        "lexical_frequency": token_summary,
        "lexical_frequency_bands": lexical_frequency_bands,
        "error_by_elapsed_time": elapsed_table,
        "error_by_client": _summarize_binary(frame, "client", "client"),
        "error_by_history_depth": history_table,
        "error_by_session_position": session_position_table,
        "history_distribution": frame[
            ["prior_interactions", "prior_exercises", "prior_error_rate", "prior_token_exposures"]
        ].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index(),
    }


def _save_figure(fig: Any, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=active_dpi(), bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")


def _annotate_bars(axis: Any, values: np.ndarray, labels: list[str], *, horizontal: bool = False) -> None:
    for index, (value, label) in enumerate(zip(values, labels)):
        if horizontal:
            axis.annotate(label, (value, index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        else:
            axis.annotate(label, (index, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)


def _error_rate_figure(
    plt: Any,
    table: pd.DataFrame,
    category: str,
    output_dir: Path,
    stem: str,
    title: str,
    color: str,
    *,
    max_categories: int = 15,
) -> None:
    plotted = table.loc[table["interactions"] > 0].nlargest(max_categories, "interactions").sort_values("error_rate")
    y = np.arange(len(plotted))
    rates = plotted["error_rate"].to_numpy(float)
    errors = np.vstack((rates - plotted["ci95_lower"].to_numpy(float), plotted["ci95_upper"].to_numpy(float) - rates))
    fig, axis = plt.subplots(figsize=(9.5, max(5.0, 0.42 * len(plotted) + 1.8)), constrained_layout=True)
    axis.errorbar(rates, y, xerr=errors, fmt="o", color=color, ecolor="#4d4d4d", capsize=3, markersize=7)
    axis.set_yticks(y, plotted[category].fillna("missing").astype(str))
    axis.set(xlabel="Observed error rate (95% Wilson CI)", title=title, xlim=(0, min(1.0, max(0.35, float(plotted["ci95_upper"].max()) * 1.22))))
    axis.grid(False)
    for position, rate, count in zip(y, rates, plotted["interactions"]):
        axis.annotate(f"{rate:.3f}; n={int(count):,}", (rate, position), xytext=(7, 7), textcoords="offset points", fontsize=8)
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def _ordered_error_line(
    plt: Any,
    table: pd.DataFrame,
    category: str,
    output_dir: Path,
    stem: str,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    plotted = table.loc[table["interactions"] > 0].copy()
    x = np.arange(len(plotted))
    rates = plotted["error_rate"].to_numpy(float)
    lower = plotted["ci95_lower"].to_numpy(float)
    upper = plotted["ci95_upper"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    axis.plot(x, rates, marker="o", linewidth=2.1, color=color)
    axis.fill_between(x, lower, upper, alpha=0.18, color=color, label="95% Wilson CI")
    axis.set_xticks(x, plotted[category].astype(str), rotation=30, ha="right")
    axis.set(xlabel=xlabel, ylabel="Observed error rate", title=title)
    axis.grid(False)
    axis.legend(frameon=False)
    for position, rate, count in zip(x, rates, plotted["interactions"]):
        axis.annotate(f"{rate:.3f}\nn={int(count):,}", (position, rate), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def make_eda_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[str]:
    """Render separate, uncertainty-aware EDA figures as PNG/PDF/SVG."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("EDA figure generation requires matplotlib") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    if serif != "Times New Roman":
        LOGGER.warning("Times New Roman is unavailable; using %s", serif)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [serif], "font.size": 13,
        "axes.labelsize": 13, "axes.titlesize": 14, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11, "savefig.dpi": 800,
    })
    produced: list[str] = []

    class_table = tables["class_distribution"]
    fig, axis = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    values = class_table["proportion"].to_numpy(float)
    axis.bar(class_table["class"], values, color=["#9ecae1", "#fdae6b"], edgecolor="black")
    axis.set(ylabel="Proportion", title="Response-label distribution", ylim=(0, min(1.0, values.max() * 1.18)))
    _annotate_bars(axis, values, [f"{value:.3f}\nn={count:,}" for value, count in zip(values, class_table["interactions"])])
    _save_figure(fig, output_dir, "class_error_distribution")
    plt.close(fig)
    produced.append("class_error_distribution")

    split_table = tables["official_split_distribution"]
    fig, axis = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    split_values = split_table["interactions"].to_numpy(float)
    axis.bar(split_table["official_split"].astype(str), split_values, color="#a1d99b", edgecolor="black")
    axis.set(ylabel="Interactions", title="Official corpus partitions")
    axis.ticklabel_format(axis="y", style="plain")
    _annotate_bars(axis, split_values, [f"n={int(value):,}" for value in split_values])
    _save_figure(fig, output_dir, "official_split_distribution")
    plt.close(fig)
    produced.append("official_split_distribution")

    for table_name, stem, title, xlabel, color in [
        ("learner_sequence_lengths", "learner_history_length_ecdf", "Learner history-length distribution", "Interactions per learner", "#3182bd"),
        ("session_lengths", "session_length_ecdf", "Session-length distribution", "Interactions per session", "#31a354"),
    ]:
        values = np.sort(tables[table_name]["interactions"].to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        fig, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        axis.step(values, cumulative, where="post", linewidth=2.1, color=color)
        axis.set_xscale("log")
        axis.set(xlabel=f"{xlabel} (log scale)", ylabel="Empirical cumulative proportion", title=title, ylim=(0, 1.01))
        axis.grid(False)
        median = float(np.median(values))
        axis.axvline(median, linestyle="--", color="#555555", label=f"Median = {median:,.0f}; n = {len(values):,}")
        axis.legend(frameon=False, loc="lower right")
        _save_figure(fig, output_dir, stem)
        plt.close(fig)
        produced.append(stem)

    for table_name, category, stem, title, color, limit in [
        ("exercise_format_distribution", "exercise_format", "exercise_format_error_rates", "Error rate by exercise format", "#e6550d", 12),
        ("pos_distribution", "part_of_speech", "part_of_speech_error_rates", "Error rate by part of speech", "#756bb1", 15),
        ("error_by_client", "client", "client_error_rates", "Error rate by client platform", "#636363", 12),
    ]:
        _error_rate_figure(plt, tables[table_name], category, output_dir, stem, title, color, max_categories=limit)
        produced.append(stem)

    for table_name, category, stem, title, xlabel, color in [
        ("error_by_elapsed_time", "elapsed_bin", "elapsed_time_error_rates", "Elapsed time and learner error", "Time since prior exercise", "#3182bd"),
        ("error_by_history_depth", "history_bin", "history_depth_error_rates", "Prior learner history and observed error", "Prior interactions", "#31a354"),
        ("error_by_session_position", "session_position_bin", "session_position_error_rates", "Within-session position and observed error", "Interaction position in session", "#756bb1"),
        ("lexical_frequency_bands", "frequency_band", "lexical_frequency_error_rates", "Token frequency and observed error", "Corpus frequency per token type", "#e6550d"),
    ]:
        _ordered_error_line(plt, tables[table_name], category, output_dir, stem, title, xlabel, color)
        produced.append(stem)

    morphology = tables["morphology_distribution"].head(15).sort_values("interactions")
    fig, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    counts = morphology["interactions"].to_numpy(float)
    axis.barh(morphology["morphological_attribute"].astype(str), counts, color="#bcbddc", edgecolor="black")
    axis.set(xlabel="Interactions", title="Most frequent morphological annotations")
    axis.ticklabel_format(axis="x", style="plain")
    _annotate_bars(axis, counts, [f"n={int(value):,}" for value in counts], horizontal=True)
    _save_figure(fig, output_dir, "morphology_attribute_prevalence")
    plt.close(fig)
    produced.append("morphology_attribute_prevalence")
    return produced


def run_eda(*, root: Path | None = None, quick: bool = False) -> dict[str, Any]:
    """Generate machine-readable EDA tables and publication figures."""

    root = _root(root)
    frame, manifest = _load(root, quick)
    tables = compute_eda_tables(frame)
    table_dir = root / "results" / "tables" / "eda"
    figure_dir = root / "results" / "figures" / "eda"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    figures = make_eda_figures(tables, figure_dir)
    summary = {
        "status": "DEBUG/QUICK MODE — NOT FOR PUBLICATION" if quick else "publication data",
        "track": manifest["track"],
        "preprocessing_hash": manifest.get("preprocessing_hash"),
        "rows_analyzed": len(frame),
        "table_files": [str((table_dir / f"{name}.csv").relative_to(root)) for name in tables],
        "figure_stems": figures,
        "uncertainty": "95% Wilson score intervals for binomial error rates",
    }
    output = root / "results" / "publication" / "eda_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_eda(root=args.root, quick=args.quick), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
