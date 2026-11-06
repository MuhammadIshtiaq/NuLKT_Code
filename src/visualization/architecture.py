"""Programmatic architecture diagram for NuLKT."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .style import configure_publication_style, save_figure


def _box(ax, x: float, y: float, width: float, height: float, text: str, color: str, fontsize: float = 10.5):
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.018,rounding_size=0.012", facecolor=color, edgecolor="black", linewidth=1.05)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.15)
    return patch


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], dashed: bool = False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.05, color="#222222", linestyle="--" if dashed else "-", connectionstyle="arc3"))


def create_architecture(output_dir: Path) -> list[Path]:
    configure_publication_style()
    fig, ax = plt.subplots(figsize=(7.1, 13.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.998, "NuLKT Architecture", ha="center", va="top", fontsize=14, fontweight="bold")

    _box(ax, 0.5, 0.93, 0.50, 0.045, "Official Duolingo SLAM Dataset", "#DCEAF7", 11)
    _box(ax, 0.5, 0.865, 0.60, 0.047, "Leakage-Safe Chronological Preprocessing", "#E8F1F8", 10.5)
    _arrow(ax, (0.5, 0.905), (0.5, 0.891))

    features = [
        (0.14, "Learner State\nRepresentation", "#E4F3E6"),
        (0.38, "Linguistic\nRepresentation", "#FFF0D5"),
        (0.62, "Exercise\nContext", "#F8E5ED"),
        (0.86, "Temporal\nFeatures", "#E9E4F6"),
    ]
    for x, label, color in features:
        _box(ax, x, 0.795, 0.205, 0.060, label, color, 9.5)
        _arrow(ax, (0.5, 0.839), (x, 0.828))
        _arrow(ax, (x, 0.762), (0.5, 0.724))

    stages = [
        (0.70, "Gated Feature Fusion", "#DCEAF7"),
        (0.625, "Local Causal Transformer", "#DCEAF7"),
        (0.55, "Session Encoder", "#E4F3E6"),
        (0.475, "Time-Aware Global Learner Memory", "#E9E4F6"),
        (0.400, "Learner-Conditioned Mixture-of-Experts", "#FFF0D5"),
        (0.325, "Contrastive Personalized Representation", "#F8E5ED"),
    ]
    for idx, (y, label, color) in enumerate(stages):
        _box(ax, 0.5, y, 0.61 if idx not in {2} else 0.43, 0.047, label, color, 10.3)
        if idx:
            _arrow(ax, (0.5, stages[idx - 1][0] - 0.027), (0.5, y + 0.027))

    _box(ax, 0.27, 0.235, 0.35, 0.060, "Error and Mastery\nPrediction", "#DCEAF7", 10)
    _box(ax, 0.73, 0.235, 0.31, 0.060, "Uncertainty Head", "#E9E4F6", 10)
    _arrow(ax, (0.5, 0.298), (0.27, 0.269))
    _arrow(ax, (0.5, 0.298), (0.73, 0.269))
    _box(ax, 0.5, 0.145, 0.53, 0.050, "Transparent Explicit-Instruction Engine", "#FFF0D5", 10.3)
    _arrow(ax, (0.27, 0.201), (0.44, 0.173))
    _arrow(ax, (0.73, 0.201), (0.56, 0.173))
    _box(ax, 0.5, 0.065, 0.63, 0.052, "Personalized Recommendation or Teacher Review", "#E4F3E6", 10.5)
    _arrow(ax, (0.5, 0.117), (0.5, 0.094))

    _box(ax, 0.10, 0.385, 0.16, 0.060, "Auxiliary\nContrastive Loss", "#F4F4F4", 8.5)
    _arrow(ax, (0.18, 0.365), (0.39, 0.334), dashed=True)
    _box(ax, 0.90, 0.315, 0.16, 0.060, "Calibration\nLoss", "#F4F4F4", 8.5)
    _arrow(ax, (0.82, 0.300), (0.74, 0.267), dashed=True)
    ax.text(0.5, 0.008, "Solid arrows: inference path     Dashed arrows: auxiliary training objectives", ha="center", fontsize=8.5, color="#333333")

    paths = save_figure(fig, output_dir / "architecture", include_svg=True)
    # Required vector filenames omit the DPI suffix.
    pdf = output_dir / "architecture.pdf"
    svg = output_dir / "architecture.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    return [paths[0], pdf, svg]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    create_architecture(args.output_dir)


if __name__ == "__main__":
    main()
