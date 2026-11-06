"""Project learned learner embeddings into two dimensions, coloured by ability.

The model carries a per-learner embedding table. If personalization is doing
anything, learners who behave similarly should sit near each other in that
space, and the projection should show accuracy varying smoothly across it rather
than scattered at random.

This figure is diagnostic and worth reading sceptically. A neighbourhood
structure that lines up with accuracy shows the embedding encodes ability; it
does *not* show the model uses that encoding to predict better - the ablation
table is what speaks to that. A projection that looks like noise is the more
informative outcome, because it would say the embedding table is not learning
learner structure at all.

t-SNE distances are not metric: cluster *positions* and inter-cluster distances
carry no meaning, only local neighbourhood membership does. Labels on the figure
are therefore attached to individual learners, never to invented cluster names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import configure_publication_style, save_figure

MUTED = "#898781"
INK = "#0b0b0b"


def _load_learner_embedding(checkpoint_path: Path) -> np.ndarray | None:
    """Read the learner embedding matrix out of a training checkpoint."""
    import torch

    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    for key in ("learner_embedding.weight", "module.learner_embedding.weight"):
        if key in state:
            return state[key].detach().cpu().numpy()
    return None


def _learner_accuracy(predictions_path: Path) -> pd.DataFrame | None:
    """Observed correctness per learner, from held-out predictions."""
    if not predictions_path.is_file():
        return None
    frame = pd.read_csv(predictions_path, usecols=["label", "learner_id"])
    grouped = frame.groupby("learner_id")["label"].agg(["mean", "size"])
    # ``label`` is 1 for an error, so accuracy is its complement.
    grouped["accuracy"] = 1.0 - grouped["mean"]
    return grouped.reset_index().rename(columns={"size": "interactions"})


def learner_embedding_map(
    results_root: Path,
    output: Path,
    *,
    run: str = "enhanced",
    seed: int = 42,
    perplexity: float = 30.0,
    min_interactions: int = 30,
    annotate: int = 4,
) -> list[str] | None:
    """Two-dimensional map of the learner embedding, coloured by observed accuracy."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    run_dir = results_root / "raw_runs" / run / f"seed_{seed}"
    embedding = _load_learner_embedding(run_dir / "checkpoints" / f"best_seed_{seed}.pt")
    accuracy = _learner_accuracy(run_dir / "validation_predictions.csv")
    if embedding is None or accuracy is None or accuracy.empty:
        return None

    # Learners with very few interactions have accuracies dominated by sampling
    # noise, which would colour the map with something other than ability.
    accuracy = accuracy[accuracy["interactions"] >= min_interactions]
    accuracy = accuracy[accuracy["learner_id"] < len(embedding)]
    if len(accuracy) < 10:
        return None

    vectors = embedding[accuracy["learner_id"].to_numpy()]
    perplexity = float(min(perplexity, max(5.0, (len(vectors) - 1) / 3.0)))
    projected = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=42, max_iter=1000,
    ).fit_transform(vectors)

    values = accuracy["accuracy"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1], c=values, cmap="RdYlBu",
        s=26, linewidths=0.3, edgecolors="white", alpha=0.9,
    )
    bar = fig.colorbar(scatter, ax=ax, pad=0.02)
    bar.set_label("Learner accuracy (held-out)")
    bar.outline.set_visible(False)

    # Annotate a spread of ability levels so the reader can anchor the colour
    # scale to concrete learners rather than trusting the gradient alone.
    order = np.argsort(values)
    picks = np.unique(np.linspace(0, len(order) - 1, annotate).astype(int))
    for rank in picks:
        index = order[rank]
        ax.annotate(
            f"Acc {values[index]:.4f}",
            xy=(projected[index, 0], projected[index, 1]),
            xytext=(projected[index, 0] + 6, projected[index, 1] - 9),
            fontsize=10, color=INK,
            arrowprops={"arrowstyle": "-", "color": "#2a5d78", "linewidth": 1.2,
                        "connectionstyle": "arc3,rad=0.25"},
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#c3c2b7")
    ax.set_xlabel(f"t-SNE of learner embedding ({len(vectors)} learners, perplexity {perplexity:.0f})")
    paths = save_figure(fig, output, include_svg=True)
    plt.close(fig)
    return [str(path) for path in paths]


def generate_embedding_figures(results_root: Path) -> dict[str, dict[str, Any]]:
    """Render the embedding map for whichever runs have a usable checkpoint."""
    configure_publication_style()
    figure_dir = results_root / "figures" / "embedding"
    figure_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict[str, Any]] = {}
    for run in ("enhanced", "proposed"):
        files = learner_embedding_map(results_root, figure_dir / f"learner_embedding_{run}", run=run)
        registry[f"learner_embedding_{run}"] = (
            {"status": "available", "files": files}
            if files
            else {"status": "unavailable", "reason": f"no usable checkpoint or predictions for {run}"}
        )
    (figure_dir / "embedding_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry
