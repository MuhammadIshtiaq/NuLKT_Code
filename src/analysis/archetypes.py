"""N3 - learner archetypes discovered from mixture-of-experts routing.

The learner-conditioned router assigns every token a distribution over experts.
Averaged per learner, that distribution is a compact behavioural signature the
model produced without ever being told what kinds of learner exist. This module
clusters those signatures and then asks whether the clusters correspond to
anything an educator would recognize.

Two guards keep the result honest. Cluster count is chosen by silhouette score
over a fixed candidate range rather than picked to suit the story, and every
profile difference is tested against a label-permutation null, because k-means on
any continuous matrix always yields visually distinct groups.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Attributes used only to *describe* clusters after the fact; never to form them.
PROFILE_COLUMNS = (
    "prior_error_rate",
    "recent_20_error_rate",
    "prior_interactions",
    "prior_token_error_rate",
    "elapsed_since_prior_exercise_seconds",
    "label",
)


def learner_routing_signatures(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean expert-utilization vector per learner."""

    expert_columns = sorted(name for name in predictions.columns if name.startswith("expert_"))
    if not expert_columns:
        raise KeyError("No expert_* columns present; collect predictions with expert weights")
    signatures = predictions.groupby("learner_id", observed=True)[expert_columns].mean()
    counts = predictions.groupby("learner_id", observed=True).size().rename("n_interactions")
    return signatures.join(counts)


def choose_clusters(
    signatures: pd.DataFrame,
    *,
    candidates: Iterable[int] = (2, 3, 4, 5, 6, 7, 8),
    seed: int = 42,
) -> dict[str, Any]:
    """Pick k by silhouette score over a pre-declared candidate range."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    expert_columns = [name for name in signatures.columns if name.startswith("expert_")]
    matrix = signatures[expert_columns].to_numpy(dtype=float)
    if len(matrix) < 10:
        raise ValueError("Too few learners to cluster")
    scores: list[dict[str, Any]] = []
    for k in candidates:
        if k >= len(matrix):
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matrix)
        scores.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(matrix, model.labels_)),
                "inertia": float(model.inertia_),
            }
        )
    if not scores:
        raise ValueError("No candidate cluster count was feasible")
    best = max(scores, key=lambda row: row["silhouette"])
    return {"selection": scores, "chosen_k": best["k"], "chosen_silhouette": best["silhouette"]}


def assign_archetypes(signatures: pd.DataFrame, k: int, *, seed: int = 42) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    expert_columns = [name for name in signatures.columns if name.startswith("expert_")]
    model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(signatures[expert_columns].to_numpy(dtype=float))
    assigned = signatures.copy()
    assigned["archetype"] = model.labels_
    return assigned


def profile_archetypes(
    assigned: pd.DataFrame,
    learner_attributes: pd.DataFrame,
    *,
    columns: Iterable[str] = PROFILE_COLUMNS,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Describe clusters by held-out behavioural attributes with a permutation null.

    The statistic is the between-cluster variance of each attribute's cluster
    means. Permuting archetype labels across learners gives the distribution of
    that statistic under no real structure.
    """

    merged = assigned[["archetype"]].join(learner_attributes, how="inner")
    available = [name for name in columns if name in merged.columns]
    if not available:
        raise KeyError("None of the requested profile columns are available")
    rng = np.random.default_rng(seed)
    labels = merged["archetype"].to_numpy()

    profiles = merged.groupby("archetype", observed=True)[available].mean()
    sizes = merged.groupby("archetype", observed=True).size().rename("n_learners")

    tests: dict[str, Any] = {}
    for name in available:
        values = merged[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 10:
            tests[name] = {"status": "insufficient_data"}
            continue
        observed_statistic = float(np.var(pd.Series(values[finite]).groupby(labels[finite]).mean()))
        null = np.empty(n_permutations, dtype=float)
        for index in range(n_permutations):
            shuffled = rng.permutation(labels[finite])
            null[index] = np.var(pd.Series(values[finite]).groupby(shuffled).mean())
        p_value = float((np.count_nonzero(null >= observed_statistic) + 1) / (n_permutations + 1))
        tests[name] = {
            "status": "tested",
            "between_cluster_variance": observed_statistic,
            "permutation_p_value": p_value,
            "n_permutations": int(n_permutations),
        }

    from src.evaluation.statistics import holm_bonferroni

    testable = [value for value in tests.values() if value.get("status") == "tested"]
    if testable:
        correction = holm_bonferroni([value["permutation_p_value"] for value in testable])
        for value, adjusted_p, rejected in zip(
            testable, correction["adjusted_p_value"], correction["reject"]
        ):
            value["holm_adjusted_p_value"] = float(adjusted_p)
            value["significant_at_0_05"] = bool(rejected)

    return {
        "cluster_sizes": sizes.to_dict(),
        "cluster_profiles": profiles.to_dict(orient="index"),
        "profile_tests": tests,
        "note": (
            "Profile attributes were not used to form clusters. Significance indicates the "
            "routing-derived grouping aligns with behaviour, not that the groups are causal "
            "learner types."
        ),
    }


def run(
    root: Path,
    *,
    seed: int = 42,
    split: str = "test",
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Execute N3 against the frozen proposed checkpoint."""

    from src.analysis.inference import (
        collect_predictions,
        indexed_dataloader,
        join_corpus_attributes,
        load_proposed_model,
    )

    output_dir = root / "results" / "analysis" / "archetypes"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    if model.config.ablations.get("moe", False):
        payload = {"status": "unavailable", "reason": "mixture of experts is ablated in this checkpoint"}
        (output_dir / "archetype_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    loader, frame = indexed_dataloader(
        root, split=split, sequence_length=sequence_length, batch_size=batch_size
    )
    predictions = collect_predictions(model, loader, resolved_device)
    enriched = join_corpus_attributes(
        predictions, frame, [name for name in PROFILE_COLUMNS if name != "label"]
    )

    signatures = learner_routing_signatures(predictions)
    signatures.to_csv(output_dir / "learner_routing_signatures.csv")
    selection = choose_clusters(signatures, seed=seed)
    assigned = assign_archetypes(signatures, selection["chosen_k"], seed=seed)
    assigned.to_csv(output_dir / "learner_archetypes.csv")

    learner_attributes = enriched.groupby("learner_id", observed=True)[
        [name for name in PROFILE_COLUMNS if name in enriched.columns]
    ].mean()
    profiles = profile_archetypes(assigned, learner_attributes, seed=seed)

    expert_columns = [name for name in signatures.columns if name.startswith("expert_")]
    utilization = signatures[expert_columns].mean()
    payload = {
        "provenance": provenance,
        "partition": split,
        "learners": int(len(signatures)),
        "experts": len(expert_columns),
        "mean_expert_utilization": utilization.to_dict(),
        "routing_collapse_check": {
            "max_mean_utilization": float(utilization.max()),
            "uniform_share": 1.0 / len(expert_columns),
            "collapsed": bool(utilization.max() > 0.9),
        },
        "cluster_selection": selection,
        "archetype_profiles": profiles,
    }
    (output_dir / "archetype_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
