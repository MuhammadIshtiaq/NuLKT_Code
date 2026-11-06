"""Cluster-aware statistical comparisons for longitudinal learner data."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


Metric = Callable[[np.ndarray, np.ndarray], float]


def metric_function(name: str) -> Metric:
    key = name.lower()
    if key == "auroc":
        return lambda y, p: float(roc_auc_score(y, p))
    if key == "auprc":
        return lambda y, p: float(average_precision_score(y, p))
    if key == "brier_score":
        return lambda y, p: float(np.mean((p - y) ** 2))
    if key == "log_loss":
        return lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))))
    raise ValueError(f"unsupported metric: {name}")


def _validate(y: Iterable[int], a: Iterable[float], b: Iterable[float], learner: Iterable[Any]) -> tuple[np.ndarray, ...]:
    arrays = (np.asarray(list(y), dtype=np.int8), np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float), np.asarray(list(learner)))
    if len({len(item) for item in arrays}) != 1 or len(arrays[0]) == 0:
        raise ValueError("all inputs must be non-empty and equal length")
    if not np.isin(arrays[0], [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(arrays[1]).all() or not np.isfinite(arrays[2]).all():
        raise ValueError("probabilities must be finite")
    return arrays


def learner_clustered_bootstrap(
    y_true: Iterable[int],
    probability_a: Iterable[float],
    probability_b: Iterable[float],
    learner_id: Iterable[Any],
    metric: str = "auroc",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap that resamples learners, preserving within-learner rows."""
    y, a, b, groups = _validate(y_true, probability_a, probability_b, learner_id)
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100 for inferential use")
    fn = metric_function(metric)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two learners")
    group_rows = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    skipped = 0
    for _ in range(n_bootstrap):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_rows[group] for group in selected])
        if np.unique(y[indices]).size < 2 and metric in {"auroc", "auprc"}:
            skipped += 1
            continue
        differences.append(fn(y[indices], a[indices]) - fn(y[indices], b[indices]))
    if len(differences) < max(100, int(0.8 * n_bootstrap)):
        raise RuntimeError("too many invalid bootstrap replicates; inference is unreliable")
    diff = np.asarray(differences)
    alpha = 1 - confidence
    observed = fn(y, a) - fn(y, b)
    lower_tail = (np.count_nonzero(diff <= 0) + 1) / (len(diff) + 1)
    upper_tail = (np.count_nonzero(diff >= 0) + 1) / (len(diff) + 1)
    p_value = float(min(1.0, 2 * min(lower_tail, upper_tail)))
    return {
        "method": "paired learner-clustered percentile bootstrap",
        "metric": metric,
        "higher_is_better": metric in {"auroc", "auprc"},
        "model_a_minus_b": float(observed),
        "ci_lower": float(np.quantile(diff, alpha / 2)),
        "ci_upper": float(np.quantile(diff, 1 - alpha / 2)),
        "confidence": confidence,
        "raw_p_value": p_value,
        "replicates_requested": n_bootstrap,
        "replicates_valid": len(differences),
        "replicates_skipped": skipped,
        "cluster_count": int(len(unique)),
        "seed": seed,
    }


def learner_level_wilcoxon(
    y_true: Iterable[int], probability_a: Iterable[float], probability_b: Iterable[float], learner_id: Iterable[Any]
) -> dict[str, Any]:
    """Compare per-learner Brier losses; positive effect favors model A."""
    y, a, b, groups = _validate(y_true, probability_a, probability_b, learner_id)
    unique = np.unique(groups)
    loss_a = np.asarray([np.mean((a[groups == g] - y[groups == g]) ** 2) for g in unique])
    loss_b = np.asarray([np.mean((b[groups == g] - y[groups == g]) ** 2) for g in unique])
    delta = loss_b - loss_a
    if not np.any(delta):
        return {"method": "Wilcoxon signed-rank on learner Brier loss", "statistic": 0.0, "raw_p_value": 1.0, "effect_median": 0.0, "learner_count": int(len(unique))}
    result = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
    return {
        "method": "Wilcoxon signed-rank on learner Brier loss",
        "statistic": float(result.statistic),
        "raw_p_value": float(result.pvalue),
        "effect_median": float(np.median(delta)),
        "effect_definition": "median learner Brier(model B) - Brier(model A); positive favors A",
        "learner_count": int(len(unique)),
    }


def mcnemar_exact(y_true: Iterable[int], prediction_a: Iterable[int], prediction_b: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(prediction_a), dtype=int)
    b = np.asarray(list(prediction_b), dtype=int)
    if len({len(y), len(a), len(b)}) != 1 or len(y) == 0:
        raise ValueError("all inputs must be non-empty and equal length")
    correct_a, correct_b = a == y, b == y
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue)
    return {"method": "exact McNemar/binomial test", "a_correct_b_wrong": a_only, "a_wrong_b_correct": b_only, "raw_p_value": p}


def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> dict[str, list]:
    """Return Holm-adjusted p-values and step-down rejection decisions."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a non-empty collection in [0, 1]")
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, (len(p) - np.arange(len(p))) * sorted_p))
    rejected_sorted = np.zeros(len(p), dtype=bool)
    continue_rejecting = True
    for rank, value in enumerate(sorted_p):
        if continue_rejecting and value <= alpha / (len(p) - rank):
            rejected_sorted[rank] = True
        else:
            continue_rejecting = False
    adjusted = np.empty_like(adjusted_sorted)
    rejected = np.empty_like(rejected_sorted)
    adjusted[order] = adjusted_sorted
    rejected[order] = rejected_sorted
    return {"adjusted_p_value": adjusted.tolist(), "reject": rejected.tolist(), "alpha": alpha, "method": "Holm-Bonferroni"}


def compare_many_models(
    y_true: Iterable[int],
    reference: Iterable[float],
    competitors: dict[str, Iterable[float]],
    learner_id: Iterable[Any],
    *,
    metric: str = "auroc",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare one reference model against several competitors with Holm control.

    Running a separate two-model bootstrap per competitor and reporting each raw
    p-value would inflate the family-wise error rate in proportion to the number
    of baselines. Every comparison is made against the same reference on the same
    rows, and the resulting p-values are corrected jointly.
    """

    reference_values = np.asarray(list(reference), dtype=float)
    comparisons: list[dict[str, Any]] = []
    for name, probabilities in competitors.items():
        result = learner_clustered_bootstrap(
            y_true, reference_values, probabilities, learner_id,
            metric=metric, n_bootstrap=n_bootstrap, seed=seed,
        )
        comparisons.append({"competitor": name, **result})
    if not comparisons:
        raise ValueError("at least one competitor is required")
    correction = holm_bonferroni([row["raw_p_value"] for row in comparisons], alpha=alpha)
    for row, adjusted, rejected in zip(comparisons, correction["adjusted_p_value"], correction["reject"]):
        row["holm_adjusted_p_value"] = float(adjusted)
        row["reject_null_after_correction"] = bool(rejected)
    return {
        "metric": metric,
        "family_size": len(comparisons),
        "alpha": alpha,
        "correction": correction["method"],
        "comparisons": comparisons,
    }


def equivalence_test(
    differences: Iterable[float],
    margin: float,
    *,
    confidence: float = 0.90,
) -> dict[str, Any]:
    """Two one-sided tests for practical equivalence within +/- ``margin``.

    A non-significant difference is not evidence of equivalence. Where two models
    perform similarly, this reports whether the data actually support a claim of
    no meaningful difference, or are merely inconclusive.
    """

    from scipy.stats import t

    values = np.asarray(list(differences), dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite differences are required")
    if margin <= 0:
        raise ValueError("margin must be positive")
    n = len(values)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(n))
    if standard_error == 0:
        equivalent = bool(abs(mean) < margin)
        return {
            "method": "two one-sided tests",
            "mean_difference": mean, "margin": margin,
            "p_lower": 0.0 if equivalent else 1.0, "p_upper": 0.0 if equivalent else 1.0,
            "equivalent": equivalent, "degenerate_zero_variance": True,
        }
    degrees = n - 1
    t_lower = (mean + margin) / standard_error
    t_upper = (mean - margin) / standard_error
    p_lower = float(1 - t.cdf(t_lower, degrees))
    p_upper = float(t.cdf(t_upper, degrees))
    half = float(t.ppf((1 + confidence) / 2, degrees) * standard_error)
    return {
        "method": "two one-sided tests",
        "mean_difference": mean,
        "margin": margin,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
        "confidence": confidence,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_value": max(p_lower, p_upper),
        "equivalent": bool(max(p_lower, p_upper) < (1 - confidence) / 2 * 2),
        "interpretation": (
            "Equivalence is claimed only when the whole confidence interval lies inside "
            "the margin; otherwise the comparison is inconclusive, not null."
        ),
    }


def seed_variance_decomposition(
    seed_metrics: dict[str, Iterable[float]],
    *,
    metric_name: str = "auroc",
) -> dict[str, Any]:
    """Split observed metric variance into between-model and within-model parts.

    If run-to-run seed variance is comparable to the gap between models, a
    headline improvement is not reproducible, however large the point estimate.
    """

    cleaned = {
        name: np.asarray(list(values), dtype=float)
        for name, values in seed_metrics.items()
        if len(list(values)) > 0
    }
    if len(cleaned) < 2:
        raise ValueError("at least two models are required")
    counts = {name: len(values) for name, values in cleaned.items()}
    means = {name: float(values.mean()) for name, values in cleaned.items()}
    grand_mean = float(np.mean(np.concatenate(list(cleaned.values()))))

    between = sum(counts[name] * (means[name] - grand_mean) ** 2 for name in cleaned)
    within = sum(float(np.sum((values - means[name]) ** 2)) for name, values in cleaned.items())
    total_observations = sum(counts.values())
    between_df = len(cleaned) - 1
    within_df = total_observations - len(cleaned)

    between_ms = between / between_df if between_df > 0 else float("nan")
    within_ms = within / within_df if within_df > 0 else float("nan")
    ordered = sorted(means, key=means.get, reverse=True)
    best_gap = means[ordered[0]] - means[ordered[1]]
    pooled_sd = float(np.sqrt(within_ms)) if np.isfinite(within_ms) else float("nan")
    return {
        "metric": metric_name,
        "model_means": means,
        "seeds_per_model": counts,
        "between_model_mean_square": float(between_ms),
        "within_model_seed_mean_square": float(within_ms),
        "pooled_seed_sd": pooled_sd,
        "variance_ratio": float(between_ms / within_ms) if np.isfinite(within_ms) and within_ms > 0 else None,
        "top_two_gap": float(best_gap),
        "gap_in_pooled_seed_sd_units": float(best_gap / pooled_sd) if pooled_sd and pooled_sd > 0 else None,
        "gap_exceeds_seed_noise": bool(pooled_sd and pooled_sd > 0 and abs(best_gap) > 2 * pooled_sd),
        "note": (
            "A gap smaller than roughly two pooled seed standard deviations should not be "
            "described as an improvement, regardless of any single-seed comparison."
        ),
    }


def mean_sd_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float | int]:
    from scipy.stats import t

    x = np.asarray(list(values), dtype=float)
    if x.size < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite repeated-run values are required")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(t.ppf((1 + confidence) / 2, df=len(x) - 1) * sd / np.sqrt(len(x)))
    return {"n": int(len(x)), "mean": mean, "sd": sd, "ci_lower": mean - half, "ci_upper": mean + half, "confidence": confidence}
