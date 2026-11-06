"""N4 - selective prediction, uncertainty quality, and teacher-budget curves.

The architecture emits a Beta-evidential uncertainty and an abstention flag that
no existing stage evaluates. This module asks two questions a deployment review
would ask:

1. Does the evidential uncertainty rank errors better than cheaper alternatives
   (Monte-Carlo dropout, predictive entropy) and better than random abstention?
2. Under a fixed teacher review budget, how much of the real error mass does the
   system actually surface?

Random abstention is included deliberately: a risk-coverage curve that is not
compared against it can look impressive purely because easy items are abundant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(y: np.ndarray, p: np.ndarray, u: np.ndarray) -> None:
    if not (len(y) == len(p) == len(u)) or len(y) == 0:
        raise ValueError("inputs must be non-empty and equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not (np.isfinite(p).all() and np.isfinite(u).all()):
        raise ValueError("probabilities and uncertainty must be finite")


def risk_coverage(
    y_true: Iterable[int],
    probability: Iterable[float],
    uncertainty: Iterable[float],
    *,
    grid: Iterable[float] = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2)),
) -> pd.DataFrame:
    """Selective performance as the most uncertain items are handed off.

    At each coverage level the retained subset is the most confident fraction.
    Ranking metrics are reported as ``nan`` where a retained subset is single
    class, rather than being silently imputed.
    """

    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    u = np.asarray(list(uncertainty), dtype=float)
    _validate(y, p, u)
    order = np.argsort(u, kind="stable")
    y, p = y[order], p[order]
    total = len(y)
    rows: list[dict[str, Any]] = []
    for coverage in grid:
        keep = max(1, int(round(float(coverage) * total)))
        ys, ps = y[:keep], p[:keep]
        predicted = (ps >= 0.5).astype(int)
        both_classes = len(np.unique(ys)) == 2
        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": int(keep),
                "selective_accuracy": float(np.mean(predicted == ys)),
                "selective_error": float(np.mean(predicted != ys)),
                "selective_auroc": float(roc_auc_score(ys, ps)) if both_classes else float("nan"),
                "selective_auprc": float(average_precision_score(ys, ps)) if both_classes else float("nan"),
                "selective_brier": float(np.mean((ps - ys) ** 2)),
                "retained_positive_rate": float(np.mean(ys)),
            }
        )
    return pd.DataFrame(rows)


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    """Trapezoidal AURC over the covered range; lower is better."""

    coverage = curve["coverage"].to_numpy(dtype=float)
    risk = curve["selective_error"].to_numpy(dtype=float)
    if len(coverage) < 2:
        raise ValueError("AURC needs at least two coverage points")
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0]))


def compare_uncertainty_signals(
    y_true: Iterable[int],
    probability: Iterable[float],
    signals: dict[str, Iterable[float]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Rank competing uncertainty estimators, always against a random control."""

    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    rng = np.random.default_rng(seed)
    enriched = {name: np.asarray(list(values), dtype=float) for name, values in signals.items()}
    enriched["random_control"] = rng.random(len(y))
    # An oracle ordering bounds how much any estimator could possibly achieve.
    enriched["oracle"] = np.abs(p - y)

    summary: dict[str, Any] = {}
    curves: dict[str, pd.DataFrame] = {}
    for name, values in enriched.items():
        curve = risk_coverage(y, p, values)
        curves[name] = curve
        at_80 = curve.loc[np.isclose(curve["coverage"], 0.80)]
        summary[name] = {
            "aurc": area_under_risk_coverage(curve),
            "selective_accuracy_at_80_coverage": float(at_80["selective_accuracy"].iloc[0]) if len(at_80) else None,
            "selective_error_at_80_coverage": float(at_80["selective_error"].iloc[0]) if len(at_80) else None,
        }
    ranked = sorted(
        (name for name in summary if name not in {"oracle", "random_control"}),
        key=lambda name: summary[name]["aurc"],
    )
    random_aurc = summary["random_control"]["aurc"]
    oracle_aurc = summary["oracle"]["aurc"]
    for name in summary:
        denominator = random_aurc - oracle_aurc
        summary[name]["normalized_skill"] = (
            float((random_aurc - summary[name]["aurc"]) / denominator) if abs(denominator) > 1e-12 else None
        )
    return {"summary": summary, "ranking": ranked, "curves": curves}


def teacher_budget_curve(
    y_true: Iterable[int],
    probability: Iterable[float],
    *,
    budgets: Iterable[float] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Error mass captured when only the top-risk fraction is reviewed.

    This is the operating characteristic a teacher actually faces: given time to
    review ``budget`` of a learner cohort's upcoming items, what share of the
    items that would genuinely be answered incorrectly is surfaced?
    """

    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probability), dtype=float)
    if not (len(y) == len(p)) or len(y) == 0:
        raise ValueError("inputs must be non-empty and equal length")
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    total_errors = int(ranked.sum())
    if total_errors == 0:
        raise ValueError("no positive (error) labels are present")
    total = len(ranked)
    prevalence = total_errors / total
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        reviewed = max(1, int(round(float(budget) * total)))
        captured = int(ranked[:reviewed].sum())
        rows.append(
            {
                "review_budget": float(budget),
                "items_reviewed": reviewed,
                "errors_captured": captured,
                "error_recall": captured / total_errors,
                "review_precision": captured / reviewed,
                "lift_over_random": (captured / reviewed) / prevalence,
            }
        )
    return pd.DataFrame(rows)


def run(
    root: Path,
    *,
    seed: int = 42,
    split: str = "test",
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
    mc_passes: int = 20,
) -> dict[str, Any]:
    """Execute N4 against the frozen proposed checkpoint."""

    from src.analysis.inference import (
        collect_predictions,
        indexed_dataloader,
        load_proposed_model,
        monte_carlo_dropout,
    )
    from src.evaluation.uncertainty import binary_entropy

    output_dir = root / "results" / "analysis" / "selective_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    loader, _ = indexed_dataloader(
        root, split=split, sequence_length=sequence_length, batch_size=batch_size
    )
    predictions = collect_predictions(model, loader, resolved_device)
    dropout = monte_carlo_dropout(model, loader, resolved_device, passes=mc_passes, seed=seed)
    merged = predictions.merge(dropout, on="row_index", how="left", validate="one_to_one")
    merged["predictive_entropy"] = binary_entropy(merged["logit_probability"].clip(1e-9, 1 - 1e-9))
    merged.to_csv(output_dir / f"{split}_uncertainty.csv.gz", index=False, compression="gzip")

    comparison = compare_uncertainty_signals(
        merged["label"],
        merged["logit_probability"],
        {
            "beta_evidential": merged["uncertainty"],
            "monte_carlo_dropout": merged["mc_std"],
            "predictive_entropy": merged["predictive_entropy"],
        },
        seed=seed,
    )
    for name, curve in comparison.pop("curves").items():
        curve.to_csv(output_dir / f"risk_coverage_{name}.csv", index=False)

    budget = teacher_budget_curve(merged["label"], merged["logit_probability"])
    budget.to_csv(output_dir / "teacher_budget_curve.csv", index=False)

    abstention_rate = float((merged["uncertainty"] >= model.config.uncertainty_threshold).mean())
    payload = {
        "provenance": provenance,
        "partition": split,
        "observations": int(len(merged)),
        "monte_carlo_passes": mc_passes,
        "uncertainty_comparison": comparison,
        "configured_abstention_threshold": float(model.config.uncertainty_threshold),
        "abstention_rate_at_configured_threshold": abstention_rate,
        "teacher_budget": budget.to_dict(orient="records"),
        "note": (
            "Selective metrics describe triage quality on observational data. They do not "
            "establish that reviewing a flagged item improves learning outcomes."
        ),
    }
    (output_dir / "selective_prediction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
