"""N5 - learner-disjoint cold-start evaluation and adaptation warm-up curves.

The processed corpus has carried a ``cold_start_split`` column since
preprocessing, and no experiment has ever used it. It is the protocol that
answers the question a school actually asks: how does the system behave for a
learner it has never seen?

An important mechanical detail governs interpretation. Categorical vocabularies
were fitted on the official training partition, so a held-out cold-start learner
still occupies a row of the learner embedding table - but that row receives no
gradient signal during cold-start training (and decays under weight decay), so it
carries no learner-specific information. Any adaptation observed for these
learners therefore comes from the past-only behavioural history features, not
from learner identity. That is the scientifically interesting claim, and this
module is written to measure it rather than assume it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Interaction counts at which a newly arrived learner is scored. Chosen a priori
# on a roughly logarithmic scale rather than tuned against results.
WARM_UP_EDGES: tuple[float, ...] = (0, 1, 3, 5, 10, 20, 50, 100, 200, np.inf)


def warm_up_curve(
    predictions: pd.DataFrame,
    *,
    history_column: str = "prior_interactions",
    edges: Iterable[float] = WARM_UP_EDGES,
    min_bin_count: int = 200,
) -> pd.DataFrame:
    """Discrimination as a function of how much history a learner has accrued.

    For cold-start learners this traces how quickly the model becomes useful for
    somebody it has never seen before.
    """

    required = {"label", "probability", history_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Warm-up curve needs columns: {sorted(missing)}")
    bins = pd.cut(predictions[history_column], bins=list(edges), right=False, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for interval, group in predictions.groupby(bins, observed=True):
        if len(group) < min_bin_count:
            continue
        both_classes = group["label"].nunique() == 2
        rows.append(
            {
                "history_lower": float(interval.left),
                "history_upper": float(interval.right),
                "n": int(len(group)),
                "positive_rate": float(group["label"].mean()),
                "auroc": float(roc_auc_score(group["label"], group["probability"])) if both_classes else float("nan"),
                "auprc": float(average_precision_score(group["label"], group["probability"])) if both_classes else float("nan"),
                "brier": float(np.mean((group["probability"] - group["label"]) ** 2)),
                "mean_probability": float(group["probability"].mean()),
            }
        )
    if not rows:
        raise ValueError("No history bin met the minimum count requirement")
    return pd.DataFrame(rows)


def adaptation_slope(curve: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether performance actually improves as history accumulates."""

    usable = curve.dropna(subset=["auroc"])
    if len(usable) < 3:
        return {"status": "insufficient_bins", "populated_bins": int(len(usable))}
    x = np.log1p(usable["history_lower"].to_numpy(dtype=float))
    y = usable["auroc"].to_numpy(dtype=float)
    weights = usable["n"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1, w=np.sqrt(weights))
    return {
        "status": "fitted",
        "auroc_per_log_interaction": float(slope),
        "intercept": float(intercept),
        "first_bin_auroc": float(y[0]),
        "last_bin_auroc": float(y[-1]),
        "total_gain": float(y[-1] - y[0]),
        "populated_bins": int(len(usable)),
    }


def compare_protocols(
    chronological: pd.DataFrame,
    cold_start: pd.DataFrame,
) -> dict[str, Any]:
    """Quantify the generalization penalty imposed by unseen learners."""

    def _summary(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "n": int(len(frame)),
            "learners": int(frame["learner_id"].nunique()) if "learner_id" in frame else -1,
            "positive_rate": float(frame["label"].mean()),
            "auroc": float(roc_auc_score(frame["label"], frame["probability"])),
            "auprc": float(average_precision_score(frame["label"], frame["probability"])),
            "brier": float(np.mean((frame["probability"] - frame["label"]) ** 2)),
        }

    warm, cold = _summary(chronological), _summary(cold_start)
    return {
        "chronological_personalized": warm,
        "learner_disjoint_cold_start": cold,
        "auroc_penalty": float(warm["auroc"] - cold["auroc"]),
        "auprc_penalty": float(warm["auprc"] - cold["auprc"]),
        "note": (
            "The two protocols evaluate different learner populations and different "
            "numbers of observations; the penalty describes protocol difficulty, not a "
            "paired significance test."
        ),
    }


def run(root: Path, *, model_tag: str = "cold_start_nulkt", seed: int = 42) -> dict[str, Any]:
    """Analyse cold-start artifacts produced by the cold-start training stage."""

    analysis_dir = root / "results" / "analysis" / "cold_start"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dir = root / "results" / "raw_runs" / model_tag / f"seed_{seed}"
    predictions_path = run_dir / "test_predictions.csv"
    if not predictions_path.is_file():
        payload = {
            "status": "unavailable",
            "reason": f"cold-start predictions absent at {predictions_path}",
            "remedy": "python run_all.py --stage cold_start",
        }
        (analysis_dir / "cold_start_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    predictions = pd.read_csv(predictions_path)
    curve = warm_up_curve(predictions)
    curve.to_csv(analysis_dir / "warm_up_curve.csv", index=False)
    payload: dict[str, Any] = {
        "status": "available",
        "model_tag": model_tag,
        "seed": seed,
        "protocol": "learner_disjoint_cold_start",
        "warm_up_curve": curve.to_dict(orient="records"),
        "adaptation": adaptation_slope(curve),
        "mechanism_note": (
            "Held-out cold-start learners receive no gradient on their identity embedding, "
            "so adaptation across history bins is attributable to past-only behavioural "
            "features rather than to a learned learner representation."
        ),
    }

    warm_path = root / "results" / "raw_runs" / "proposed" / f"seed_{seed}" / "test_predictions.csv"
    if warm_path.is_file():
        payload["protocol_comparison"] = compare_protocols(pd.read_csv(warm_path), predictions)
    (analysis_dir / "cold_start_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
