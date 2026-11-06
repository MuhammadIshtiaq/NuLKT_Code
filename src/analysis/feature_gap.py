"""Attribute the boosted model's advantage to inputs the sequence model lacks.

The boosted arm beats the sequence model on the official partition. That could
mean the architecture is wrong, or simply that the two models are not being shown
the same evidence. This separates those explanations by asking, of the boosted
model's total split gain, how much comes from columns that never enter the
sequence model at all.

It is a diagnostic, not a proof: split gain measures what the boosted model used,
not what is available in principle, and a signal the sequence model lacks as a
named column may still be recoverable from its inputs (a transformer that attends
across a sentence can in principle infer neighbouring-token identity). Treat a
large unreachable share as a reason to test adding the inputs, which is what the
``enhanced`` arm does, rather than as a conclusion on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Columns the sequence dataset hands the model, by corpus name rather than by the
# boosted model's encoded name, so the comparison is against real inputs.
SEQUENCE_MODEL_INPUTS = {
    "token", "lemma", "part_of_speech", "morphology", "dependency_label",
    "session_type", "exercise_format", "user_id",
    "prior_interactions", "prior_error_rate", "recent_20_error_rate",
    "prior_token_exposures", "prior_token_error_rate",
    "elapsed_since_prior_exercise_seconds", "timestamp_days", "label",
}

# Inputs the exercise-context arm adds, tracked separately so the study can say
# how much of the gap that arm is positioned to close.
EXERCISE_CONTEXT_INPUTS = {
    "prompt", "response_time_seconds", "session_position", "token_index",
    # Derived by src.data.context_features rather than stored in the corpus.
    "preceding_token", "following_token", "hours_since_token_last_seen",
}


def _base_name(feature: str) -> str:
    return feature[:-3] if feature.endswith("_id") else feature


def run_feature_gap(root: Path) -> dict[str, Any]:
    """Split boosted-model gain into reachable and unreachable shares."""
    importance_path = root / "results" / "raw_runs" / "official_gbdt_benchmark" / "feature_importance.csv"
    if not importance_path.is_file():
        payload = {"status": "unavailable", "reason": f"no feature importance at {importance_path}"}
        destination = root / "results" / "analysis" / "feature_gap.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    importance = pd.read_csv(importance_path)
    total = float(importance["gain"].sum())
    if total <= 0:
        raise ValueError("Feature importance table carries no positive gain")

    rows: list[dict[str, Any]] = []
    for _, row in importance.iterrows():
        feature = str(row["feature"])
        base = _base_name(feature)
        reachable = base in SEQUENCE_MODEL_INPUTS or feature in SEQUENCE_MODEL_INPUTS
        rows.append(
            {
                "feature": feature,
                "gain": float(row["gain"]),
                "gain_fraction": float(row["gain"]) / total,
                "reaches_sequence_model": reachable,
                "added_by_exercise_context": base in EXERCISE_CONTEXT_INPUTS or feature in EXERCISE_CONTEXT_INPUTS,
            }
        )

    table = pd.DataFrame(rows).sort_values("gain_fraction", ascending=False).reset_index(drop=True)
    unreachable = table[~table["reaches_sequence_model"]]
    addressable = unreachable[unreachable["added_by_exercise_context"]]

    payload: dict[str, Any] = {
        "status": "available",
        "source": str(importance_path),
        "total_gain": total,
        "unreachable_gain_fraction": float(unreachable["gain_fraction"].sum()),
        "addressable_gain_fraction": float(addressable["gain_fraction"].sum()),
        "largest_unreachable": (
            {
                "feature": str(unreachable.iloc[0]["feature"]),
                "gain_fraction": float(unreachable.iloc[0]["gain_fraction"]),
            }
            if not unreachable.empty
            else {"feature": None, "gain_fraction": 0.0}
        ),
        "features": table.to_dict(orient="records"),
        "caveat": (
            "Split gain reports what the boosted model used, not what is knowable. "
            "A column absent from the sequence model's inputs may still be partly "
            "recoverable from them."
        ),
    }

    destination = root / "results" / "analysis" / "feature_gap.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table.to_csv(root / "results" / "metrics" / "feature_gap.csv", index=False)
    return payload
