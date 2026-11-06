"""Public helpers for target-shifted learner-history features."""

from __future__ import annotations

import pandas as pd


def prior_outcome_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute learner-level expanding statistics shifted by one exercise.

    The preprocessing pipeline uses an equivalent streaming implementation. This
    tabular helper is useful in tests and small analyses.
    """

    required = {"user_id", "exercise_id", "timestamp_days", "row_order", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing history columns: {sorted(missing)}")
    ordered = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable").copy()
    summaries = (
        ordered.groupby(["user_id", "exercise_id"], sort=False, observed=True)
        .agg(exercise_interactions=("label", "size"), exercise_errors=("label", "sum"))
        .reset_index()
    )
    summaries["prior_interactions"] = summaries.groupby("user_id", sort=False)[
        "exercise_interactions"
    ].transform(lambda series: series.cumsum().shift(fill_value=0))
    summaries["prior_errors"] = summaries.groupby("user_id", sort=False)[
        "exercise_errors"
    ].transform(lambda series: series.cumsum().shift(fill_value=0))
    summaries["prior_error_rate"] = (
        summaries["prior_errors"]
        .div(summaries["prior_interactions"].where(summaries["prior_interactions"] > 0, 1))
        .astype("float32")
    )
    return ordered.merge(
        summaries[["user_id", "exercise_id", "prior_interactions", "prior_errors", "prior_error_rate"]],
        on=["user_id", "exercise_id"],
        how="left",
        validate="many_to_one",
    )
