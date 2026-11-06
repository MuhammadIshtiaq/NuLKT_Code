"""Derive the exercise-context columns the sequence model was never given.

``results/analysis/feature_gap.json`` measures which signals the boosted model
splits on that the sequence model never receives. Two of the largest are not
stored in the corpus at all - they are derivable from it, which is why the
boosted benchmark had them and the sequence model did not:

* **neighbouring token identity** (``following_token_id`` 6.0% of boosted gain,
  ``preceding_token_id`` 3.4%) - the words either side of the target inside the
  same exercise;
* **token-level recency** (``hours_since_token_last_seen`` 1.8%) - how long since
  this learner last met this particular word.

Both are safe under the project's leakage rules, for different reasons.
Neighbouring tokens are the exercise's own surface form, which the task hands
over at prediction time and which the model already attends to under
``within_exercise_bidirectional``; naming them explicitly changes nothing about
what is knowable. Token recency is computed from strictly earlier rows of the
same learner, so it is past-only in the sense ``audit_past_only_features``
enforces - the current row never contributes to its own value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DERIVED_COLUMNS = (
    "preceding_token_code",
    "following_token_code",
    "hours_since_token_last_seen",
    "token_last_seen_observed",
    "previous_exercise_error_rate",
    "previous_exercise_length",
    "exercise_error_ema_short",
    "exercise_error_ema_long",
)


def add_neighbour_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the token ids adjacent to each token inside its own exercise.

    Shifts are taken within ``exercise_id`` so a neighbour never leaks across an
    exercise boundary; the first and last token of an exercise get 0, which the
    embedding treats as padding.
    """
    if "exercise_id" not in frame.columns or "token_id" not in frame.columns:
        frame["preceding_token_code"] = 0
        frame["following_token_code"] = 0
        return frame
    order = frame.sort_values(["exercise_id", "token_index"], kind="stable").index
    ordered = frame.loc[order]
    grouped = ordered.groupby("exercise_id", sort=False)["token_id"]
    frame.loc[order, "preceding_token_code"] = grouped.shift(1).fillna(0).astype("int64").to_numpy()
    frame.loc[order, "following_token_code"] = grouped.shift(-1).fillna(0).astype("int64").to_numpy()
    frame["preceding_token_code"] = frame["preceding_token_code"].fillna(0).astype("int64")
    frame["following_token_code"] = frame["following_token_code"].fillna(0).astype("int64")
    return frame


def add_token_recency(frame: pd.DataFrame) -> pd.DataFrame:
    """Hours since this learner last encountered this token, from earlier rows only.

    ``shift(1)`` inside each (learner, token) group is what keeps this past-only:
    the value attached to a row is the timestamp of that learner's *previous*
    encounter with the token, never the current one. First encounters carry 0
    with the companion indicator set to 0, so "never seen before" is
    distinguishable from "seen a moment ago" rather than colliding at the same
    numeric value.
    """
    required = {"user_id_id", "token_id", "timestamp_days"}
    if not required.issubset(frame.columns):
        frame["hours_since_token_last_seen"] = 0.0
        frame["token_last_seen_observed"] = 0.0
        return frame
    sort_keys = ["user_id_id", "token_id", "timestamp_days"]
    if "row_order" in frame.columns:
        sort_keys.append("row_order")
    order = frame.sort_values(sort_keys, kind="stable").index
    ordered = frame.loc[order]
    previous = ordered.groupby(["user_id_id", "token_id"], sort=False)["timestamp_days"].shift(1)
    gap_days = ordered["timestamp_days"] - previous
    observed = previous.notna()
    hours = (gap_days.to_numpy(dtype="float64") * 24.0)
    hours = np.where(np.isfinite(hours), np.clip(hours, 0.0, 24.0 * 365.0), 0.0)
    frame.loc[order, "hours_since_token_last_seen"] = hours
    frame.loc[order, "token_last_seen_observed"] = observed.to_numpy(dtype="float64")
    frame["hours_since_token_last_seen"] = frame["hours_since_token_last_seen"].fillna(0.0).astype("float32")
    frame["token_last_seen_observed"] = frame["token_last_seen_observed"].fillna(0.0).astype("float32")
    return frame


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every derived exercise-context column, in place, and return the frame."""
    if all(column in frame.columns for column in DERIVED_COLUMNS):
        return frame
    frame = add_neighbour_tokens(frame)
    frame = add_token_recency(frame)
    frame = add_completed_exercise_state(frame)
    return frame


def add_completed_exercise_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach state computed only from exercises completed before the current one.

    The exercise aggregate is shifted before being joined back to token rows.
    Thus every token in an exercise receives the same frozen prefix state and no
    current-exercise label can enter its own predictors.
    """

    required = {"user_id", "exercise_id", "timestamp_days", "row_order", "label"}
    columns = (
        "previous_exercise_error_rate",
        "previous_exercise_length",
        "exercise_error_ema_short",
        "exercise_error_ema_long",
    )
    if not required.issubset(frame.columns):
        for column in columns:
            frame[column] = 0.0
        return frame
    ordered = frame.sort_values(
        ["user_id", "timestamp_days", "row_order"], kind="stable"
    )
    exercises = (
        ordered.groupby(["user_id", "exercise_id"], sort=False, observed=True)
        .agg(
            exercise_timestamp=("timestamp_days", "min"),
            exercise_row_order=("row_order", "min"),
            exercise_error_rate=("label", "mean"),
            exercise_length=("label", "size"),
        )
        .reset_index()
        .sort_values(
            ["user_id", "exercise_timestamp", "exercise_row_order"], kind="stable"
        )
    )
    grouped = exercises.groupby("user_id", sort=False, observed=True)
    exercises["previous_exercise_error_rate"] = grouped["exercise_error_rate"].shift(1)
    exercises["previous_exercise_length"] = grouped["exercise_length"].shift(1)
    for name, alpha in (("exercise_error_ema_short", 0.5), ("exercise_error_ema_long", 0.125)):
        exercises[name] = grouped["exercise_error_rate"].transform(
            lambda values: values.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )
    state = exercises.set_index(["user_id", "exercise_id"])[list(columns)]
    keys = pd.MultiIndex.from_frame(frame[["user_id", "exercise_id"]])
    aligned = state.reindex(keys)
    for column in columns:
        frame[column] = aligned[column].fillna(0.0).to_numpy(dtype=np.float32)
    return frame
