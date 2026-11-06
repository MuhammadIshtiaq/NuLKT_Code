"""Deterministic official, chronological, and cold-start split protocols."""

from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd


SplitName = Literal["train", "validation", "test"]


def _validate_ratios(train: float, validation: float) -> None:
    if not (0.0 < train < 1.0 and 0.0 <= validation < 1.0 and train + validation < 1.0):
        raise ValueError("Split ratios must be nonnegative and sum to less than one")


def official_split(frame: pd.DataFrame) -> pd.Series:
    """Map the corpus' published train/dev/test partitions without shuffling."""

    if "official_split" not in frame:
        raise KeyError("official_split")
    mapped = frame["official_split"].map({"train": "train", "dev": "validation", "test": "test"})
    if mapped.isna().any():
        raise ValueError("Unknown official split label")
    return mapped.astype("category")


def chronological_personalized_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> pd.Series:
    """Allocate whole exercises chronologically within each learner.

    Whole-exercise assignment prevents tokens from one submitted response being
    divided across train and evaluation partitions.
    """

    _validate_ratios(train_ratio, validation_ratio)
    required = {"user_id", "exercise_id", "timestamp_days", "row_order"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing split columns: {sorted(missing)}")
    exercises = (
        frame.loc[:, list(required)]
        .sort_values(["user_id", "timestamp_days", "row_order"], kind="stable")
        .drop_duplicates(["user_id", "exercise_id"], keep="first")
    )
    exercises["exercise_position"] = exercises.groupby("user_id", sort=False).cumcount()
    exercises["exercise_count"] = exercises.groupby("user_id", sort=False)["exercise_id"].transform("size")
    fraction = (exercises["exercise_position"] + 1) / exercises["exercise_count"]
    exercises["assigned"] = "test"
    exercises.loc[fraction <= train_ratio, "assigned"] = "train"
    exercises.loc[(fraction > train_ratio) & (fraction <= train_ratio + validation_ratio), "assigned"] = "validation"
    # Very short histories must still contribute training context.
    exercises.loc[exercises["exercise_count"] < 3, "assigned"] = "train"
    lookup = exercises.set_index(["user_id", "exercise_id"])["assigned"]
    keys = pd.MultiIndex.from_frame(frame[["user_id", "exercise_id"]])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=frame.index, dtype="category")


def cold_start_learner_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Create a stable learner-disjoint split using cryptographic hashes."""

    _validate_ratios(train_ratio, validation_ratio)
    if "user_id" not in frame:
        raise KeyError("user_id")
    users = frame["user_id"].astype(str).unique()
    scored = sorted(
        users,
        key=lambda user: hashlib.sha256(f"{seed}:{user}".encode("utf-8")).digest(),
    )
    n_users = len(scored)
    n_train = max(1, int(n_users * train_ratio)) if n_users else 0
    n_val = int(n_users * validation_ratio)
    if n_users >= 3:
        n_val = max(1, n_val)
        n_train = min(n_train, n_users - n_val - 1)
    mapping: dict[str, str] = {}
    for index, user in enumerate(scored):
        mapping[user] = "train" if index < n_train else "validation" if index < n_train + n_val else "test"
    return frame["user_id"].astype(str).map(mapping).astype("category")


def assert_split_integrity(frame: pd.DataFrame, split_column: str) -> None:
    """Fail on duplicates, unknown partitions, or cold-start user overlap."""

    if frame["instance_id"].duplicated().any():
        raise RuntimeError("Duplicate instance IDs would contaminate split partitions")
    unknown = set(frame[split_column].dropna().astype(str)) - {"train", "validation", "test"}
    if unknown:
        raise RuntimeError(f"Unknown split labels: {sorted(unknown)}")
    if split_column == "cold_start_split":
        counts = frame.groupby("user_id", observed=True)[split_column].nunique()
        if (counts > 1).any():
            raise RuntimeError("Cold-start learners overlap across partitions")
