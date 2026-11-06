"""Causal learner-sequence construction with padding masks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceWindow:
    learner_id: str
    row_indices: np.ndarray
    attention_mask: np.ndarray
    # True where the position is to be scored. Positions carried purely as
    # attention context are False. Defaults to the attention mask so existing
    # non-overlapping windows behave exactly as before.
    score_mask: np.ndarray | None = None

    def scored(self) -> np.ndarray:
        return self.attention_mask if self.score_mask is None else self.score_mask


def causal_mask(length: int) -> np.ndarray:
    """Return a boolean mask where true entries are future/inaccessible."""

    if length <= 0:
        raise ValueError("length must be positive")
    return np.triu(np.ones((length, length), dtype=bool), k=1)


def iter_sequence_windows(
    frame: pd.DataFrame,
    *,
    max_length: int,
    stride: int | None = None,
) -> Iterator[SequenceWindow]:
    """Yield chronological learner windows; no label is used for construction."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    stride = stride or max_length
    if stride <= 0:
        raise ValueError("stride must be positive")
    ordered = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable")
    for learner, group in ordered.groupby("user_id", sort=False, observed=True):
        indices = group.index.to_numpy()
        for start in range(0, len(indices), stride):
            chunk = indices[start : start + max_length]
            if not len(chunk):
                continue
            mask = np.zeros(max_length, dtype=bool)
            mask[: len(chunk)] = True
            yield SequenceWindow(str(learner), chunk, mask)


def iter_context_windows(
    frame: pd.DataFrame,
    *,
    max_length: int,
    context_length: int,
) -> Iterator[SequenceWindow]:
    """Yield overlapping windows whose leading ``context_length`` rows are unscored.

    Non-overlapping windows discard a learner's attention history at every
    boundary: with a 200-token window and a median learner history near 1,100
    tokens, most interactions are predicted from a context that begins abruptly
    mid-trajectory. Here each window carries the preceding ``context_length``
    interactions purely as attention context and scores only the remainder, so
    every interaction is still scored exactly once but sees real history.

    The first window of a learner has no earlier interactions to carry, so its
    rows are all scored - matching the previous behaviour at trajectory starts.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0 <= context_length < max_length:
        raise ValueError("context_length must be non-negative and smaller than max_length")
    if context_length == 0:
        yield from iter_sequence_windows(frame, max_length=max_length)
        return

    step = max_length - context_length
    ordered = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable")
    for learner, group in ordered.groupby("user_id", sort=False, observed=True):
        indices = group.index.to_numpy()
        for scored_start in range(0, len(indices), step):
            window_start = max(0, scored_start - context_length)
            chunk = indices[window_start : scored_start + step]
            if not len(chunk):
                continue
            carried = scored_start - window_start
            attention = np.zeros(max_length, dtype=bool)
            attention[: len(chunk)] = True
            score = np.zeros(max_length, dtype=bool)
            score[carried : len(chunk)] = True
            yield SequenceWindow(str(learner), chunk, attention, score)


def pad_values(values: Sequence[int | float], max_length: int, pad_value: int | float = 0) -> np.ndarray:
    """Pad or truncate a numeric feature sequence without altering order."""

    array = np.asarray(values)
    output = np.full(max_length, pad_value, dtype=array.dtype)
    length = min(len(array), max_length)
    output[:length] = array[:length]
    return output
