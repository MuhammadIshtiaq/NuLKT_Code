"""Model-batch validation and quick-mode slicing utilities."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

REQUIRED_SEQUENCE_FIELDS = (
    "token", "lemma", "pos", "morph", "context", "exercise", "format",
    "learner", "responses", "timestamps", "elapsed_seconds", "session",
)


def validate_model_batch(batch: Mapping[str, Tensor]) -> tuple[int, int]:
    missing = [field for field in REQUIRED_SEQUENCE_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"Model batch missing fields: {missing}")
    shape = tuple(batch["responses"].shape)
    if len(shape) != 2:
        raise ValueError("responses and categorical inputs must have shape [batch, sequence]")
    for field in REQUIRED_SEQUENCE_FIELDS:
        if tuple(batch[field].shape[:2]) != shape:
            raise ValueError(f"{field} leading shape {tuple(batch[field].shape[:2])} does not match {shape}")
    if "history_features" in batch and tuple(batch["history_features"].shape[:2]) != shape:
        raise ValueError("history_features leading dimensions do not match responses")
    if "padding_mask" in batch and tuple(batch["padding_mask"].shape) != shape:
        raise ValueError("padding_mask shape does not match responses")
    return shape


def quick_batch(batch: Mapping[str, Tensor], max_examples: int = 8, max_length: int = 50) -> dict[str, Tensor]:
    """Take a deterministic prefix for a quick smoke run; no reordering occurs."""

    validate_model_batch(batch)
    if max_examples < 1 or max_length < 1:
        raise ValueError("max_examples and max_length must be positive")
    return {key: value[:max_examples, :max_length, ...] for key, value in batch.items()}
