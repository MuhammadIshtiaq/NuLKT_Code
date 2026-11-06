"""Parsing helpers for supplied Universal Dependencies linguistic features."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def parse_morphology(value: str) -> dict[str, str]:
    """Parse ``Key=Value|...`` morphology, retaining supplied fine POS (fPOS)."""

    result: dict[str, str] = {}
    for item in str(value).split("|"):
        key, separator, entry = item.partition("=")
        if separator:
            result[key] = entry
    return result


def morphology_inventory(values: Iterable[str]) -> Counter[str]:
    """Count morphology attribute names without using response labels."""

    counts: Counter[str] = Counter()
    for value in values:
        counts.update(parse_morphology(value))
    return counts


def coarse_morphology_signature(value: str) -> str:
    """Create a stable categorical signature from sorted feature names/values."""

    parsed = parse_morphology(value)
    return "|".join(f"{key}={parsed[key]}" for key in sorted(parsed)) or "<NONE>"
