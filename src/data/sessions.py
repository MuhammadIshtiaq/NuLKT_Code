"""Session identifiers and leakage-safe session segmentation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class SessionBoundary:
    session_index: int
    position: int
    gap_seconds: float
    is_new_session: bool


def segment_sessions(
    timestamps_days: Iterable[float], *, gap_minutes: float = 30.0
) -> list[SessionBoundary]:
    """Segment sorted interactions without inspecting any outcomes."""

    threshold = gap_minutes * 60.0
    result: list[SessionBoundary] = []
    previous: float | None = None
    session_index = -1
    position = 0
    for value in timestamps_days:
        current = float(value)
        gap = 0.0 if previous is None else max(0.0, (current - previous) * SECONDS_PER_DAY)
        new = previous is None or gap > threshold
        if new:
            session_index += 1
            position = 0
        result.append(SessionBoundary(session_index, position, gap, new))
        position += 1
        previous = current
    return result


def session_id_from_instance(instance_id: str) -> str:
    """Return the official eight-character session prefix."""

    if len(instance_id) != 12:
        raise ValueError(f"Unexpected SLAM instance id: {instance_id!r}")
    return instance_id[:8]


def exercise_id_from_instance(instance_id: str) -> str:
    """Return the official session plus exercise prefix."""

    if len(instance_id) != 12:
        raise ValueError(f"Unexpected SLAM instance id: {instance_id!r}")
    return instance_id[:10]
