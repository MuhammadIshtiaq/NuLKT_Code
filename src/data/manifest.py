"""Platform-portable reading of the processed-corpus manifest.

The manifest records its partition paths as written by whichever machine ran
preprocessing. A manifest written on Windows stores ``data\\processed\\en_es\\
part-00000.parquet``; handed to ``Path`` on POSIX that string is a single
filename containing literal backslashes, not a path, so every partition appears
to be missing and the corpus silently fails to load.

Reading always goes through here so the corpus stays replayable on a different
platform than it was built on - which is exactly what happens when a run moves
to a Linux GPU host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def normalise_relative(relative: str | Path) -> Path:
    """Interpret a manifest-recorded path independently of its writing platform."""
    return Path(str(relative).replace("\\", "/"))


def manifest_parts(root: Path, manifest: dict[str, Any]) -> list[Path]:
    """Absolute paths to the corpus partitions named by ``manifest``."""
    return [root / normalise_relative(part) for part in manifest.get("parts", [])]


def missing_parts(root: Path, manifest: dict[str, Any]) -> list[Path]:
    """Partitions the manifest names that are not present on disk."""
    return [path for path in manifest_parts(root, manifest) if not path.is_file()]


def iter_parts(root: Path, parts: Iterable[str | Path]) -> list[Path]:
    """Absolute paths for an already-extracted ``parts`` sequence."""
    return [root / normalise_relative(part) for part in parts]
