"""Protocol freezing and integrity hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol(protocol: Path) -> Path:
    """Write a sidecar SHA-256 without changing the frozen protocol."""
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    sidecar = protocol.with_suffix(protocol.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(protocol)}  {protocol.name}\n", encoding="ascii")
    return sidecar
