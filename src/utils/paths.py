"""Single source of truth for project paths.

The default is the checkout root. Tests and CI may override it with
``NULKT_ROOT`` (or the legacy ``EIP_KTFORMER_ROOT``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def metadata(self) -> Path:
        return self.data / "metadata"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> "ProjectPaths":
        directories = [
            self.raw,
            self.interim,
            self.processed,
            self.metadata,
            self.checkpoints,
            self.logs,
            self.results / "protocol",
            self.results / "raw_runs",
            self.results / "metrics",
            self.figures,
            self.tables,
            self.results / "statistical_tests",
            self.results / "reproduction_bundle",
            self.results / "publication",
            self.root / "references",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | Path | None = None) -> ProjectPaths:
    configured = (
        root
        or os.environ.get("NULKT_ROOT")
        or os.environ.get("EIP_KTFORMER_ROOT")
        or DEFAULT_PROJECT_ROOT
    )
    return ProjectPaths(Path(configured).expanduser().resolve())


PATHS = get_paths()
