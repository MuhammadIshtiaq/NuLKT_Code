"""Streaming, leakage-safe preprocessing for the Duolingo SLAM corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

try:
    from .download import PREPROCESSING_VERSION, project_root
    from .manifest import manifest_parts
    from .splits import cold_start_learner_split
except ImportError:  # direct script execution
    from download import PREPROCESSING_VERSION, project_root
    from manifest import manifest_parts
    from splits import cold_start_learner_split


LOGGER = logging.getLogger(__name__)
CATEGORICAL_COLUMNS = (
    "user_id",
    "token",
    "part_of_speech",
    "morphology",
    "dependency_label",
    "client",
    "session_type",
    "exercise_format",
)


@dataclass
class LearnerHistory:
    interactions: int = 0
    errors: int = 0
    exercises: int = 0
    last_timestamp_days: float | None = None
    token_exposures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_labels: deque[int] = field(default_factory=lambda: deque(maxlen=20))


def discover_track_files(root: Path) -> dict[str, Path]:
    """Find only real corpus files, excluding macOS AppleDouble artifacts."""

    directory = root / "data" / "raw" / "extracted" / "data_en_es"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"English-learning track is not extracted at {directory}; run download first"
        )
    result: dict[str, Path] = {}
    for split in ("train", "dev", "test"):
        matches = sorted(
            path
            for path in directory.glob(f"*.{split}")
            if not path.name.startswith("._") and path.is_file()
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {split} corpus file, found {matches}")
        result[split] = matches[0]
        key = Path(str(matches[0]) + ".key")
        if split != "train" and not key.is_file():
            raise FileNotFoundError(f"Official {split} labels are missing: {key}")
    return result


def load_key(path: Path) -> dict[str, int]:
    """Load an official key file with duplicate and domain validation."""

    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2 or parts[1] not in {"0", "1"}:
                raise ValueError(f"Malformed key line {number} in {path}")
            if parts[0] in labels:
                raise ValueError(f"Duplicate key instance {parts[0]}")
            labels[parts[0]] = int(parts[1])
    return labels


def _parse_header(line: str, metadata: dict[str, Any]) -> None:
    if line.startswith("# prompt:"):
        metadata["prompt"] = line.partition(":")[2].strip()
        return
    for item in line[2:].split():
        key, separator, value = item.partition(":")
        if not separator:
            raise ValueError(f"Malformed exercise header item: {item!r}")
        metadata[key] = value


def iter_exercises(
    path: Path,
    *,
    official_split_name: str,
    external_labels: dict[str, int] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield complete response exercises so same-response labels cannot leak."""

    metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    seen_external: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                if rows:
                    yield rows
                metadata = {}
                rows = []
                continue
            if line.startswith("#"):
                _parse_header(line, metadata)
                continue
            parts = line.split()
            expected = 7 if official_split_name == "train" else 6
            if len(parts) != expected:
                raise ValueError(
                    f"Expected {expected} token fields at {path}:{line_number}, got {len(parts)}"
                )
            instance_id = parts[0]
            if len(instance_id) != 12:
                raise ValueError(f"Invalid instance ID at {path}:{line_number}: {instance_id}")
            if "user" not in metadata or "days" not in metadata:
                raise ValueError(f"Token precedes a complete exercise header at {path}:{line_number}")
            if official_split_name == "train":
                label = int(parts[6])
            else:
                if external_labels is None or instance_id not in external_labels:
                    raise ValueError(f"Missing official label for {instance_id}")
                label = external_labels[instance_id]
                seen_external.add(instance_id)
            rows.append(
                {
                    "instance_id": instance_id,
                    "exercise_id": instance_id[:10],
                    "session_id": instance_id[:8],
                    "exercise_index": int(instance_id[8:10]),
                    "token_index": int(instance_id[10:12]),
                    "token": parts[1],
                    "part_of_speech": parts[2],
                    "morphology": parts[3],
                    "dependency_label": parts[4],
                    "dependency_head": int(parts[5]),
                    "label": label,
                    "user_id": metadata["user"],
                    "countries": metadata.get("countries", ""),
                    "timestamp_days": float(metadata["days"]),
                    "client": metadata.get("client", "unknown"),
                    "session_type": metadata.get("session", "unknown"),
                    "exercise_format": metadata.get("format", "unknown"),
                    "response_time_seconds": (
                        math.nan if metadata.get("time") in {None, "null"} else int(metadata["time"])
                    ),
                    "prompt": metadata.get("prompt", ""),
                    "official_split": official_split_name,
                }
            )
    if rows:
        yield rows
    if external_labels is not None and seen_external != set(external_labels):
        missing = len(set(external_labels).difference(seen_external))
        raise ValueError(f"Corpus/key mismatch for {path}: {missing} key entries unused")


def add_past_only_features(
    exercise: list[dict[str, Any]],
    histories: dict[str, LearnerHistory],
    *,
    row_order_start: int,
) -> list[dict[str, Any]]:
    """Add features using a snapshot strictly before the submitted exercise."""

    if not exercise:
        return exercise
    user = str(exercise[0]["user_id"])
    history = histories.setdefault(user, LearnerHistory())
    timestamp = float(exercise[0]["timestamp_days"])
    elapsed = (
        0.0
        if history.last_timestamp_days is None
        else max(0.0, (timestamp - history.last_timestamp_days) * 86_400.0)
    )
    recent_error_rate = (
        sum(history.recent_labels) / len(history.recent_labels)
        if history.recent_labels
        else 0.0
    )
    for offset, row in enumerate(exercise):
        token = str(row["token"])
        prior_token_count = history.token_exposures[token]
        row.update(
            {
                "row_order": row_order_start + offset,
                "prior_interactions": history.interactions,
                "prior_exercises": history.exercises,
                "prior_errors": history.errors,
                "prior_error_rate": (
                    history.errors / history.interactions if history.interactions else 0.0
                ),
                "recent_20_error_rate": recent_error_rate,
                "prior_token_exposures": prior_token_count,
                "prior_token_errors": history.token_errors[token],
                "prior_token_error_rate": (
                    history.token_errors[token] / prior_token_count if prior_token_count else 0.0
                ),
                "elapsed_since_prior_exercise_seconds": elapsed,
                "log1p_elapsed_seconds": math.log1p(elapsed),
                "session_position": max(0, int(row["exercise_index"]) - 1),
            }
        )
    # Update once the full response has been featurized; same-exercise outcomes are unavailable above.
    for row in exercise:
        label = int(row["label"])
        token = str(row["token"])
        history.interactions += 1
        history.errors += label
        history.token_exposures[token] += 1
        history.token_errors[token] += label
        history.recent_labels.append(label)
    history.exercises += 1
    history.last_timestamp_days = timestamp
    return exercise


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows)
    integer_columns = {
        "exercise_index": "uint16",
        "token_index": "uint16",
        "dependency_head": "uint16",
        "label": "uint8",
        "row_order": "uint32",
        "prior_interactions": "uint32",
        "prior_exercises": "uint32",
        "prior_errors": "uint32",
        "prior_token_exposures": "uint16",
        "prior_token_errors": "uint16",
        "session_position": "uint16",
    }
    for column, dtype in integer_columns.items():
        frame[column] = frame[column].astype(dtype)
    float_columns = (
        "timestamp_days",
        "response_time_seconds",
        "prior_error_rate",
        "recent_20_error_rate",
        "prior_token_error_rate",
        "elapsed_since_prior_exercise_seconds",
        "log1p_elapsed_seconds",
    )
    for column in float_columns:
        frame[column] = frame[column].astype("float32")
    return frame


def _input_hash(paths: list[Path], parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_vocabularies(
    part_paths: list[Path], *, min_frequency: int = 2
) -> dict[str, dict[str, int]]:
    """Fit categorical mappings exclusively on official training rows."""

    counters = {column: Counter() for column in CATEGORICAL_COLUMNS}
    for path in part_paths:
        frame = pd.read_parquet(path, columns=[*CATEGORICAL_COLUMNS, "official_split"])
        training = frame[frame["official_split"] == "train"]
        for column in CATEGORICAL_COLUMNS:
            counters[column].update(training[column].fillna("unknown").astype(str))
    vocabularies: dict[str, dict[str, int]] = {}
    for column, counter in counters.items():
        ordered = sorted(
            (value for value, count in counter.items() if count >= min_frequency),
            key=lambda value: (-counter[value], value),
        )
        vocabularies[column] = {"<PAD>": 0, "<UNK>": 1} | {
            value: index + 2 for index, value in enumerate(ordered)
        }
    return vocabularies


def _apply_encodings_and_splits(
    part_paths: list[Path], vocabularies: dict[str, dict[str, int]]
) -> None:
    # Count whole exercises without materializing millions of string-rich rows.
    exercise_totals: Counter[str] = Counter()
    users: set[str] = set()
    for path in part_paths:
        columns = pd.read_parquet(path, columns=["user_id", "exercise_id"])
        exercises = columns.drop_duplicates(["user_id", "exercise_id"], keep="first")
        exercise_totals.update(exercises["user_id"].astype(str))
        users.update(exercises["user_id"].astype(str))
    user_frame = pd.DataFrame(
        {
            "user_id": sorted(users),
            "exercise_id": [f"cold-{index}" for index in range(len(users))],
            "timestamp_days": 0.0,
            "row_order": range(len(users)),
        }
    )
    cold_mapping = dict(
        zip(
            user_frame["user_id"],
            cold_start_learner_split(user_frame).astype(str),
        )
    )
    seen_exercises: Counter[str] = Counter()
    for path in part_paths:
        frame = pd.read_parquet(path)
        if frame["instance_id"].duplicated().any():
            raise RuntimeError(f"Duplicate instance IDs within {path}")
        exercises = frame[["user_id", "exercise_id"]].drop_duplicates(
            ["user_id", "exercise_id"], keep="first"
        )
        exercises["position"] = (
            exercises.groupby("user_id", sort=False).cumcount()
            + exercises["user_id"].astype(str).map(seen_exercises).fillna(0).astype(int)
            + 1
        )
        exercises["total"] = exercises["user_id"].astype(str).map(exercise_totals).astype(int)
        fraction = exercises["position"] / exercises["total"]
        exercises["assigned"] = "test"
        exercises.loc[fraction <= 0.70, "assigned"] = "train"
        exercises.loc[(fraction > 0.70) & (fraction <= 0.85), "assigned"] = "validation"
        exercises.loc[exercises["total"] < 3, "assigned"] = "train"
        lookup = exercises.set_index(["user_id", "exercise_id"])["assigned"]
        keys = pd.MultiIndex.from_frame(frame[["user_id", "exercise_id"]])
        frame["chronological_split"] = lookup.reindex(keys).to_numpy()
        frame["cold_start_split"] = frame["user_id"].astype(str).map(cold_mapping)
        if frame[["chronological_split", "cold_start_split"]].isna().any().any():
            raise RuntimeError(f"Split lookup failed for {path}")
        seen_exercises.update(exercises["user_id"].astype(str))
        for column, vocabulary in vocabularies.items():
            frame[f"{column}_id"] = (
                frame[column].fillna("unknown").astype(str).map(vocabulary).fillna(1).astype("uint32")
            )
        frame.to_parquet(path, index=False, compression="zstd")


def process_corpus(
    *,
    root: Path | None = None,
    chunk_size: int = 100_000,
    min_frequency: int = 2,
    max_exercises: int | None = None,
    force: bool = False,
    unsafe_intermediate: bool = False,
) -> dict[str, Any]:
    """Build an intermediate parse only for immediate split-frozen rewriting.

    The generic parser updates hidden split histories and is invalid for model
    fitting until `build_offline_safe_data.py` replaces those fields. Its direct
    use is therefore refused by default in this corrected source copy.
    """

    if not unsafe_intermediate:
        raise RuntimeError(
            "Direct generic preprocessing is not safe for official fitting. "
            "Use scripts/reproduce_offline_safe.py --prepare."
        )

    if chunk_size <= 0 or min_frequency <= 0:
        raise ValueError("chunk_size and min_frequency must be positive")
    root = (root or project_root()).resolve()
    files = discover_track_files(root)
    key_paths = [Path(str(files[name]) + ".key") for name in ("dev", "test")]
    parameters = {
        "version": PREPROCESSING_VERSION,
        "chunk_size": chunk_size,
        "min_frequency": min_frequency,
        "max_exercises": max_exercises,
    }
    fingerprint = _input_hash([*files.values(), *key_paths], parameters)
    output_dir = root / "data" / "processed" / "en_es"
    manifest_path = output_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("preprocessing_hash") == fingerprint and all(
            path.is_file() for path in manifest_parts(root, manifest)
        ):
            LOGGER.info("Using cached processed corpus: %s", fingerprint)
            summary = root / "results" / "preprocessing_summary.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("part-*.parquet"):
        existing.unlink()

    histories: dict[str, LearnerHistory] = {}
    buffer: list[dict[str, Any]] = []
    part_paths: list[Path] = []
    split_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    row_order = 0
    exercises_seen = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        path = output_dir / f"part-{len(part_paths):05d}.parquet"
        _frame(buffer).to_parquet(path, index=False, compression="zstd")
        part_paths.append(path)
        buffer = []

    stop = False
    for split_name in ("train", "dev", "test"):
        labels = None if split_name == "train" else load_key(Path(str(files[split_name]) + ".key"))
        for exercise in iter_exercises(
            files[split_name], official_split_name=split_name, external_labels=labels
        ):
            if max_exercises is not None and exercises_seen >= max_exercises:
                stop = True
                break
            enriched = add_past_only_features(exercise, histories, row_order_start=row_order)
            row_order += len(enriched)
            exercises_seen += 1
            buffer.extend(enriched)
            split_counts[split_name] += len(enriched)
            label_counts.update(int(row["label"]) for row in enriched)
            if len(buffer) >= chunk_size:
                flush()
        if stop:
            break
    flush()
    if not part_paths:
        raise RuntimeError("No interactions were parsed")
    vocabularies = build_vocabularies(part_paths, min_frequency=min_frequency)
    (output_dir / "vocabularies.json").write_text(
        json.dumps(vocabularies, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _apply_encodings_and_splits(part_paths, vocabularies)
    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_hash": fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": "en_es",
        "track_meaning": "English learners who already speak Spanish (official Dataverse description)",
        "parts": [str(path.relative_to(root)) for path in part_paths],
        "interactions": sum(split_counts.values()),
        "exercises": exercises_seen,
        "learners": len(histories),
        "official_split_counts": dict(split_counts),
        "label_counts": {str(key): value for key, value in label_counts.items()},
        "categorical_encoding_fit_partition": "official train only",
        "categorical_pad_id": 0,
        "categorical_unknown_id": 1,
        "past_only_features": [
            "prior_interactions",
            "prior_exercises",
            "prior_errors",
            "prior_error_rate",
            "recent_20_error_rate",
            "prior_token_exposures",
            "prior_token_errors",
            "prior_token_error_rate",
            "elapsed_since_prior_exercise_seconds",
        ],
        "same_exercise_outcomes_excluded": True,
        "response_time_warning": (
            "response_time_seconds is preserved for compatible official-benchmark analyses but is "
            "post-response and must be excluded from pre-response recommendation models"
        ),
        "debug_subset": max_exercises is not None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_summary = root / "results" / "preprocessing_summary.json"
    results_summary.parent.mkdir(parents=True, exist_ok=True)
    results_summary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-exercises", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(
        json.dumps(
            process_corpus(
                root=args.root,
                chunk_size=args.chunk_size,
                min_frequency=args.min_frequency,
                max_exercises=args.max_exercises,
                force=args.force,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
