"""PyTorch sequence datasets over leakage-safe processed SLAM partitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .context_features import DERIVED_COLUMNS, add_context_features
from .manifest import manifest_parts
from .sequences import iter_context_windows, iter_sequence_windows


MODEL_FIELD_SOURCES = {
    "token": "token_id",
    # SLAM supplies surface tokens but no lemma column. The architecture retains a
    # separate input slot; for this corpus it is explicitly tied to surface token.
    "lemma": "token_id",
    "pos": "part_of_speech_id",
    "morph": "morphology_id",
    "context": "dependency_label_id",
    "exercise": "session_type_id",
    "format": "exercise_format_id",
    "learner": "user_id_id",
}


def load_processed_frame(
    root: Path,
    row_limit: int | None = None,
    *,
    derive_context: bool = True,
    split_filter: tuple[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / "data" / "processed" / "en_es"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    remaining = row_limit
    for part_path in manifest_parts(root, manifest):
        if split_filter is not None:
            split_column, allowed = split_filter
            # Push the partition predicate into the Parquet reader so protected
            # test rows (including labels) never materialize in model-selection
            # memory. A second in-memory check below is defense in depth.
            frame = pd.read_parquet(
                part_path,
                filters=[(split_column, "in", sorted(allowed))],
            )
            if split_column not in frame:
                raise KeyError(f"Unknown split protocol column: {split_column}")
            frame = frame[frame[split_column].astype(str).isin(allowed)]
        else:
            frame = pd.read_parquet(part_path)
        if remaining is not None:
            frame = frame.iloc[:remaining]
            remaining -= len(frame)
        frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    if not frames:
        raise RuntimeError("Processed manifest contains no readable partitions")
    combined = pd.concat(frames, ignore_index=True)
    if manifest.get("offline_safe_protocol"):
        missing_safe = sorted(set(DERIVED_COLUMNS).difference(combined.columns))
        if missing_safe:
            raise RuntimeError(
                f"Offline-safe processed data lack frozen context columns: {missing_safe}"
            )
    if derive_context:
        combined = add_context_features(combined)
    return combined, manifest


def vocab_sizes_for_model(root: Path, prompt_size: int | None = None) -> dict[str, int]:
    path = root / "data" / "processed" / "en_es" / "vocabularies.json"
    vocabularies = json.loads(path.read_text(encoding="utf-8"))
    source_vocab = {
        "token": "token",
        "lemma": "token",
        "pos": "part_of_speech",
        "morph": "morphology",
        "context": "dependency_label",
        "exercise": "session_type",
        "format": "exercise_format",
        "learner": "user_id",
    }
    sizes = {field: len(vocabularies[source]) for field, source in source_vocab.items()}
    # Preprocessing predates the prompt embedding and does not emit a prompt
    # vocabulary, so derive one and cache it beside the others. Index 0 stays
    # reserved for padding and for a prompt unseen at fit time.
    sizes["prompt"] = int(prompt_size) if prompt_size is not None else len(prompt_vocabulary(root)) + 1
    return sizes


def prompt_vocabulary(
    root: Path,
    *,
    frame: pd.DataFrame | None = None,
    protocol: str = "chronological_split",
) -> dict[str, int]:
    """Stable prompt-to-index map, built once and cached beside the corpus.

    Codes must not be derived per split: factorising within a partition would
    give the same prompt different indices in train and test. Sorting the
    distinct prompts makes the mapping deterministic and independent of row
    order, so the same corpus always yields the same vocabulary.
    """
    if frame is not None:
        if protocol not in frame:
            raise KeyError(f"Unknown split protocol column: {protocol}")
        training = frame[frame[protocol].astype(str) == "train"]
        if training.empty:
            raise ValueError(f"Protocol {protocol!r} contains no training rows")
        prompts = sorted(training["prompt"].dropna().astype(str).unique())
        return {prompt: index for index, prompt in enumerate(prompts, start=1)}

    directory = root / "data" / "processed" / "en_es"
    cache = directory / "prompt_vocabulary.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    prompts: set[str] = set()
    for part_path in manifest_parts(root, manifest):
        prompts.update(pd.read_parquet(part_path, columns=["prompt"])["prompt"].dropna().astype(str))
    vocabulary = {prompt: index for index, prompt in enumerate(sorted(prompts), start=1)}
    cache.write_text(json.dumps(vocabulary, indent=0), encoding="utf-8")
    return vocabulary


class EncodedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length causal windows constructed without consulting target values."""

    def __init__(
        self,
        frame: pd.DataFrame,
        max_length: int = 200,
        context_length: int = 0,
        feature_regime: str = "pre_response",
    ) -> None:
        required = {
            *MODEL_FIELD_SOURCES.values(),
            "user_id",
            "timestamp_days",
            "row_order",
            "session_id",
            "label",
            "prior_interactions",
            "prior_error_rate",
            "recent_20_error_rate",
            "prior_token_exposures",
            "prior_token_error_rate",
            "elapsed_since_prior_exercise_seconds",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Processed data are missing model columns: {sorted(missing)}")
        self.frame = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable")
        # Official dev/test labels are available for metrics only. They are
        # never observable responses within a blind evaluation partition.
        if "official_split" not in self.frame:
            raise RuntimeError("Offline-safe sequence loader requires official_split")
        split_values = set(self.frame["official_split"].astype(str).unique())
        if len(split_values) != 1 or not split_values <= {"train", "dev", "test"}:
            raise RuntimeError(f"Expected one official split per dataset, got {split_values}")
        self.observed_response_available = split_values == {"train"}
        self.max_length = max_length
        self.context_length = context_length
        if feature_regime not in {"pre_response", "benchmark_post_response"}:
            raise ValueError(
                "feature_regime must be 'pre_response' or 'benchmark_post_response'"
            )
        self.feature_regime = feature_regime
        # Carrying attention context across window boundaries keeps predictions
        # from restarting mid-trajectory; see iter_context_windows.
        self.windows = list(
            iter_context_windows(self.frame, max_length=max_length, context_length=context_length)
            if context_length
            else iter_sequence_windows(self.frame, max_length=max_length)
        )
        if not self.windows:
            raise ValueError("No learner sequences could be constructed")

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _pad(values: np.ndarray, length: int, dtype: torch.dtype) -> torch.Tensor:
        result = torch.zeros(length, dtype=dtype)
        count = min(len(values), length)
        if count:
            result[:count] = torch.as_tensor(np.array(values[:count], copy=True), dtype=dtype)
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        rows = self.frame.loc[window.row_indices]
        count = len(rows)
        batch: dict[str, torch.Tensor] = {
            field: self._pad(rows[source].to_numpy(dtype=np.int64), self.max_length, torch.long)
            for field, source in MODEL_FIELD_SOURCES.items()
        }
        batch["responses"] = self._pad(rows["label"].to_numpy(dtype=np.int64), self.max_length, torch.long)
        # -1 encodes a withheld response; after shifting the model maps it to
        # embedding index 0. The true `responses` tensor stays as the target.
        observed = (
            rows["label"].to_numpy(dtype=np.int64)
            if self.observed_response_available
            else np.full(count, -1, dtype=np.int64)
        )
        batch["observed_responses"] = self._pad(observed, self.max_length, torch.long)
        batch["row_index"] = self._pad(
            rows["row_order"].to_numpy(dtype=np.int64), self.max_length, torch.long
        )
        batch["timestamps"] = self._pad(
            rows["timestamp_days"].to_numpy(dtype=np.float64) * 86_400.0,
            self.max_length,
            torch.float32,
        )
        batch["elapsed_seconds"] = self._pad(
            rows["elapsed_since_prior_exercise_seconds"].to_numpy(dtype=np.float32),
            self.max_length,
            torch.float32,
        )
        session_codes, _ = pd.factorize(rows["session_id"], sort=False)
        batch["session"] = self._pad(session_codes.astype(np.int64) + 1, self.max_length, torch.long)
        # Exercise grouping drives within-exercise bidirectional attention, which
        # is legitimate because an exercise's surface form is given at prediction
        # time while its labels are not.
        grouping = "exercise_id" if "exercise_id" in rows.columns else "session_id"
        exercise_codes, _ = pd.factorize(rows[grouping], sort=False)
        batch["exercise_group"] = self._pad(exercise_codes.astype(np.int64) + 1, self.max_length, torch.long)
        history = np.column_stack(
            [
                np.log1p(rows["prior_interactions"].to_numpy(dtype=np.float32)),
                rows["prior_error_rate"].to_numpy(dtype=np.float32),
                rows["recent_20_error_rate"].to_numpy(dtype=np.float32),
                np.log1p(rows["prior_token_exposures"].to_numpy(dtype=np.float32)),
                rows["prior_token_error_rate"].to_numpy(dtype=np.float32),
                np.log1p(rows["elapsed_since_prior_exercise_seconds"].to_numpy(dtype=np.float32)),
            ]
        )
        padded_history = torch.zeros(self.max_length, history.shape[1], dtype=torch.float32)
        padded_history[:count] = torch.from_numpy(history)
        batch["history_features"] = padded_history
        def _float_column(name: str) -> np.ndarray:
            if name in rows:
                return rows[name].to_numpy(dtype=np.float32)
            return np.zeros(count, dtype=np.float32)

        exercise_state = np.column_stack(
            [
                _float_column("previous_exercise_error_rate"),
                np.log1p(_float_column("previous_exercise_length")),
                _float_column("exercise_error_ema_short"),
                _float_column("exercise_error_ema_long"),
            ]
        ).astype(np.float32)
        padded_exercise_state = torch.zeros(self.max_length, 4, dtype=torch.float32)
        padded_exercise_state[:count] = torch.from_numpy(exercise_state)
        batch["completed_exercise_state"] = padded_exercise_state
        # Exercise context: prompt identity plus the exercise-level scalars that
        # are handed over with the exercise itself. Absent columns degrade to
        # zeros so a corpus preprocessed before these fields still loads.
        if "prompt_code" in rows.columns:
            batch["prompt"] = self._pad(rows["prompt_code"].to_numpy(dtype=np.int64), self.max_length, torch.long)
        else:
            batch["prompt"] = torch.zeros(self.max_length, dtype=torch.long)

        def _column(name: str) -> np.ndarray:
            return _float_column(name)

        safe_scalars = [
            _column("session_position"),
            _column("token_index"),
            # Token-level recency is strictly past-only. Its flag distinguishes a
            # first encounter from a genuinely short revisit interval.
            np.log1p(_column("hours_since_token_last_seen")),
            _column("token_last_seen_observed"),
        ]
        if self.feature_regime == "benchmark_post_response":
            latency = (
                rows["response_time_seconds"].to_numpy(dtype=np.float32)
                if "response_time_seconds" in rows
                else np.full(count, np.nan, dtype=np.float32)
            )
            observed = np.isfinite(latency)
            clipped = np.clip(np.nan_to_num(latency, nan=0.0), 0.0, 600.0)
            safe_scalars = [np.log1p(clipped), observed.astype(np.float32), *safe_scalars]
        scalars = np.column_stack(safe_scalars).astype(np.float32)
        neighbours = np.column_stack(
            [_column("preceding_token_code"), _column("following_token_code")]
        ).astype(np.int64)
        padded_neighbours = torch.zeros(self.max_length, 2, dtype=torch.long)
        padded_neighbours[:count] = torch.from_numpy(neighbours)
        batch["neighbour_tokens"] = padded_neighbours
        padded_scalars = torch.zeros(self.max_length, scalars.shape[1], dtype=torch.float32)
        padded_scalars[:count] = torch.from_numpy(scalars)
        batch["exercise_scalars"] = padded_scalars
        padding = torch.ones(self.max_length, dtype=torch.bool)
        padding[:count] = False
        batch["padding_mask"] = padding
        # Positions carried only as attention context are attended to but never
        # scored, so each interaction contributes to loss and metrics once.
        score = torch.zeros(self.max_length, dtype=torch.bool)
        scored = window.scored()
        score[: len(scored)] = torch.as_tensor(np.asarray(scored, dtype=bool))
        batch["score_mask"] = score & ~padding
        return batch


def prepare_dataloaders(
    root: Path,
    *,
    protocol: str = "chronological_split",
    max_length: int = 200,
    batch_size: int = 128,
    num_workers: int = 0,
    row_limit: int | None = None,
    seed: int = 42,
    context_length: int = 0,
    feature_regime: str = "pre_response",
    include_test: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Create train/validation/test loaders while keeping test outside training APIs."""

    frame, _ = load_processed_frame(
        root,
        row_limit=row_limit,
        derive_context=include_test,
        split_filter=None if include_test else (protocol, {"train", "validation"}),
    )
    if protocol not in frame.columns:
        raise KeyError(f"Unknown split protocol column: {protocol}")
    if not include_test:
        # Drop the protected partition before any loader-specific derived feature
        # construction. The general processed-frame reader remains available to
        # retrospective analyses, but model selection receives no test rows.
        frame = frame[frame[protocol].astype(str).isin(["train", "validation"])].copy()
        frame = add_context_features(frame)
    if "prompt" in frame:
        # Fit only on the declared training partition. Validation/test prompts
        # unseen during fitting map to the reserved unknown index 0.
        vocabulary = prompt_vocabulary(root, frame=frame, protocol=protocol)
        frame["prompt_code"] = (
            frame["prompt"].astype(str).map(vocabulary).fillna(0).astype("int64")
        )
    loaders: dict[str, DataLoader] = {}
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    for split in splits:
        subset = frame[frame[protocol].astype(str) == split].copy()
        if subset.empty:
            continue
        dataset = EncodedSequenceDataset(
            subset,
            max_length=max_length,
            context_length=context_length,
            feature_regime=feature_regime,
        )
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator,
        )
    vocab_sizes = vocab_sizes_for_model(root, prompt_size=len(vocabulary) + 1 if "prompt" in frame else 1)
    if "prompt" in frame:
        vocab_sizes["prompt"] = len(vocabulary) + 1
    return loaders, vocab_sizes
