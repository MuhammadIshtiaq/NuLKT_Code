"""Rebuild official processed features without blind-partition label histories.

This script writes only to the isolated `work/offline_safe` checkout.  Labels in
dev/test remain in Parquet solely as scoring targets.  For each learner, a
target exercise receives state from strictly earlier timestamps: train rows
read earlier train outcomes; dev reads earlier train outcomes; test reads
earlier train and dev outcomes.  No state is updated from test outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "en_es"
PARTS = sorted(DATA.glob("part-*.parquet"))
READ_COLUMNS = [
    "user_id", "exercise_id", "timestamp_days", "row_order", "official_split",
    "token", "token_id", "token_index", "prompt", "label",
]
SAFE_COLUMNS = {
    "prior_interactions": np.uint32,
    "prior_exercises": np.uint32,
    "prior_errors": np.uint32,
    "prior_error_rate": np.float32,
    "recent_20_error_rate": np.float32,
    "prior_token_exposures": np.uint32,
    "prior_token_errors": np.uint32,
    "prior_token_error_rate": np.float32,
    "elapsed_since_prior_exercise_seconds": np.float32,
    "log1p_elapsed_seconds": np.float32,
    "previous_exercise_error_rate": np.float32,
    "previous_exercise_length": np.float32,
    "exercise_error_ema_short": np.float32,
    "exercise_error_ema_long": np.float32,
    "hours_since_token_last_seen": np.float32,
    "token_last_seen_observed": np.float32,
    "preceding_token_code": np.int32,
    "following_token_code": np.int32,
    "preceding_token_id": np.int32,
    "following_token_id": np.int32,
    "sentence_length": np.int32,
    "token_relative_position": np.float32,
    "prompt_id": np.int32,
    "prompt_code": np.int32,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class State:
    interactions: int = 0
    exercises: int = 0
    errors: int = 0
    last_time: float | None = None
    last_ex_rate: float = 0.0
    last_ex_len: int = 0
    ema_short: float | None = None
    ema_long: float | None = None
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=20))
    token_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    token_last_time: dict[str, float] = field(default_factory=dict)

    def update(self, tokens: list[str], labels: np.ndarray, time: float) -> None:
        self.interactions += len(tokens)
        self.exercises += 1
        self.errors += int(labels.sum())
        self.last_time = time
        self.last_ex_len = len(tokens)
        self.last_ex_rate = float(labels.mean())
        self.ema_short = (self.last_ex_rate if self.ema_short is None
                          else 0.5 * self.last_ex_rate + 0.5 * self.ema_short)
        self.ema_long = (self.last_ex_rate if self.ema_long is None
                         else 0.125 * self.last_ex_rate + 0.875 * self.ema_long)
        for token, label in zip(tokens, labels, strict=True):
            self.token_count[token] += 1
            self.token_errors[token] += int(label)
            self.token_last_time[token] = time
            self.recent.append(int(label))


def safe_features(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    n = len(frame)
    row_order = frame.row_order.to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(row_order), np.arange(n)):
        raise RuntimeError("row_order must uniquely cover 0..n-1")
    user, _ = pd.factorize(frame.user_id, sort=True)
    exercise, _ = pd.factorize(frame.exercise_id, sort=False)
    time = frame.timestamp_days.to_numpy(dtype=np.float64)
    order = np.lexsort((row_order, time, user))
    user = user[order]
    exercise = exercise[order]
    time = time[order]
    rows = row_order[order]
    splits = frame.official_split.astype(str).to_numpy()[order]
    tokens = frame.token.astype(str).to_numpy()[order]
    token_codes = frame.token_id.to_numpy(dtype=np.int32)[order]
    token_index = frame.token_index.to_numpy(dtype=np.int32)[order]
    labels = frame.label.to_numpy(dtype=np.int8)[order]
    if not set(splits) <= {"train", "dev", "test"}:
        raise RuntimeError(f"Unknown official split: {set(splits)}")
    result = {name: np.zeros(n, dtype=dtype) for name, dtype in SAFE_COLUMNS.items()}

    # Exercise contiguity is required to freeze an entire response before its
    # labels can change any state. Full official corpus has 1,054,368 exercises.
    new_ex = np.r_[True, (user[1:] != user[:-1]) | (exercise[1:] != exercise[:-1])]
    starts = np.flatnonzero(new_ex)
    ends = np.r_[starts[1:], n]
    if len(starts) != len(pd.MultiIndex.from_arrays([user, exercise]).unique()):
        raise RuntimeError("One exercise appears in nonconsecutive time-order blocks")
    if any(np.unique(splits[a:b]).size != 1 for a, b in zip(starts, ends, strict=True)):
        raise RuntimeError("An exercise crosses official split boundaries")

    train_state = State()
    test_state = State()
    current_user = -1
    i = 0
    while i < len(starts):
        a = starts[i]
        if user[a] != current_user:
            current_user = int(user[a])
            train_state, test_state = State(), State()
        stamp = time[a]
        j = i + 1
        while j < len(starts) and user[starts[j]] == current_user and time[starts[j]] == stamp:
            j += 1
        pending: list[tuple[str, list[str], np.ndarray]] = []
        for k in range(i, j):
            a, b = int(starts[k]), int(ends[k])
            split = splits[a]
            state = train_state if split in {"train", "dev"} else test_state
            idx = rows[a:b]
            word = tokens[a:b]
            y = labels[a:b]
            if not np.all(time[a:b] == stamp):
                raise RuntimeError("Exercise tokens have differing timestamps")
            count = state.interactions
            result["prior_interactions"][idx] = count
            result["prior_exercises"][idx] = state.exercises
            result["prior_errors"][idx] = state.errors
            result["prior_error_rate"][idx] = state.errors / count if count else 0.0
            result["recent_20_error_rate"][idx] = (
                sum(state.recent) / len(state.recent) if state.recent else 0.0
            )
            elapsed = 0.0 if state.last_time is None else max(0.0, (stamp - state.last_time) * 86400.0)
            result["elapsed_since_prior_exercise_seconds"][idx] = elapsed
            result["log1p_elapsed_seconds"][idx] = math.log1p(elapsed)
            result["previous_exercise_error_rate"][idx] = state.last_ex_rate
            result["previous_exercise_length"][idx] = state.last_ex_len
            result["exercise_error_ema_short"][idx] = state.ema_short or 0.0
            result["exercise_error_ema_long"][idx] = state.ema_long or 0.0
            for m, token in enumerate(word):
                r = idx[m]
                exposures = state.token_count[token]
                result["prior_token_exposures"][r] = exposures
                result["prior_token_errors"][r] = state.token_errors[token]
                result["prior_token_error_rate"][r] = (
                    state.token_errors[token] / exposures if exposures else 0.0
                )
                last = state.token_last_time.get(token)
                if last is not None:
                    result["hours_since_token_last_seen"][r] = min(max((stamp - last) * 24.0, 0.0), 8760.0)
                    result["token_last_seen_observed"][r] = 1.0
            length = b - a
            result["sentence_length"][idx] = length
            result["token_relative_position"][idx] = token_index[a:b] / max(length, 1)
            if length > 1:
                result["preceding_token_code"][idx[1:]] = token_codes[a:b-1]
                result["following_token_code"][idx[:-1]] = token_codes[a+1:b]
                result["preceding_token_id"][idx[1:]] = token_codes[a:b-1]
                result["following_token_id"][idx[:-1]] = token_codes[a+1:b]
            if split in {"train", "dev"}:
                pending.append((split, list(word), y.copy()))
        # No exercise at this timestamp can see a contemporaneous outcome.
        for split, word, y in pending:
            if split == "train":
                train_state.update(word, y, stamp)
                test_state.update(word, y, stamp)
            else:
                test_state.update(word, y, stamp)
        i = j
    return result, {"n_rows": n, "n_exercises": len(starts),
                    "n_learners": int(np.unique(user).size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if not args.build:
        ap.error("Pass --build to rewrite the isolated copied processed data")
    if not PARTS:
        raise RuntimeError("Prepare local Parquet data before building split-frozen features")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_manifest = manifest.copy()
    frames = [pd.read_parquet(part, columns=READ_COLUMNS) for part in PARTS]
    minimal = pd.concat(frames, ignore_index=True)
    if len(minimal) != int(manifest["interactions"]):
        raise RuntimeError("Manifest interaction count mismatch")
    features, counts = safe_features(minimal)
    train_prompts = sorted(minimal.loc[minimal.official_split == "train", "prompt"].dropna().astype(str).unique())
    prompt_registry = {p: k + 1 for k, p in enumerate(train_prompts)}
    features["prompt_code"][minimal.row_order.to_numpy(dtype=np.int64)] = (
        minimal.prompt.astype(str).map(prompt_registry).fillna(0).to_numpy(dtype=np.int32)
    )
    features["prompt_id"][:] = features["prompt_code"]
    (DATA / "prompt_vocabulary.json").write_text(
        json.dumps(prompt_registry, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    cursor = 0
    new_hashes = {}
    for part, small in zip(PARTS, frames, strict=True):
        full = pd.read_parquet(part)
        if len(full) != len(small) or not full.row_order.equals(small.row_order):
            raise RuntimeError(f"Part order mismatch: {part}")
        idx = full.row_order.to_numpy(dtype=np.int64)
        for name, values in features.items():
            full[name] = values[idx]
        temp = part.with_suffix(".safe.tmp.parquet")
        full.to_parquet(temp, index=False, compression="zstd")
        temp.replace(part)
        new_hashes[part.name] = sha(part)
        cursor += len(full)
    if cursor != len(minimal):
        raise RuntimeError("Part rewrite count mismatch")
    manifest.update({
        "offline_safe_protocol": "split-frozen-v1",
        "source_preprocessing_hash": original_manifest["preprocessing_hash"],
        "preprocessing_hash": hashlib.sha256(
            (original_manifest["preprocessing_hash"] + sha(Path(__file__)) + sha(DATA / "prompt_vocabulary.json")).encode()
        ).hexdigest(),
        "offline_safe_created_utc": datetime.now(timezone.utc).isoformat(),
        "offline_safe_features": list(SAFE_COLUMNS),
        "offline_safe_row_counts": counts,
        "offline_safe_prompt_registry_size": len(prompt_registry),
        "offline_safe_prompt_registry_sha256": sha(DATA / "prompt_vocabulary.json"),
        "offline_safe_part_sha256": new_hashes,
        "offline_safe_semantics": (
            "train targets see earlier TRAIN outcomes; DEV sees earlier TRAIN outcomes; "
            "TEST sees earlier TRAIN+DEV outcomes; no same-timestamp or TEST outcomes; "
            "all prior label-derived context and token recency frozen by this rule"
        ),
        "offline_safe_response_policy": (
            "TRAIN model inputs observe earlier train outcomes; DEV/TEST model inputs "
            "use -1 unknown response at every valid held-out position, while true labels "
            "remain targets for validation and reporting"
        ),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "built", "counts": counts,
                      "prompt_registry_size": len(prompt_registry),
                      "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
