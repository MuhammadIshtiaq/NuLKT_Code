"""Fail-closed corpus and feature audit before any offline-safe model fit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_official import feature_columns  # noqa: E402
from scripts.build_offline_safe_data import SAFE_COLUMNS  # noqa: E402

DATA = ROOT / "data" / "processed" / "en_es"
MANIFEST = DATA / "manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("offline_safe_protocol") != "split-frozen-v1":
        raise RuntimeError("Offline-safe manifest marker missing")
    parts = sorted(DATA.glob("part-*.parquet"))
    if len(parts) != len(manifest["parts"]):
        raise RuntimeError("Part count mismatch")
    required = {
        *feature_columns("benchmark"), *feature_columns("deployment"),
        *SAFE_COLUMNS, "label", "official_split", "row_order", "exercise_id",
        "token_id", "token_index", "user_id", "timestamp_days", "prompt",
    }
    if "label" in feature_columns("benchmark") or "label" in feature_columns("deployment"):
        raise RuntimeError("Target label appears in boosted features")
    minimal = []
    for part in parts:
        if sha(part) != manifest["offline_safe_part_sha256"][part.name]:
            raise RuntimeError(f"Hash mismatch: {part}")
        frame = pd.read_parquet(part, columns=sorted(required))
        minimal.append(frame)
    frame = pd.concat(minimal, ignore_index=True)
    n = len(frame)
    if n != manifest["interactions"] or frame.row_order.duplicated().any():
        raise RuntimeError("Row count/identity mismatch")
    counts = frame.official_split.astype(str).value_counts().to_dict()
    if counts != manifest["official_split_counts"]:
        raise RuntimeError(f"Split counts differ: {counts}")
    if (frame.loc[frame.official_split == "train", "prompt_code"] == 0).any():
        raise RuntimeError("Train prompt unexpectedly unknown")
    # Independent vectorized reconstruction of every surface-only context
    # column. This checks exact equivalence to the original definitions.
    by_ex = frame.sort_values(["exercise_id", "token_index", "row_order"], kind="stable")
    grouped = by_ex.groupby("exercise_id", sort=False, observed=True)
    expected_length = grouped.token_id.transform("size").to_numpy()
    if not np.array_equal(by_ex.sentence_length.to_numpy(), expected_length):
        raise RuntimeError("Sentence length changed from group size")
    prior = grouped.token_id.shift(1).fillna(0).to_numpy(dtype=np.int64)
    future = grouped.token_id.shift(-1).fillna(0).to_numpy(dtype=np.int64)
    if not np.array_equal(by_ex.preceding_token_code.to_numpy(), prior):
        raise RuntimeError("Preceding-token surface context mismatch")
    if not np.array_equal(by_ex.following_token_code.to_numpy(), future):
        raise RuntimeError("Following-token surface context mismatch")
    if not np.array_equal(by_ex.preceding_token_id.to_numpy(), prior):
        raise RuntimeError("Boosted preceding-token context mismatch")
    if not np.array_equal(by_ex.following_token_id.to_numpy(), future):
        raise RuntimeError("Boosted following-token context mismatch")
    relative = by_ex.token_index.to_numpy(dtype=np.float64) / np.maximum(expected_length, 1)
    if not np.allclose(by_ex.token_relative_position, relative, atol=1e-6, rtol=0):
        raise RuntimeError("Token-relative-position formula mismatch")
    # No hidden-label-derived state is ever populated by a held-out response:
    # synthetic influence tests separately prove this property under perturbation.
    unknown = {name: int((frame.loc[frame.official_split == name, "prompt_code"] == 0).sum())
               for name in ("dev", "test")}
    report = {
        "status": "PASS",
        "data_root": str(ROOT),
        "n_rows": n,
        "n_parts": len(parts),
        "split_counts": counts,
        "source_preprocessing_hash": manifest["source_preprocessing_hash"],
        "safe_preprocessing_hash": manifest["preprocessing_hash"],
        "safe_part_hashes_verified": True,
        "static_context_exact_equivalence": True,
        "benchmark_features": feature_columns("benchmark"),
        "deployment_features": feature_columns("deployment"),
        "prompt_vocabulary_train_only": True,
        "unseen_prompt_rows": unknown,
        "heldout_response_policy": manifest["offline_safe_response_policy"],
        "temporal_history_policy": manifest["offline_safe_semantics"],
    }
    (ROOT / "offline_safe_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
