"""Raw-data validation and severe-leakage audit gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import manifest_parts

try:
    from .download import project_root
    from .preprocess import discover_track_files, iter_exercises, load_key
except ImportError:  # direct script execution
    from download import project_root
    from preprocess import discover_track_files, iter_exercises, load_key


def validate_raw_corpus(
    *, root: Path | None = None, max_exercises: int | None = None
) -> dict[str, Any]:
    """Parse official files and fail on duplicate IDs, key mismatch, or time reversal."""

    root = (root or project_root()).resolve()
    files = discover_track_files(root)
    seen_ids: set[str] = set()
    last_user_time: dict[str, float] = {}
    learners: set[str] = set()
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for split in ("train", "dev", "test"):
        labels = None if split == "train" else load_key(Path(str(files[split]) + ".key"))
        for exercise in iter_exercises(files[split], official_split_name=split, external_labels=labels):
            counts[f"{split}_exercises"] += 1
            for row in exercise:
                instance = str(row["instance_id"])
                user = str(row["user_id"])
                timestamp = float(row["timestamp_days"])
                if instance in seen_ids:
                    errors.append(f"duplicate instance_id: {instance}")
                seen_ids.add(instance)
                if timestamp < last_user_time.get(user, float("-inf")):
                    errors.append(f"timestamp reversal for learner {user}")
                last_user_time[user] = timestamp
                learners.add(user)
                counts[f"{split}_interactions"] += 1
                counts[f"label_{int(row['label'])}"] += 1
            if max_exercises is not None and sum(
                counts[f"{name}_exercises"] for name in ("train", "dev", "test")
            ) >= max_exercises:
                break
        if max_exercises is not None and sum(
            counts[f"{name}_exercises"] for name in ("train", "dev", "test")
        ) >= max_exercises:
            break
    return {
        "status": "PASS" if not errors else "FAIL",
        "dataset": "Duolingo SLAM en_es (English learners who speak Spanish)",
        "counts": dict(counts),
        "learners": len(learners),
        "unique_instances": len(seen_ids),
        "errors": errors[:100],
        "truncated_error_count": max(0, len(errors) - 100),
        "partial_scan": max_exercises is not None,
    }


def _load_processed(root: Path) -> pd.DataFrame:
    manifest_path = root / "data" / "processed" / "en_es" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return pd.concat(
        [pd.read_parquet(path) for path in manifest_parts(root, manifest)],
        ignore_index=True,
    )


def audit_past_only_features(frame: pd.DataFrame) -> list[str]:
    """Recompute histories at exercise boundaries and return all violations."""

    required = {
        "instance_id",
        "exercise_id",
        "user_id",
        "timestamp_days",
        "row_order",
        "label",
        "prior_interactions",
        "prior_errors",
        "prior_token_exposures",
        "prior_token_errors",
        "token",
    }
    missing = required.difference(frame.columns)
    if missing:
        return [f"missing audit columns: {sorted(missing)}"]
    violations: list[str] = []
    ordered = frame.sort_values(["user_id", "timestamp_days", "row_order"], kind="stable")
    for user, learner in ordered.groupby("user_id", sort=False, observed=True):
        interactions = 0
        errors = 0
        token_counts: dict[str, int] = defaultdict(int)
        token_errors: dict[str, int] = defaultdict(int)
        previous_time = float("-inf")
        for _, exercise in learner.groupby("exercise_id", sort=False, observed=True):
            current_time = float(exercise["timestamp_days"].iloc[0])
            if current_time < previous_time:
                violations.append(f"future timestamp/order violation for user {user}")
            previous_time = current_time
            if not (exercise["prior_interactions"].astype(int) == interactions).all():
                violations.append(f"prior_interactions includes current/future data: {user}")
            if not (exercise["prior_errors"].astype(int) == errors).all():
                violations.append(f"prior_errors includes current/future targets: {user}")
            for row in exercise.itertuples(index=False):
                token = str(row.token)
                if int(row.prior_token_exposures) != token_counts[token]:
                    violations.append(f"token exposure leakage: {row.instance_id}")
                if int(row.prior_token_errors) != token_errors[token]:
                    violations.append(f"token target leakage: {row.instance_id}")
            for row in exercise.itertuples(index=False):
                label = int(row.label)
                interactions += 1
                errors += label
                token_counts[str(row.token)] += 1
                token_errors[str(row.token)] += label
            if len(violations) >= 100:
                return violations
    return violations


def _write_audit_text(root: Path, report: dict[str, Any]) -> None:
    output = root / "results" / "data_leakage_audit.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "NuLKT data leakage audit",
        f"Generated UTC: {report['audit_time_utc']}",
        f"Overall: {report['status']}",
        f"Rows audited: {report['rows_audited']}",
        "",
    ]
    for name, result in report["checks"].items():
        lines.extend([f"[{result['status']}] {name}", str(result["details"]), ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    metadata = root / "data" / "metadata" / "data_leakage_audit.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _stream_processed_audit(root: Path, *, raise_on_failure: bool) -> dict[str, Any]:
    """Audit all Parquet parts with bounded memory and exact exercise boundaries."""

    manifest_path = root / "data" / "processed" / "en_es" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = manifest_parts(root, manifest)
    seen_hashes: set[int] = set()
    duplicates = 0
    rows_audited = 0
    chronology_bad = 0
    invalid_ids = 0
    history_violations: list[str] = []
    interactions: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    token_counts: dict[tuple[str, str], int] = defaultdict(int)
    token_errors: dict[tuple[str, str], int] = defaultdict(int)
    last_time: dict[str, float] = {}
    cold_partition: dict[str, str] = {}
    cold_overlap_users: set[str] = set()
    missing_counts: Counter[str] = Counter()
    split_counts = {
        protocol: Counter()
        for protocol in ("official_split", "chronological_split", "cold_start_split")
    }
    label_domain: set[int] = set()
    learners: set[str] = set()
    all_columns: list[str] = []
    for path in parts:
        frame = pd.read_parquet(path)
        if not all_columns:
            all_columns = list(frame.columns)
        rows_audited += len(frame)
        for column in frame.columns:
            missing_counts[column] += int(frame[column].isna().sum())
        hashes = pd.util.hash_pandas_object(frame["instance_id"], index=False).to_numpy()
        for value in hashes:
            hashed = int(value)
            if hashed in seen_hashes:
                duplicates += 1
            seen_hashes.add(hashed)
        label_domain.update(int(value) for value in frame["label"].dropna().unique())
        learners.update(frame["user_id"].astype(str).unique())
        for protocol, counts in split_counts.items():
            counts.update(frame[protocol].astype(str))
        for row in frame[["user_id", "cold_start_split"]].drop_duplicates().itertuples(index=False):
            user, partition = str(row.user_id), str(row.cold_start_split)
            existing = cold_partition.setdefault(user, partition)
            if existing != partition:
                cold_overlap_users.add(user)
        encoded = [
            column
            for column in frame
            if column.endswith("_id") and pd.api.types.is_numeric_dtype(frame[column])
        ]
        invalid_ids += sum(int((frame[column] < 0).sum()) for column in encoded)
        for (user_value, _), exercise in frame.groupby(
            ["user_id", "exercise_id"], sort=False, observed=True
        ):
            user = str(user_value)
            current_time = float(exercise["timestamp_days"].iloc[0])
            if current_time < last_time.get(user, float("-inf")):
                chronology_bad += 1
                if len(history_violations) < 100:
                    history_violations.append(f"timestamp reversal for user {user}")
            if not (exercise["prior_interactions"].astype(int) == interactions[user]).all():
                if len(history_violations) < 100:
                    history_violations.append(f"prior_interactions leakage for user {user}")
            if not (exercise["prior_errors"].astype(int) == errors[user]).all():
                if len(history_violations) < 100:
                    history_violations.append(f"prior_errors leakage for user {user}")
            for row in exercise.itertuples(index=False):
                key = (user, str(row.token))
                if int(row.prior_token_exposures) != token_counts[key] and len(history_violations) < 100:
                    history_violations.append(f"token exposure leakage: {row.instance_id}")
                if int(row.prior_token_errors) != token_errors[key] and len(history_violations) < 100:
                    history_violations.append(f"token error leakage: {row.instance_id}")
            for row in exercise.itertuples(index=False):
                label = int(row.label)
                key = (user, str(row.token))
                interactions[user] += 1
                errors[user] += label
                token_counts[key] += 1
                token_errors[key] += label
            last_time[user] = current_time
        del frame

    checks: dict[str, dict[str, Any]] = {
        "duplicate_instances": {
            "status": "PASS" if duplicates == 0 else "FAIL",
            "details": f"{duplicates} duplicate instance-ID hashes (64-bit; collision probability negligible)",
        },
        "past_only_history": {
            "status": "PASS" if not history_violations else "FAIL",
            "details": history_violations[:20] or "all histories use earlier exercises only",
        },
        "cold_start_learner_disjointness": {
            "status": "PASS" if not cold_overlap_users else "FAIL",
            "details": f"{len(cold_overlap_users)} learners occur in multiple cold-start partitions",
        },
        "chronological_order": {
            "status": "PASS" if chronology_bad == 0 else "FAIL",
            "details": f"{chronology_bad} learner-level timestamp reversals",
        },
        "categorical_encoding": {
            "status": "PASS" if invalid_ids == 0 else "FAIL",
            "details": f"{invalid_ids} negative categorical IDs; vocabularies are train-fitted by manifest",
        },
        "pre_response_feature_policy": {
            "status": "WARNING",
            "details": (
                "label is target-only. response_time_seconds is post-response and preserved solely for "
                "official-benchmark compatibility; both must be excluded from model inputs."
            ),
            "excluded_from_default": ["label", "response_time_seconds"],
        },
    }
    severe = [name for name, result in checks.items() if result["status"] == "FAIL"]
    report = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if severe else "PASS",
        "severe_failures": severe,
        "rows_audited": rows_audited,
        "checks": checks,
    }
    _write_audit_text(root, report)
    missing_report = {
        "status": "FAIL" if missing_counts.get("label", 0) else "PASS",
        "rows": rows_audited,
        "policy": (
            "Missing response time is retained as NaN; categorical missing values map to unknown. "
            "No target labels may be missing in processed labeled partitions."
        ),
        "target_missing": missing_counts.get("label", 0),
        "columns": {
            column: {
                "missing_count": missing_counts[column],
                "missing_fraction": missing_counts[column] / rows_audited,
            }
            for column in all_columns
        },
    }
    integrity_report = {
        "status": report["status"],
        "rows": rows_audited,
        "unique_instances": len(seen_hashes),
        "unique_learners": len(learners),
        "label_domain": sorted(label_domain),
        "duplicate_instances": duplicates,
        "timestamp_reversals": chronology_bad,
        "source": "processed data derived from checksum-verified Harvard Dataverse files",
        "preprocessing_hash": manifest.get("preprocessing_hash"),
    }
    split_report = {
        "status": "PASS" if duplicates == 0 and not cold_overlap_users else "FAIL",
        "duplicate_instances": duplicates,
        "cold_start_overlapping_learners": len(cold_overlap_users),
        "counts": {name: dict(counts) for name, counts in split_counts.items()},
    }
    for name, payload in (
        ("missing_data_report.json", missing_report),
        ("data_integrity_report.json", integrity_report),
        ("split_contamination_audit.json", split_report),
    ):
        (root / "results" / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if severe and raise_on_failure:
        raise RuntimeError(f"Severe data leakage audit failure(s): {', '.join(severe)}")
    return report


def run_leakage_audit(
    *,
    root: Path | None = None,
    frame: pd.DataFrame | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Audit target histories, split duplication, contamination, and encoder IDs."""

    root = (root or project_root()).resolve()
    if frame is None:
        return _stream_processed_audit(root, raise_on_failure=raise_on_failure)
    frame = frame.copy()
    checks: dict[str, dict[str, Any]] = {}

    duplicates = int(frame["instance_id"].duplicated().sum())
    checks["duplicate_instances"] = {
        "status": "PASS" if duplicates == 0 else "FAIL",
        "details": f"{duplicates} duplicate instance IDs",
    }
    history_violations = audit_past_only_features(frame)
    checks["past_only_history"] = {
        "status": "PASS" if not history_violations else "FAIL",
        "details": history_violations[:20] or "all histories use earlier exercises only",
    }
    cold_overlap = int(
        (frame.groupby("user_id", observed=True)["cold_start_split"].nunique() > 1).sum()
    )
    checks["cold_start_learner_disjointness"] = {
        "status": "PASS" if cold_overlap == 0 else "FAIL",
        "details": f"{cold_overlap} learners occur in multiple cold-start partitions",
    }
    chronology_bad = 0
    for _, learner in frame.sort_values("row_order").groupby("user_id", sort=False, observed=True):
        values = learner["timestamp_days"].to_numpy()
        chronology_bad += int((values[1:] < values[:-1]).sum())
    checks["chronological_order"] = {
        "status": "PASS" if chronology_bad == 0 else "FAIL",
        "details": f"{chronology_bad} learner-level timestamp reversals",
    }
    encoded = [
        column
        for column in frame
        if column.endswith("_id") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    invalid_ids = sum(int((frame[column] < 0).sum()) for column in encoded)
    checks["categorical_encoding"] = {
        "status": "PASS" if invalid_ids == 0 else "FAIL",
        "details": f"{invalid_ids} negative categorical IDs; vocabularies are train-fitted by manifest",
    }
    forbidden_default_features = ["label", "response_time_seconds"]
    checks["pre_response_feature_policy"] = {
        "status": "WARNING",
        "details": (
            "label is target-only. response_time_seconds is post-response and preserved solely for "
            "official-benchmark compatibility; both must be excluded from model inputs."
        ),
        "excluded_from_default": forbidden_default_features,
    }
    severe = [name for name, result in checks.items() if result["status"] == "FAIL"]
    report = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if severe else "PASS",
        "severe_failures": severe,
        "rows_audited": len(frame),
        "checks": checks,
    }
    output = root / "results" / "data_leakage_audit.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "NuLKT data leakage audit",
        f"Generated UTC: {report['audit_time_utc']}",
        f"Overall: {report['status']}",
        f"Rows audited: {len(frame)}",
        "",
    ]
    for name, result in checks.items():
        lines.extend([f"[{result['status']}] {name}", str(result["details"]), ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    (root / "data" / "metadata" / "data_leakage_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    missing = {
        column: {
            "missing_count": int(frame[column].isna().sum()),
            "missing_fraction": float(frame[column].isna().mean()),
        }
        for column in frame.columns
    }
    missing_report = {
        "status": "PASS",
        "rows": len(frame),
        "policy": (
            "Missing response time is retained as NaN; categorical missing values map to unknown. "
            "No target labels may be missing in processed labeled partitions."
        ),
        "target_missing": int(frame["label"].isna().sum()),
        "columns": missing,
    }
    if missing_report["target_missing"]:
        missing_report["status"] = "FAIL"
    (root / "results" / "missing_data_report.json").write_text(
        json.dumps(missing_report, indent=2), encoding="utf-8"
    )
    integrity_report = {
        "status": report["status"],
        "rows": len(frame),
        "unique_instances": int(frame["instance_id"].nunique()),
        "unique_learners": int(frame["user_id"].nunique()),
        "label_domain": sorted(int(value) for value in frame["label"].dropna().unique()),
        "duplicate_instances": duplicates,
        "timestamp_reversals": chronology_bad,
        "source": "processed data derived from checksum-verified Harvard Dataverse files",
    }
    (root / "results" / "data_integrity_report.json").write_text(
        json.dumps(integrity_report, indent=2), encoding="utf-8"
    )
    split_report = {
        "status": "PASS" if duplicates == 0 and cold_overlap == 0 else "FAIL",
        "duplicate_instances": duplicates,
        "cold_start_overlapping_learners": cold_overlap,
        "counts": {
            protocol: frame[protocol].astype(str).value_counts().to_dict()
            for protocol in ("official_split", "chronological_split", "cold_start_split")
        },
    }
    (root / "results" / "split_contamination_audit.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    if severe and raise_on_failure:
        raise RuntimeError(f"Severe data leakage audit failure(s): {', '.join(severe)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--raw", action="store_true", help="validate raw corpus instead of processed data")
    parser.add_argument("--max-exercises", type=int, default=None)
    args = parser.parse_args()
    result = (
        validate_raw_corpus(root=args.root, max_exercises=args.max_exercises)
        if args.raw
        else run_leakage_audit(root=args.root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
