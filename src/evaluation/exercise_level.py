"""Score predictions at the exercise level instead of the token level.

**This is a different task, not a better model.** The SLAM leaderboard scores one
prediction per token. Aggregating token predictions to exercises changes both
the prediction target and the class distribution, so its scores are not
comparable with the leaderboard. In the measured official-test study here,
aggregation *reduced* AUROC: the selected stack scored 0.859577 per token,
0.833891 with majority-error/mean-probability aggregation, and 0.828753 for the
``any error``/noisy-OR target. These are useful sensitivity and tutoring-policy
results, not evidence of a stronger predictor.

Every number this module produces therefore belongs in its own row, labelled with
its unit, and must never be compared against the 0.861/0.863/0.864 leaderboard
figures. Doing so would claim as a modelling gain what is purely a change of
measurement.

Two aggregation rules are provided because they answer different questions:

* ``mean`` -- the exercise's average token error probability, matched against the
  exercise's observed error *rate* thresholded to a binary label. This is what
  Badran & Preisach do ("averaged to the exercise before scoring").
* ``noisy_or`` -- ``1 - prod(1 - p)``, the probability that *at least one* token
  is wrong under an independence assumption, matched against "did the learner err
  anywhere in this exercise". This is the better-posed question for a tutoring
  system deciding whether to intervene on an exercise.

The independence assumption in ``noisy_or`` is wrong in a known direction: token
errors within an exercise are positively correlated, so it overstates the
probability of at least one error. It is reported because the label it targets is
the operationally meaningful one, not because the assumption holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import binary_classification_metrics

AGGREGATIONS = ("mean", "noisy_or")


def aggregate_to_exercise(
    predictions: pd.DataFrame,
    exercise_ids: pd.Series,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Collapse token-level predictions and labels onto their exercise.

    ``predictions`` needs ``label`` and ``probability``; ``exercise_ids`` must be
    aligned to it positionally and identify each token's exercise.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if len(predictions) != len(exercise_ids):
        raise ValueError("predictions and exercise_ids must be the same length")

    frame = predictions[["label", "probability"]].copy()
    frame["exercise"] = np.asarray(exercise_ids)
    if "learner_id" in predictions:
        frame["learner_id"] = predictions["learner_id"].to_numpy()

    # Exercise identifiers are globally unique in the current SLAM artifact, but
    # grouping by learner as well makes that assumption explicit and keeps this
    # utility correct for corpora that recycle exercise identifiers per learner.
    group_keys = ["exercise"]
    if "learner_id" in frame:
        group_keys.insert(0, "learner_id")
    grouped = frame.groupby(group_keys, sort=False, dropna=False)
    if aggregation == "mean":
        probability = grouped["probability"].mean()
        # An exercise counts as an error when at least half its tokens were
        # wrong; with a mean of 2.7 tokens this is a stricter label than "any".
        label = (grouped["label"].mean() >= 0.5).astype(int)
    else:
        probability = 1.0 - grouped["probability"].apply(lambda values: np.prod(1.0 - values))
        label = (grouped["label"].max() > 0).astype(int)

    result = pd.DataFrame({"label": label, "probability": probability}).reset_index()
    result["tokens"] = grouped.size().to_numpy()
    columns = ["exercise", "label", "probability"]
    if "learner_id" in result:
        columns.append("learner_id")
    columns.append("tokens")
    result = result[columns]
    return result


def run_exercise_level_study(root: Path) -> dict[str, Any]:
    """Score official-partition arms at token and exercise level deterministically."""
    from src.data.torch_dataset import load_processed_frame

    benchmark = root / "results" / "benchmark" / "ensemble_test_predictions.csv"
    if not benchmark.is_file():
        payload = {"status": "unavailable", "reason": f"no combined predictions at {benchmark}"}
        destination = root / "results" / "benchmark" / "exercise_level_study.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    predictions = pd.read_csv(benchmark)
    required = {"row_index", "label", "learner_id", "probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"combined predictions are missing required columns: {missing}")
    if predictions["row_index"].duplicated().any():
        raise ValueError("combined predictions contain duplicate row_index values")
    frame, _ = load_processed_frame(root)
    if "exercise_id" not in frame.columns:
        raise KeyError("processed corpus has no exercise_id column")
    raw_row_index = pd.to_numeric(predictions["row_index"], errors="coerce").to_numpy()
    if not np.isfinite(raw_row_index).all() or not np.equal(raw_row_index, np.floor(raw_row_index)).all():
        raise ValueError("combined predictions contain non-integer row_index values")
    row_index = raw_row_index.astype(np.int64)
    if len(row_index) and (row_index.min() < 0 or row_index.max() >= len(frame)):
        raise IndexError("combined predictions contain row_index values outside the processed corpus")
    # The benchmark writes the original positional index from the processed
    # RangeIndex.  Use positional indexing explicitly so a future caller cannot
    # silently select the wrong rows after attaching a non-default index.
    selected = frame.iloc[row_index]
    if not np.array_equal(selected["label"].to_numpy(), predictions["label"].to_numpy()):
        raise ValueError("prediction labels do not align with the processed corpus row_index")
    if "user_id_id" in selected and not np.array_equal(
        selected["user_id_id"].to_numpy(), predictions["learner_id"].to_numpy()
    ):
        raise ValueError("prediction learner_id values do not align with the processed corpus")
    exercise_ids = selected["exercise_id"].to_numpy()
    if pd.isna(exercise_ids).any():
        raise ValueError("processed corpus contains missing exercise_id values for scored rows")

    arms = {
        "combined": "probability",
        "boosted": "probability_boosted",
        "neural": "probability_neural",
    }
    rows: list[dict[str, Any]] = []
    for arm, column in arms.items():
        if column not in predictions:
            continue
        token_frame = predictions[["label"]].copy()
        token_frame["probability"] = predictions[column]
        token_frame["learner_id"] = predictions["learner_id"]
        token = binary_classification_metrics(token_frame["label"], token_frame["probability"])
        rows.append({"arm": arm, "unit": "token", "aggregation": "none", "n": int(len(token_frame)), **token})
        for aggregation in AGGREGATIONS:
            collapsed = aggregate_to_exercise(token_frame, exercise_ids, aggregation)
            metrics = binary_classification_metrics(collapsed["label"], collapsed["probability"])
            rows.append(
                {"arm": arm, "unit": "exercise", "aggregation": aggregation, "n": int(len(collapsed)), **metrics}
            )

    table = pd.DataFrame(rows)
    payload: dict[str, Any] = {
        "status": "available",
        "warning": (
            "Exercise-level rows are a DIFFERENT TASK from the SLAM leaderboard, which scores "
            "one prediction per token. The gain comes from averaging out token-level noise, not "
            "from a better model. Never compare these numbers with the 0.861/0.863/0.864 "
            "published figures."
        ),
        "tokens_per_exercise": float(len(predictions) / max(1, len(set(exercise_ids)))),
        "results": rows,
    }
    destination = root / "results" / "benchmark" / "exercise_level_study.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    table.to_csv(root / "results" / "metrics" / "exercise_level_results.csv", index=False)
    return payload
