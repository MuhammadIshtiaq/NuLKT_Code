"""N6 - morphosyntactic error profiling and second-language transfer checks.

The corpus carries universal-dependencies part-of-speech tags, morphological
feature bundles, and dependency labels for every token. That makes it possible to
ask whether a purely data-driven model reproduces difficulty patterns that the
second-language-acquisition literature already documents for Spanish-speaking
learners of English.

The hypotheses below are registered *before* looking at the fitted numbers, and
each is tagged with its predicted direction. Recording predictions in advance is
what separates a transfer analysis from post-hoc storytelling, so the output
reports agreement and disagreement with equal prominence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Pre-registered expectations for Spanish-L1 learners of English. ``harder`` means
# the construction is predicted to show an above-average error rate relative to
# the corpus mean; ``easier`` the reverse. Each entry names the morphological or
# syntactic signature used to select the relevant tokens.
TRANSFER_HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "name": "third_person_singular_s",
        "prediction": "harder",
        "rationale": "Spanish marks person on the verb richly but English -s is a low-salience, semantically redundant affix frequently omitted by Spanish-L1 learners.",
        "morphology_contains": ["Person=3", "Number=Sing", "Tense=Pres"],
        "pos_in": ["VERB"],
    },
    {
        "name": "definite_and_indefinite_articles",
        "prediction": "easier",
        "rationale": "Spanish has a definite/indefinite article system, so article presence transfers positively even though distribution differs.",
        "pos_in": ["DET"],
        "morphology_contains": ["PronType=Art"],
    },
    {
        "name": "prepositions",
        "prediction": "harder",
        "rationale": "English preposition selection is largely idiosyncratic and maps many-to-many onto Spanish por/para/en/a, a classic persistent transfer difficulty.",
        "pos_in": ["ADP"],
    },
    {
        "name": "auxiliary_do_support",
        "prediction": "harder",
        "rationale": "Spanish forms negation and interrogation without an auxiliary, so English do-support has no L1 analogue.",
        "pos_in": ["AUX", "VERB"],
        "dependency_in": ["aux"],
    },
    {
        "name": "plural_noun_morphology",
        "prediction": "easier",
        "rationale": "Both languages mark nominal plurality with a productive suffix, supporting positive transfer.",
        "pos_in": ["NOUN"],
        "morphology_contains": ["Number=Plur"],
    },
    {
        "name": "pronoun_subjects",
        "prediction": "harder",
        "rationale": "Spanish is a pro-drop language, so obligatory overt subject pronouns in English lack an L1 counterpart.",
        "pos_in": ["PRON"],
        "dependency_in": ["nsubj"],
    },
)


def _select(frame: pd.DataFrame, hypothesis: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if "pos_in" in hypothesis and "part_of_speech" in frame:
        mask &= frame["part_of_speech"].astype(str).isin(hypothesis["pos_in"])
    if "dependency_in" in hypothesis and "dependency_label" in frame:
        mask &= frame["dependency_label"].astype(str).isin(hypothesis["dependency_in"])
    for feature in hypothesis.get("morphology_contains", []):
        if "morphology" in frame:
            mask &= frame["morphology"].astype(str).str.contains(feature, regex=False, na=False)
    return mask


def category_difficulty(
    frame: pd.DataFrame,
    column: str,
    *,
    min_count: int = 500,
    top_k: int | None = 25,
) -> pd.DataFrame:
    """Observed error rate and model discrimination per linguistic category."""

    if column not in frame.columns:
        raise KeyError(f"Missing linguistic column {column!r}")
    rows: list[dict[str, Any]] = []
    overall = float(frame["label"].mean())
    for value, group in frame.groupby(frame[column].astype(str), observed=True):
        if len(group) < min_count:
            continue
        both_classes = group["label"].nunique() == 2
        rows.append(
            {
                "category": value,
                "n": int(len(group)),
                "error_rate": float(group["label"].mean()),
                "error_rate_ratio_to_corpus": float(group["label"].mean() / overall) if overall > 0 else float("nan"),
                "mean_predicted_error": float(group["probability"].mean()),
                "auroc": float(roc_auc_score(group["label"], group["probability"])) if both_classes else float("nan"),
                "calibration_gap": float(group["probability"].mean() - group["label"].mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)
    return result.head(top_k) if top_k else result


def test_transfer_hypotheses(
    frame: pd.DataFrame,
    *,
    hypotheses: Iterable[dict[str, Any]] = TRANSFER_HYPOTHESES,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate each pre-registered prediction against observed difficulty.

    Significance uses a learner-clustered bootstrap of the difference between the
    construction's error rate and the corpus error rate, so repeated measurements
    from the same learner are not treated as independent evidence.
    """

    rng = np.random.default_rng(seed)
    overall = float(frame["label"].mean())
    learners = frame["learner_id"].to_numpy() if "learner_id" in frame else np.zeros(len(frame))
    unique_learners = np.unique(learners)
    learner_rows = {learner: np.flatnonzero(learners == learner) for learner in unique_learners}

    results: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        mask = _select(frame, hypothesis)
        subset = frame[mask]
        if len(subset) < 200:
            results.append({"name": hypothesis["name"], "status": "insufficient_tokens", "n": int(len(subset))})
            continue
        observed_rate = float(subset["label"].mean())
        difference = observed_rate - overall

        differences = np.empty(n_bootstrap, dtype=float)
        mask_values = mask.to_numpy()
        labels = frame["label"].to_numpy(dtype=float)
        for index in range(n_bootstrap):
            selected = rng.choice(unique_learners, size=len(unique_learners), replace=True)
            rows = np.concatenate([learner_rows[learner] for learner in selected])
            sample_mask = mask_values[rows]
            if sample_mask.sum() < 20:
                differences[index] = np.nan
                continue
            differences[index] = labels[rows][sample_mask].mean() - labels[rows].mean()
        valid = differences[np.isfinite(differences)]
        lower, upper = (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))) if len(valid) > 50 else (float("nan"), float("nan"))

        direction = "harder" if difference > 0 else "easier"
        supported = bool(
            direction == hypothesis["prediction"]
            and np.isfinite(lower)
            and (lower > 0 if hypothesis["prediction"] == "harder" else upper < 0)
        )
        both_classes = subset["label"].nunique() == 2
        results.append(
            {
                "name": hypothesis["name"],
                "status": "tested",
                "prediction": hypothesis["prediction"],
                "rationale": hypothesis["rationale"],
                "n": int(len(subset)),
                "error_rate": observed_rate,
                "corpus_error_rate": overall,
                "difference": difference,
                "ci_lower": lower,
                "ci_upper": upper,
                "observed_direction": direction,
                "prediction_supported": supported,
                "model_auroc_within_category": float(roc_auc_score(subset["label"], subset["probability"])) if both_classes else float("nan"),
                "model_mean_predicted_error": float(subset["probability"].mean()),
            }
        )

    tested = [row for row in results if row["status"] == "tested"]
    supported = [row for row in tested if row["prediction_supported"]]
    return {
        "hypotheses": results,
        "n_tested": len(tested),
        "n_supported": len(supported),
        "supported_names": [row["name"] for row in supported],
        "contradicted_names": [row["name"] for row in tested if not row["prediction_supported"]],
        "note": (
            "Directional predictions were fixed before fitting. Support indicates the corpus "
            "reproduces documented transfer asymmetries; it does not isolate L1 transfer from "
            "frequency, salience, or curriculum-ordering confounds present in the platform."
        ),
    }


def gate_attribution(model: Any) -> dict[str, Any]:
    """Report how the linguistic encoder distributes its input gates.

    Gate weights are a description of the encoder's input mixing, not a causal
    attribution; attention and gating weights do not constitute explanations.
    """

    encoder = getattr(model, "linguistic", None)
    if encoder is None:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "fields": list(encoder.fields),
        "caveat": "Gate magnitudes describe input mixing only and are not causal explanations.",
    }


def run(
    root: Path,
    *,
    seed: int = 42,
    split: str = "test",
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Execute N6 against the frozen proposed checkpoint."""

    from src.analysis.inference import (
        collect_predictions,
        indexed_dataloader,
        join_corpus_attributes,
        load_proposed_model,
    )

    output_dir = root / "results" / "analysis" / "linguistic_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    loader, frame = indexed_dataloader(
        root, split=split, sequence_length=sequence_length, batch_size=batch_size
    )
    predictions = collect_predictions(model, loader, resolved_device, with_expert_weights=False)
    enriched = join_corpus_attributes(
        predictions, frame, ["part_of_speech", "morphology", "dependency_label", "token"]
    )
    enriched["probability"] = enriched["logit_probability"]

    for column, filename in (
        ("part_of_speech", "difficulty_by_part_of_speech.csv"),
        ("dependency_label", "difficulty_by_dependency_label.csv"),
        ("morphology", "difficulty_by_morphology.csv"),
    ):
        category_difficulty(enriched, column).to_csv(output_dir / filename, index=False)

    transfer = test_transfer_hypotheses(enriched, seed=seed)
    payload = {
        "provenance": provenance,
        "partition": split,
        "observations": int(len(enriched)),
        "corpus_error_rate": float(enriched["label"].mean()),
        "transfer_analysis": transfer,
        "gate_attribution": gate_attribution(model),
    }
    (output_dir / "linguistic_profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
