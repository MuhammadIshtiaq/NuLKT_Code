"""N7 - auditing the explicit-instruction engine against observed behaviour.

The decision engine is deterministic and auditable but has never been checked
against anything the corpus actually records. Two audits are run here.

**Behavioural consistency.** The engine escalates to spaced review or retrieval
practice when its forgetting signal is high. SLAM contains natural variation in
how long learners leave between meeting the same token, so it is possible to ask
whether long gaps really do precede more errors once prior ability is accounted
for. Learners choose their own study gaps, so this is a matched observational
comparison, reported as consistency evidence and never as an intervention
effect. Covariate balance before and after matching is reported so a reader can
judge how much confounding remains.

**Operational profile.** Strategy coverage, abstention rate, and whether the
distribution of recommendations differs across country and device subgroups once
predicted risk is held constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.instruction.explicit_instruction_engine import ExplicitInstructionEngine

# Deterministic mapping from universal part-of-speech tags to the weakness
# vocabulary the engine understands. Declared here so the mapping is auditable
# rather than buried in a call site.
WEAKNESS_BY_POS: dict[str, str] = {
    "NOUN": "lexical", "PROPN": "lexical", "VERB": "lexical", "ADJ": "lexical", "ADV": "lexical",
    "NUM": "lexical", "INTJ": "lexical",
    "DET": "grammar", "ADP": "grammar", "PRON": "grammar", "AUX": "grammar",
    "SCONJ": "grammar", "CCONJ": "grammar", "PART": "grammar",
}
MATCHING_COVARIATES = (
    "prior_interactions",
    "prior_error_rate",
    "recent_20_error_rate",
    "prior_token_exposures",
    "prior_token_error_rate",
)


def assign_weakness(frame: pd.DataFrame) -> pd.Series:
    """Map each token to the engine's weakness vocabulary."""

    pos = frame.get("part_of_speech", pd.Series("", index=frame.index)).astype(str)
    weakness = pos.map(WEAKNESS_BY_POS).fillna("contextual")
    if "morphology" in frame:
        rich = frame["morphology"].astype(str).str.count(r"\|") >= 3
        weakness = weakness.mask(rich & pos.isin(["VERB", "NOUN"]), "morphology")
    return weakness


def apply_engine(
    frame: pd.DataFrame,
    engine: ExplicitInstructionEngine | None = None,
    *,
    sample: int | None = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the decision engine over model evidence, one row per interaction."""

    engine = engine or ExplicitInstructionEngine()
    working = frame
    if sample is not None and len(frame) > sample:
        working = frame.sample(n=sample, random_state=seed)
    weakness = assign_weakness(working)

    records: list[dict[str, Any]] = []
    for (_, row), weak in zip(working.iterrows(), weakness):
        recommendation = engine.recommend(
            {
                "error_probability": float(np.clip(row["probability"], 0.0, 1.0)),
                "mastery": float(np.clip(row.get("mastery", 0.0), 0.0, 1.0)),
                "uncertainty": float(np.clip(row.get("uncertainty", 0.0), 0.0, 1.0)),
                "forgetting_signal": float(np.clip(row.get("forgetting_signal", 0.0), 0.0, 1.0)),
                "previous_exposures": int(row.get("prior_token_exposures", 0)),
                "historical_errors": int(row.get("prior_token_errors", 0)),
                "weakness": weak,
            }
        )
        records.append(
            {
                "strategy": recommendation.strategy,
                "priority": recommendation.priority,
                "confidence": recommendation.confidence,
                "teacher_review": recommendation.teacher_review,
                "weakness": weak,
                "label": int(row["label"]),
            }
        )
    result = pd.DataFrame(records, index=working.index)
    for column in ("countries", "client", "probability", "uncertainty"):
        if column in working.columns:
            result[column] = working[column].to_numpy()
    return result


def calibrate_abstention_threshold(
    validation: pd.DataFrame,
    *,
    target_review_rates: Iterable[float] = (0.01, 0.05, 0.10, 0.20),
    configured_threshold: float = 0.45,
) -> dict[str, Any]:
    """Derive abstention thresholds from a teacher-review budget on validation.

    A fixed uncertainty threshold is only meaningful if the uncertainty scale is
    known in advance, which it is not: the Beta-evidential head sets its own
    scale during training. Selecting the threshold as a quantile of validation
    uncertainty makes the abstention rate an explicit design choice - the share
    of items a teacher agrees to review - instead of an accident of that scale.
    """

    if "uncertainty" not in validation.columns:
        raise KeyError("validation frame lacks an 'uncertainty' column")
    values = validation["uncertainty"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for rate in target_review_rates:
        threshold = float(np.quantile(values, 1.0 - float(rate)))
        rows.append(
            {
                "target_review_rate": float(rate),
                "threshold": threshold,
                "achieved_validation_rate": float((values >= threshold).mean()),
            }
        )
    return {
        "configured_threshold": configured_threshold,
        "configured_threshold_review_rate": float((values >= configured_threshold).mean()),
        "configured_threshold_active": bool((values >= configured_threshold).any()),
        "uncertainty_quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        "budget_calibrated_thresholds": rows,
        "note": (
            "Thresholds are quantiles of validation uncertainty only; no test data is used. "
            "If the configured threshold is inactive, the deployed system never abstains and "
            "the human-review safeguard described in the architecture is not operative."
        ),
    }


def strategy_profile(recommendations: pd.DataFrame) -> dict[str, Any]:
    """Coverage of each strategy and the observed error rate it is applied to."""

    grouped = recommendations.groupby("strategy", observed=True)
    profile = grouped.agg(
        n=("label", "size"),
        share=("label", lambda s: len(s) / len(recommendations)),
        observed_error_rate=("label", "mean"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    return {
        "n": int(len(recommendations)),
        "strategies": profile.sort_values("n", ascending=False).to_dict(orient="records"),
        "abstention_rate": float(recommendations["teacher_review"].mean()),
        "priority_distribution": recommendations["priority"].value_counts(normalize=True).to_dict(),
    }


def subgroup_recommendation_parity(
    recommendations: pd.DataFrame,
    *,
    attributes: Iterable[str] = ("countries", "client"),
    risk_bins: int = 5,
    min_count: int = 500,
) -> dict[str, Any]:
    """Compare strategy mixes across subgroups at matched predicted-risk levels.

    Conditioning on a predicted-risk stratum matters: an unconditional difference
    in recommendations across subgroups may simply reflect a genuine difference in
    difficulty rather than differential treatment.
    """

    results: dict[str, Any] = {}
    if "probability" not in recommendations:
        return {"status": "unavailable", "reason": "predicted risk column absent"}
    strata = pd.qcut(recommendations["probability"], q=risk_bins, duplicates="drop")
    for attribute in attributes:
        if attribute not in recommendations.columns:
            results[attribute] = {"status": "unavailable"}
            continue
        rows: list[dict[str, Any]] = []
        for stratum, block in recommendations.groupby(strata, observed=True):
            counts = block.groupby(attribute, observed=True).size()
            eligible = counts[counts >= min_count].index
            if len(eligible) < 2:
                continue
            block = block[block[attribute].isin(eligible)]
            mix = pd.crosstab(block[attribute], block["strategy"], normalize="index")
            # Total-variation distance between the most and least similar subgroup
            # mixes inside this risk stratum.
            spread = float((mix.max(axis=0) - mix.min(axis=0)).sum() / 2)
            rows.append(
                {
                    "risk_stratum": str(stratum),
                    "subgroups": int(len(eligible)),
                    "n": int(len(block)),
                    "max_total_variation_distance": spread,
                    "abstention_rate_range": float(
                        block.groupby(attribute, observed=True)["teacher_review"].mean().max()
                        - block.groupby(attribute, observed=True)["teacher_review"].mean().min()
                    ),
                }
            )
        results[attribute] = {
            "status": "tested" if rows else "insufficient_data",
            "strata": rows,
            "max_distance_across_strata": max((row["max_total_variation_distance"] for row in rows), default=None),
        }
    results["note"] = (
        "Total-variation distance is descriptive. A non-zero value flags differential "
        "recommendation mixes for review; it is not by itself evidence of unfair treatment."
    )
    return results


def _standardized_differences(
    treated: pd.DataFrame, control: pd.DataFrame, covariates: Iterable[str]
) -> dict[str, float]:
    differences: dict[str, float] = {}
    for name in covariates:
        a, b = treated[name].to_numpy(dtype=float), control[name].to_numpy(dtype=float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        differences[name] = float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0
    return differences


def matched_spacing_analysis(
    revisits: pd.DataFrame,
    *,
    covariates: Iterable[str] = MATCHING_COVARIATES,
    caliper: float = 0.02,
    seed: int = 42,
    max_pairs: int = 100_000,
) -> dict[str, Any]:
    """Compare accuracy after long versus short revisit gaps, propensity matched.

    Treatment is a revisit gap above the corpus median. Matching is on past-only
    covariates, so nothing observed at or after the outcome enters the model.
    """

    from sklearn.linear_model import LogisticRegression

    usable = revisits.dropna(subset=list(covariates) + ["lag_hours", "correct"]).copy()
    if len(usable) < 1000:
        return {"status": "insufficient_data", "n": int(len(usable))}
    median_lag = float(usable["lag_hours"].median())
    usable["treated"] = (usable["lag_hours"] > median_lag).astype(int)
    if usable["treated"].nunique() < 2:
        return {"status": "no_variation_in_gap"}

    features = usable[list(covariates)].to_numpy(dtype=float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    propensity = LogisticRegression(max_iter=1000).fit(features, usable["treated"]).predict_proba(features)[:, 1]
    usable["propensity"] = propensity

    treated = usable[usable["treated"] == 1]
    control = usable[usable["treated"] == 0]
    before = _standardized_differences(treated, control, covariates)

    rng = np.random.default_rng(seed)
    if len(treated) > max_pairs:
        treated = treated.sample(n=max_pairs, random_state=seed)
    control_sorted = control.sort_values("propensity")
    control_values = control_sorted["propensity"].to_numpy()
    positions = np.searchsorted(control_values, treated["propensity"].to_numpy())
    positions = np.clip(positions, 0, len(control_values) - 1)
    distances = np.abs(control_values[positions] - treated["propensity"].to_numpy())
    keep = distances <= caliper
    if keep.sum() < 200:
        return {"status": "insufficient_matches", "matched_pairs": int(keep.sum()), "caliper": caliper}

    matched_treated = treated[keep]
    matched_control = control_sorted.iloc[positions[keep]]
    after = _standardized_differences(matched_treated, matched_control, covariates)

    difference = float(matched_treated["correct"].mean() - matched_control["correct"].mean())
    # Paired bootstrap over matched pairs.
    pair_difference = matched_treated["correct"].to_numpy() - matched_control["correct"].to_numpy()
    draws = rng.choice(pair_difference, size=(1000, len(pair_difference)), replace=True).mean(axis=1)
    return {
        "status": "matched",
        "median_lag_hours": median_lag,
        "matched_pairs": int(keep.sum()),
        "caliper": caliper,
        "accuracy_long_gap": float(matched_treated["correct"].mean()),
        "accuracy_short_gap": float(matched_control["correct"].mean()),
        "accuracy_difference": difference,
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "standardized_differences_before_matching": before,
        "standardized_differences_after_matching": after,
        "max_absolute_standardized_difference_after": float(max(abs(value) for value in after.values())),
        "balance_acceptable": bool(max(abs(value) for value in after.values()) < 0.1),
        "causal_disclaimer": (
            "Learners chose their own study gaps. This is a matched observational contrast on "
            "past-only covariates and does not estimate the effect of assigning a review schedule. "
            "Unmeasured confounding, notably motivation and curriculum ordering, remains plausible."
        ),
    }


def run(
    root: Path,
    *,
    seed: int = 42,
    split: str = "test",
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
    engine_sample: int | None = 200_000,
) -> dict[str, Any]:
    """Execute N7 against the frozen proposed checkpoint."""

    from src.analysis.forgetting import token_revisit_lags
    from src.analysis.inference import (
        collect_predictions,
        indexed_dataloader,
        join_corpus_attributes,
        load_proposed_model,
    )
    from src.data.torch_dataset import load_processed_frame

    output_dir = root / "results" / "analysis" / "instruction_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    loader, frame = indexed_dataloader(
        root, split=split, sequence_length=sequence_length, batch_size=batch_size
    )
    predictions = collect_predictions(model, loader, resolved_device, with_expert_weights=False)
    enriched = join_corpus_attributes(
        predictions,
        frame,
        ["part_of_speech", "morphology", "prior_token_exposures", "prior_token_errors", "countries", "client"],
    )
    enriched["probability"] = enriched["logit_probability"]

    recommendations = apply_engine(enriched, sample=engine_sample, seed=seed)
    recommendations.to_csv(output_dir / "recommendations_sample.csv.gz", index=False, compression="gzip")

    # The abstention threshold is a design choice about teacher workload, so it is
    # calibrated on validation uncertainty and never on the test partition.
    validation_loader, frame = indexed_dataloader(
        root, split="validation", sequence_length=sequence_length, batch_size=batch_size, frame=frame
    )
    validation_predictions = collect_predictions(
        model, validation_loader, resolved_device, with_expert_weights=False
    )
    abstention = calibrate_abstention_threshold(
        validation_predictions, configured_threshold=float(model.config.uncertainty_threshold)
    )

    full_frame, _ = load_processed_frame(root)
    train = full_frame[full_frame["chronological_split"].astype(str) == "train"]
    revisits = token_revisit_lags(train)
    covariates = train.loc[revisits.index, list(MATCHING_COVARIATES)]
    spacing = matched_spacing_analysis(revisits.join(covariates), seed=seed)

    payload = {
        "provenance": provenance,
        "partition": split,
        "engine_thresholds": ExplicitInstructionEngine().thresholds(),
        "abstention_calibration": abstention,
        "strategy_profile": strategy_profile(recommendations),
        "subgroup_parity": subgroup_recommendation_parity(recommendations),
        "matched_spacing_analysis": spacing,
        "scope": (
            "This audit evaluates the recommendation logic's coverage, consistency, and "
            "subgroup behaviour. It does not evaluate whether delivering a recommendation "
            "improves learning, which requires a prospective controlled study."
        ),
    }
    (output_dir / "instruction_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
