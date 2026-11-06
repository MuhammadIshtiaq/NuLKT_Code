"""Model-combination search over the official-partition arms.

The boosted and neural arms disagree enough to be worth combining (their test
probabilities correlate at 0.90), but a fixed-weight probability average
recovers very little of that headroom. This module searches a family of
combiners under learner-grouped cross-validation *inside the dev partition*,
freezes the winner, and only then reads the test partition once.

Protocol invariants, all enforced rather than assumed:

* every combiner is fitted on dev and scored on test exactly once;
* combiner selection never sees a test label - selection is by grouped-CV AUROC
  on dev alone;
* cross-validation folds split on ``learner_id`` so that no learner's rows
  appear in both the fitting and scoring half of a fold, which would leak the
  learner-specific signal the combiner is meant to generalise across.

Diversity extensions (opt-in; the default code path is byte-for-byte the study
that has already been published)
-------------------------------------------------------------------------

Giving the neural arm the boosted model's own features made it stronger *and*
made it redundant: the two arms' test probabilities went from correlating at
0.84 to correlating at 0.90, and the combination gain collapsed accordingly.
The extensions below exist to test whether that redundancy is what caps the
ensemble:

* :func:`load_official_arms` can pull additional arms that are on the *same*
  official partition, currently the pre-feature-transfer ("bare") NuLKT,
  which is weaker but decorrelated from the boosted arm;
* :func:`build_combiners` and the stacking helpers take an arbitrary arm tuple,
  so a combiner can use a *subset* of the available arms - subset choice then
  becomes part of the ordinary dev-CV selection rather than a separate step;
* :func:`select_combiner` can score candidates on CV AUROC penalised by the
  mean pairwise correlation of the arms they consume, which is the operational
  form of "prefer complementary arms";
* :func:`grouped_oof_predictions` produces out-of-fold meta-learner predictions
  under the same learner-grouped folds, so a stacker is never scored on a
  learner it was fitted on.

Every extension is off by default. ``run_ensemble_study`` still takes no new
required argument and still writes the same files; the diversity search lives
in :func:`run_diversity_ensemble_study` and writes to its own output path so it
cannot overwrite the published result.

What the search actually found (official test partition, measured)
------------------------------------------------------------------

The redundancy hypothesis did not survive contact with the data. Grouping the
candidates by which arms they use, and reading off each group's dev-CV winner:

===================  =====================  ==========  ===========
arm subset           mean pairwise corr.    dev CV      test AUROC
===================  =====================  ==========  ===========
neural+boosted+bare  0.8998                 0.859083    0.859673
neural+boosted       0.8985                 0.859066    0.859577
boosted+bare         0.8535                 0.859000    0.859480
boosted alone        -                      0.858924    0.859336
===================  =====================  ==========  ===========

The *least* correlated pair, boosted+bare at 0.8535, produces the *worst*
combination; correlation ranks inversely to combination gain across the whole
table. Diversity is therefore not what caps this ensemble - arm strength is.
The bare arm reaches only 0.8276 on test, and a weak arm cannot contribute much
however decorrelated it is. Selecting with ``criterion='diversity_penalised'``
at any penalty above about 0.002 duly switches to boosted+bare and *loses*
0.000193 AUROC against plain CV selection.

Keep the correlation-penalised option anyway: it is what makes that a measured
negative result rather than an untested assumption, and it becomes worth
re-running the moment a genuinely strong decorrelated arm exists.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .metrics import binary_classification_metrics
from .statistics import learner_clustered_bootstrap

EPSILON = 1e-6

#: The two arms the published study combines. Every generalised helper defaults
#: to this tuple so that omitting ``arms`` reproduces the original behaviour.
DEFAULT_ARMS: tuple[str, ...] = ("neural", "boosted")

#: Arms whose saved predictions live on the *official* dev/test partition and
#: carry ``row_index``, so they can be joined row-for-row. Anything not listed
#: here is on a different partition and must never be mixed in - see
#: :data:`PARTITION_INCOMPATIBLE_RUNS` for what was checked and rejected.
OPTIONAL_ARM_SOURCES: dict[str, str] = {
    "bare": "superseded_official_bare_config_20260813",
}

#: Saved runs that were audited as candidate arms and rejected, with the reason.
#: Recorded because "why is the ten-model baseline suite not in the ensemble?"
#: is otherwise a question that gets re-answered by someone re-deriving it, and
#: because the ``best_baseline`` case is a trap: it carries ``row_index`` and
#: joins without error, but its rows straddle the official dev *and* test
#: partitions, so an inner join yields an ensemble scored on neither.
PARTITION_INCOMPATIBLE_RUNS: dict[str, str] = {
    "baseline_*": "chronological split, 532834 validation rows, no row_index and no test split",
    "classical_baselines": "chronological split, validation predictions only, no row_index",
    "classical_validation": "chronological split, validation predictions only, no row_index",
    "proposed": "chronological split, 534973 test rows, no row_index",
    "enhanced": "chronological split, 534973 test rows, no row_index",
    "best_baseline": (
        "chronological split; its 534973 test rows carry row_index but overlap the official "
        "test partition on 381312 rows and the official dev partition on 151105, so joining it "
        "would mix partitions"
    ),
    "cold_start_eip_ktformer": (
        "cold-start learner-restricted partition of 516856 rows; covers only 57989 of the "
        "386604 official test rows, so it cannot score the test partition"
    ),
    "cold_start_nulkt": (
        "cold-start learner-restricted partition; it cannot score the complete official "
        "test partition"
    ),
    "superseded_neural_only_selection_20260811": "chronological split, no row_index",
    "superseded_pre_leakage_fix_20260810": "chronological split and pre-leakage-fix predictions",
}

#: Human-readable arm names for the result tables.
ARM_LABELS: dict[str, str] = {
    "neural": "NuLKT",
    "boosted": "Gradient boosting",
    "bare": "NuLKT without transferred boosted features",
}


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _rank_uniform(value: np.ndarray) -> np.ndarray:
    """Map scores to (0, 1) by rank, which removes each arm's scale entirely."""
    order = np.asarray(value, dtype=float).argsort().argsort()
    return (order + 0.5) / len(order)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC; ties averaged. Returns NaN for a single-class input."""
    y = np.asarray(y, dtype=int)
    positive, negative = int(y.sum()), int((1 - y).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.asarray(p, dtype=float).argsort()
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    sorted_p = np.asarray(p, dtype=float)[order]
    start = 0
    for index in range(1, len(sorted_p) + 1):
        if index == len(sorted_p) or sorted_p[index] != sorted_p[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


@dataclass
class Combiner:
    """One candidate combination rule: fit on a dev slice, apply to any slice.

    ``arms`` records which prediction arms the rule actually consumes. It is
    what makes subset search and the correlation penalty possible: a combiner
    that ignores an arm must not be charged for that arm's redundancy.
    """

    name: str
    description: str
    fit: Callable[[pd.DataFrame, np.ndarray], Any]
    apply: Callable[[Any, pd.DataFrame], np.ndarray]
    parameters: dict[str, Any] = field(default_factory=dict)
    arms: tuple[str, ...] = DEFAULT_ARMS


def _weight_grid(step: float = 0.02) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + step / 2, step), 4)


def _simplex_grid(n_arms: int, step: float = 0.05) -> Iterator[tuple[float, ...]]:
    """Enumerate non-negative weight vectors summing to one, on a lattice.

    Integer arithmetic keeps the constraint exact; floating accumulation would
    leave weights summing to 0.9999999 and silently rescale the blend.
    """
    ticks = int(round(1.0 / step))

    def recurse(remaining: int, depth: int) -> Iterator[tuple[int, ...]]:
        if depth == n_arms - 1:
            yield (remaining,)
            return
        for taken in range(remaining + 1):
            for rest in recurse(remaining - taken, depth + 1):
                yield (taken, *rest)

    for combination in recurse(ticks, 0):
        yield tuple(value / ticks for value in combination)


def _arm_values(
    frame: pd.DataFrame, arms: Sequence[str], transform: Callable[[np.ndarray], np.ndarray]
) -> list[np.ndarray]:
    return [transform(frame[f"probability_{arm}"].to_numpy()) for arm in arms]


def _blend(values: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted sum, accumulated term by term.

    Deliberately not a matrix product: BLAS accumulates in a different order,
    which shifts the blended rank scores by ~1e-10 and moves the published
    ``rank_blend`` figure in its last digits for no reason at all.
    """
    total = float(weights[0]) * values[0]
    for weight, value in zip(weights[1:], values[1:]):
        total = total + float(weight) * value
    return total


def _fit_best_weight(
    frame: pd.DataFrame,
    y: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    arms: Sequence[str] = DEFAULT_ARMS,
) -> dict:
    """Grid-search mixing weights on the supplied slice.

    Two arms keep the original one-dimensional sweep, so the published result
    is reproduced exactly; three or more sweep the probability simplex.
    """
    values = _arm_values(frame, arms, transform)
    if len(arms) == 1:
        return {"weights": [1.0], "dev_auroc": float(_auroc(y, values[0]))}
    if len(arms) == 2:
        a, b = values
        best_weight, best_score = 0.0, -np.inf
        for weight in _weight_grid():
            score = _auroc(y, weight * a + (1.0 - weight) * b)
            if score > best_score:
                best_weight, best_score = float(weight), score
        return {"weight": best_weight, "weights": [best_weight, 1.0 - best_weight],
                "dev_auroc": float(best_score)}
    best_weights, best_score = None, -np.inf
    for weights in _simplex_grid(len(arms)):
        score = _auroc(y, _blend(values, weights))
        if score > best_score:
            best_weights, best_score = list(weights), score
    return {"weights": best_weights, "dev_auroc": float(best_score)}


def _apply_weight(
    state: dict,
    frame: pd.DataFrame,
    transform: Callable[[np.ndarray], np.ndarray],
    arms: Sequence[str] = DEFAULT_ARMS,
) -> np.ndarray:
    return _blend(_arm_values(frame, arms, transform), state["weights"])


def _stack_matrix(
    frame: pd.DataFrame,
    interactions: bool,
    arms: Sequence[str] = DEFAULT_ARMS,
    meta_features: bool = False,
) -> np.ndarray:
    """Design matrix for a stacker: arm log-odds, then optional extra columns.

    For the default two arms with ``interactions=True`` the columns are exactly
    ``[neural, boosted, neural*boosted, |neural-boosted|]`` as before.
    """
    logits = _arm_values(frame, arms, _logit)
    columns = list(logits)
    if interactions:
        # The arms disagree most where they are individually least certain, so
        # give the stacker the disagreement and the joint-confidence signal.
        for left, right in itertools.combinations(range(len(logits)), 2):
            columns.append(logits[left] * logits[right])
            columns.append(np.abs(logits[left] - logits[right]))
    if meta_features:
        history = _meta_column(frame, "log_history_length")
        columns.append(history)
        # Interacted, not just added: the question is whether an arm should be
        # trusted *more* on learners with a long history, not whether history
        # predicts errors on its own.
        columns.extend(logit * history for logit in logits)
    return np.column_stack(columns)


def _meta_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise KeyError(
            f"meta-feature '{name}' is missing; call add_meta_features() on each "
            "partition separately before fitting a meta-feature combiner"
        )
    return frame[name].to_numpy(dtype=float)


def add_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach learner-history meta-features, computed *within* one partition.

    ``history_length`` counts a learner's earlier rows in ``row_index`` order,
    so it is causal: row *n* never sees rows *n+1...*. It is derived from the
    prediction frame alone - no label is consulted - and must be computed
    per partition, because a count that spanned dev and test would let a test
    row's feature depend on how many dev rows the learner happened to have.
    """
    if "row_index" not in frame.columns or "learner_id" not in frame.columns:
        raise KeyError("meta-features need 'row_index' and 'learner_id' columns")
    # Scatter positionally rather than by label: the caller's frame keeps its own
    # row order, and a duplicated or non-monotonic pandas index must not be able
    # to misalign a feature against its label.
    order = np.argsort(frame["row_index"].to_numpy(), kind="mergesort")
    counts = frame.iloc[order].groupby("learner_id").cumcount().to_numpy(dtype=float)
    history = np.empty(len(frame), dtype=float)
    history[order] = counts
    enriched = frame.copy()
    enriched["history_length"] = history
    enriched["log_history_length"] = np.log1p(history)
    return enriched


def _fit_stacker(
    frame: pd.DataFrame,
    y: np.ndarray,
    interactions: bool,
    arms: Sequence[str] = DEFAULT_ARMS,
    meta_features: bool = False,
) -> dict:
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(_stack_matrix(frame, interactions, arms, meta_features), y)
    return {"model": model, "interactions": interactions, "arms": tuple(arms),
            "meta_features": meta_features}


def _apply_stacker(state: dict, frame: pd.DataFrame) -> np.ndarray:
    matrix = _stack_matrix(
        frame, state["interactions"], state.get("arms", DEFAULT_ARMS),
        state.get("meta_features", False),
    )
    return state["model"].predict_proba(matrix)[:, 1]


def _fit_boosted_stacker(
    frame: pd.DataFrame,
    y: np.ndarray,
    arms: Sequence[str] = DEFAULT_ARMS,
    meta_features: bool = False,
    seed: int = 42,
) -> dict:
    """Gradient-boosted meta-learner over the arm log-odds.

    ``early_stopping=False`` is deliberate and load-bearing: scikit-learn's
    internal early-stopping split is a *random* row split, so it would validate
    the meta-learner on rows belonging to learners in its own training set -
    exactly the leak the learner-grouped protocol exists to prevent. The
    iteration budget is fixed instead, and the whole stacker is judged by the
    outer grouped CV like every other candidate.
    """
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(_stack_matrix(frame, False, arms, meta_features), y)
    return {"model": model, "interactions": False, "arms": tuple(arms),
            "meta_features": meta_features}


def _reference_combiner(arm: str) -> Combiner:
    label = ARM_LABELS.get(arm, arm)
    return Combiner(
        f"{arm}_only", f"{label} alone (reference arm)",
        lambda frame, y: {}, lambda state, frame, arm=arm: frame[f"probability_{arm}"].to_numpy(),
        arms=(arm,),
    )


def build_combiners(arms: Sequence[str] = DEFAULT_ARMS) -> list[Combiner]:
    """The candidate family, ordered from simplest to most expressive.

    Called with no argument this returns exactly the seven combiners the
    published study searched, under their original names.
    """
    arms = tuple(arms)
    suffix = "" if arms == DEFAULT_ARMS else "_" + "_".join(arms)
    joined = " + ".join(ARM_LABELS.get(arm, arm) for arm in arms)
    blend_word = "Fixed-weight" if len(arms) == 2 else "Simplex-weighted"
    return [
        *(_reference_combiner(arm) for arm in arms),
        Combiner(
            f"probability_blend{suffix}",
            "Fixed-weight average of probabilities (incumbent method)"
            if arms == DEFAULT_ARMS else f"{blend_word} average of probabilities over {joined}",
            lambda frame, y: _fit_best_weight(frame, y, lambda v: v, arms),
            lambda state, frame: _apply_weight(state, frame, lambda v: v, arms),
            arms=arms,
        ),
        Combiner(
            f"logit_blend{suffix}",
            "Weighted average in log-odds space"
            if arms == DEFAULT_ARMS else f"Weighted average in log-odds space over {joined}",
            lambda frame, y: _fit_best_weight(frame, y, _logit, arms),
            lambda state, frame: _sigmoid(_apply_weight(state, frame, _logit, arms)),
            arms=arms,
        ),
        Combiner(
            f"rank_blend{suffix}",
            "Weighted average of within-split ranks (scale-free)"
            if arms == DEFAULT_ARMS else f"Weighted average of within-split ranks over {joined}",
            lambda frame, y: _fit_best_weight(frame, y, _rank_uniform, arms),
            lambda state, frame: _apply_weight(state, frame, _rank_uniform, arms),
            arms=arms,
        ),
        Combiner(
            f"logistic_stack{suffix}",
            "Logistic regression over both arms' log-odds"
            if arms == DEFAULT_ARMS else f"Logistic regression over the log-odds of {joined}",
            lambda frame, y: _fit_stacker(frame, y, False, arms),
            _apply_stacker,
            arms=arms,
        ),
        Combiner(
            f"logistic_stack_interactions{suffix}",
            "Logistic stack plus agreement and disagreement terms"
            if arms == DEFAULT_ARMS
            else f"Logistic stack over {joined} plus pairwise agreement and disagreement terms",
            lambda frame, y: _fit_stacker(frame, y, True, arms),
            _apply_stacker,
            arms=arms,
        ),
    ]


def build_diversity_combiners(
    arms: Sequence[str],
    min_subset_size: int = 2,
    include_boosted_stack: bool = True,
    include_meta_features: bool = True,
    seed: int = 42,
) -> list[Combiner]:
    """Candidates over every arm *subset*, so subset choice is part of selection.

    Restricting a stacker to a subset is the cheapest way to buy diversity: if
    two arms are near-duplicates, a combiner that drops one of them is a real
    candidate and dev CV gets to say so. Doing it inside one CV ranking - not
    as a separate pre-filter - keeps the single-selection protocol intact.
    """
    arms = tuple(arms)
    if len(set(arms)) != len(arms):
        raise ValueError("arm names must be unique")
    candidates: list[Combiner] = [_reference_combiner(arm) for arm in arms]
    seen: set[str] = {combiner.name for combiner in candidates}
    for size in range(max(2, min_subset_size), len(arms) + 1):
        for subset in itertools.combinations(arms, size):
            for combiner in build_combiners(subset):
                if combiner.name not in seen:
                    seen.add(combiner.name)
                    candidates.append(combiner)
    joined = " + ".join(ARM_LABELS.get(arm, arm) for arm in arms)
    if include_meta_features:
        candidates.append(
            Combiner(
                "logistic_stack_meta_" + "_".join(arms),
                f"Logistic stack over {joined} with learner-history meta-features",
                lambda frame, y: _fit_stacker(frame, y, True, arms, meta_features=True),
                _apply_stacker,
                arms=arms,
            )
        )
    if include_boosted_stack:
        candidates.append(
            Combiner(
                "boosted_stack_" + "_".join(arms),
                f"Gradient-boosted meta-learner over the log-odds of {joined}",
                lambda frame, y: _fit_boosted_stacker(frame, y, arms, False, seed),
                _apply_stacker,
                arms=arms,
            )
        )
        if include_meta_features:
            candidates.append(
                Combiner(
                    "boosted_stack_meta_" + "_".join(arms),
                    f"Gradient-boosted meta-learner over {joined} with learner history",
                    lambda frame, y: _fit_boosted_stacker(frame, y, arms, True, seed),
                    _apply_stacker,
                    arms=arms,
                )
            )
    return candidates


def _load_seed_averaged_arm(
    base: Path, split: str, seeds: Sequence[int] | None
) -> tuple[pd.DataFrame, list[int]]:
    """Average a multi-seed arm's predictions over its seeds.

    A single seed understates a neural arm: seed averaging is free variance
    reduction that the boosted arm effectively already has through its own
    ensembling, so comparing one neural seed against it would stack the deck
    against the neural model.
    """
    available = sorted(
        int(path.name.removeprefix("seed_"))
        for path in base.glob("seed_*")
        if (path / f"{split}_predictions.csv").is_file()
    )
    chosen = available if seeds is None else [s for s in seeds if s in available]
    if not chosen:
        raise FileNotFoundError(f"no official neural {split} predictions under {base}")
    accumulated: pd.DataFrame | None = None
    for seed in chosen:
        frame = pd.read_csv(base / f"seed_{seed}" / f"{split}_predictions.csv")
        required = {"row_index", "label", "learner_id", "probability"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"seed {seed} {split} predictions are missing columns: {missing}")
        if frame["row_index"].duplicated().any():
            raise ValueError(f"seed {seed} {split} predictions contain duplicate row_index values")
        frame = frame.sort_values("row_index").reset_index(drop=True)
        if accumulated is None:
            accumulated = frame[["row_index", "label", "learner_id"]].copy()
            accumulated["probability"] = frame["probability"].to_numpy(dtype=float)
        else:
            if not accumulated["row_index"].equals(frame["row_index"]):
                raise ValueError(f"seed {seed} {split} rows do not align with the first seed")
            if not accumulated["label"].equals(frame["label"]):
                raise ValueError(f"seed {seed} {split} labels do not align with the first seed")
            if not accumulated["learner_id"].equals(frame["learner_id"]):
                raise ValueError(f"seed {seed} {split} learners do not align with the first seed")
            accumulated["probability"] += frame["probability"].to_numpy(dtype=float)
    accumulated["probability"] /= len(chosen)
    return accumulated, chosen


def _attach_extra_arm(
    root: Path, frame: pd.DataFrame, arm: str, split: str, seeds: Sequence[int] | None
) -> pd.DataFrame:
    """Join one optional arm onto an already-joined frame, or refuse to.

    The refusal cases are the point. Most saved runs in ``results/raw_runs``
    are on the chronological re-split, which has a different row count and, for
    the ones that do carry ``row_index``, rows drawn from both the official dev
    and the official test partition. Joining such a run would mix partitions,
    so this only accepts arms that align exactly, row for row and label for
    label, with the official frame already in hand.
    """
    if arm not in OPTIONAL_ARM_SOURCES:
        raise KeyError(f"unknown optional arm '{arm}'; known: {sorted(OPTIONAL_ARM_SOURCES)}")
    base = root / "results" / "raw_runs" / OPTIONAL_ARM_SOURCES[arm]
    extra, _ = _load_seed_averaged_arm(base, split, seeds)
    merged = frame.merge(
        extra.rename(columns={"probability": f"probability_{arm}", "label": f"label_{arm}"})[
            ["row_index", f"probability_{arm}", f"label_{arm}"]
        ],
        on="row_index",
        how="inner",
    )
    if len(merged) != len(frame) or len(merged) != len(extra):
        raise ValueError(
            f"{split}: arm '{arm}' does not align on row_index "
            f"({len(merged)} joined against {len(frame)} official rows and {len(extra)} arm rows); "
            "this usually means the arm is on the chronological split, not the official one"
        )
    if not (merged["label"] == merged[f"label_{arm}"]).all():
        raise ValueError(f"{split}: arm '{arm}' disagrees on labels after joining")
    return merged.drop(columns=[f"label_{arm}"])


def load_official_arms(
    root: Path,
    seed: int = 42,
    neural_seeds: Sequence[int] | None = None,
    extra_arms: Sequence[str] = (),
) -> tuple[dict[str, pd.DataFrame], list[int]]:
    """Join the neural and boosted arms on ``row_index`` for both splits.

    ``extra_arms`` names further arms from :data:`OPTIONAL_ARM_SOURCES`; each is
    joined under the same alignment and label checks and the call fails rather
    than silently dropping rows if an arm is on a different partition.
    """
    neural_dir = root / "results" / "raw_runs" / "official_eip_ktformer"
    boosted_dir = root / "results" / "raw_runs" / "official_gbdt_benchmark"
    frames: dict[str, pd.DataFrame] = {}
    seeds_used: list[int] = []
    for split in ("dev", "test"):
        neural, split_seeds = _load_seed_averaged_arm(neural_dir, split, neural_seeds)
        if seeds_used and split_seeds != seeds_used:
            raise ValueError(
                "official neural dev/test predictions do not contain the same seeds: "
                f"dev={seeds_used}, test={split_seeds}"
            )
        seeds_used = split_seeds
        boosted = pd.read_csv(boosted_dir / f"{split}_predictions.csv")
        if boosted["row_index"].duplicated().any():
            raise ValueError(f"{split}: boosted arm contains duplicate row_index values")
        merged = neural.merge(
            boosted,
            on="row_index",
            suffixes=("_neural", "_boosted"),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(neural) or len(merged) != len(boosted):
            raise ValueError(f"{split}: arms do not align on row_index ({len(merged)} joined)")
        if not (merged["label_neural"] == merged["label_boosted"]).all():
            raise ValueError(f"{split}: arms disagree on labels after joining")
        joined = merged.rename(columns={"label_neural": "label", "learner_id_neural": "learner_id"})[
            ["row_index", "label", "learner_id", "probability_neural", "probability_boosted"]
        ]
        for arm in extra_arms:
            joined = _attach_extra_arm(root, joined, arm, split, neural_seeds)
        frames[split] = joined
    shared = set(frames["dev"]["row_index"]).intersection(frames["test"]["row_index"])
    if shared:
        raise ValueError(f"dev and test share {len(shared)} rows; the split is not disjoint")
    return frames, seeds_used


def arm_correlation_matrix(
    frame: pd.DataFrame, arms: Sequence[str], space: str = "logit"
) -> pd.DataFrame:
    """Pairwise Pearson correlation between arms, in log-odds space by default.

    Log-odds rather than probabilities: two arms that agree on ranking but
    differ in calibration look less correlated on the probability scale purely
    because of the squashing, and it is ranking agreement that determines
    whether combining them can help AUROC.
    """
    if space not in {"logit", "probability"}:
        raise ValueError("space must be 'logit' or 'probability'")
    transform = _logit if space == "logit" else (lambda v: np.asarray(v, dtype=float))
    values = {arm: transform(frame[f"probability_{arm}"].to_numpy()) for arm in arms}
    return pd.DataFrame(values).corr()


def mean_pairwise_correlation(correlations: pd.DataFrame, arms: Sequence[str]) -> float:
    """Mean off-diagonal correlation among ``arms``.

    A single arm has no pair, and is reported as 1.0 rather than 0.0: a lone
    arm is maximally redundant with itself, and scoring it as perfectly diverse
    would let a diversity-rewarding criterion pick a single model and call that
    an ensemble.
    """
    arms = tuple(arms)
    if len(arms) < 2:
        return 1.0
    pairs = [abs(float(correlations.loc[a, b])) for a, b in itertools.combinations(arms, 2)]
    return float(np.mean(pairs))


def cross_validate(
    combiners: Sequence[Combiner], dev: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Score every combiner by learner-grouped CV inside dev."""
    y = dev["label"].to_numpy()
    groups = dev["learner_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    for combiner in combiners:
        fold_scores: list[float] = []
        for fit_index, score_index in splitter.split(dev, y, groups):
            state = combiner.fit(dev.iloc[fit_index], y[fit_index])
            predicted = combiner.apply(state, dev.iloc[score_index])
            fold_scores.append(_auroc(y[score_index], predicted))
        scores = np.asarray(fold_scores, dtype=float)
        rows.append(
            {
                "combiner": combiner.name,
                "description": combiner.description,
                "arms": "+".join(combiner.arms),
                "cv_auroc_mean": float(np.nanmean(scores)),
                "cv_auroc_sd": float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "folds": int(np.sum(~np.isnan(scores))),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_auroc_mean", ascending=False).reset_index(drop=True)


def grouped_oof_predictions(
    combiner: Combiner, dev: pd.DataFrame, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions for one combiner, plus each row's holdout fold.

    Every row's prediction comes from a fit that excluded that row's *learner*,
    not merely that row. This is the stacking-safe quantity: a meta-learner
    trained on ordinary in-sample predictions would learn how much to trust an
    arm on learners it had already memorised, and would look far better on dev
    than it can be on test. Returned alongside the fold labels so a caller can
    verify the disjointness rather than trust it.
    """
    y = dev["label"].to_numpy()
    groups = dev["learner_id"].to_numpy()
    predictions = np.full(len(dev), np.nan, dtype=float)
    fold_of_row = np.full(len(dev), -1, dtype=int)
    for fold, (fit_index, score_index) in enumerate(
        GroupKFold(n_splits=n_splits).split(dev, y, groups)
    ):
        state = combiner.fit(dev.iloc[fit_index], y[fit_index])
        predictions[score_index] = combiner.apply(state, dev.iloc[score_index])
        fold_of_row[score_index] = fold
    if np.isnan(predictions).any() or (fold_of_row < 0).any():
        raise RuntimeError("grouped OOF left rows unscored; folds did not cover the partition")
    return predictions, fold_of_row


def select_combiner(
    cv_table: pd.DataFrame,
    correlations: pd.DataFrame | None = None,
    criterion: str = "cv_auroc",
    correlation_penalty: float = 0.0,
    exclude: Sequence[str] = (),
) -> tuple[str, pd.DataFrame]:
    """Pick one combiner from a dev-CV table, never from test.

    ``criterion='cv_auroc'`` is the published rule: highest grouped-CV AUROC
    wins. ``criterion='diversity_penalised'`` subtracts
    ``correlation_penalty`` times the mean pairwise correlation of the arms a
    combiner consumes, which is the operational form of "reward
    complementarity": at a large enough penalty a slightly weaker combiner
    built on decorrelated arms outranks a stronger one built on near-duplicates.
    Both rules look only at dev.
    """
    if criterion not in {"cv_auroc", "diversity_penalised"}:
        raise ValueError("criterion must be 'cv_auroc' or 'diversity_penalised'")
    if criterion == "diversity_penalised" and correlations is None:
        raise ValueError("diversity_penalised selection needs an arm correlation matrix")
    scored = cv_table.copy()
    if correlations is not None:
        scored["mean_arm_correlation"] = [
            mean_pairwise_correlation(correlations, str(arms).split("+"))
            for arms in scored["arms"]
        ]
    else:
        scored["mean_arm_correlation"] = np.nan
    if criterion == "cv_auroc":
        scored["selection_score"] = scored["cv_auroc_mean"]
    else:
        scored["selection_score"] = (
            scored["cv_auroc_mean"] - correlation_penalty * scored["mean_arm_correlation"]
        )
    eligible = scored[~scored["combiner"].isin(set(exclude))]
    if eligible.empty:
        raise ValueError("every candidate was excluded from selection")
    ordered = eligible.sort_values("selection_score", ascending=False, kind="mergesort")
    return str(ordered.iloc[0]["combiner"]), scored


def score_on_test(
    combiners: Sequence[Combiner],
    dev: pd.DataFrame,
    test: pd.DataFrame,
    selected_name: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Refit every candidate on all of dev, then read test exactly once.

    ``selected_name`` is an input, not an output: the winner must already be
    fixed before this function runs. Scoring the losers too is reporting, not
    selection - nothing downstream may re-rank on ``test_auroc``.
    """
    y_dev, y_test = dev["label"].to_numpy(), test["label"].to_numpy()
    if selected_name not in {combiner.name for combiner in combiners}:
        raise ValueError(f"selected combiner '{selected_name}' is not in the candidate family")
    test_rows: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    for combiner in combiners:
        state = combiner.fit(dev, y_dev)
        predicted = combiner.apply(state, test)
        test_predictions[combiner.name] = predicted
        metrics = binary_classification_metrics(y_test, predicted)
        fitted: dict[str, Any] = {}
        if "weight" in state:
            fitted["neural_weight"] = state["weight"]
        if "weights" in state and "weight" not in state:
            fitted["arm_weights"] = [float(v) for v in state["weights"]]
        if "model" in state and hasattr(state["model"], "coef_"):
            fitted["coefficients"] = [float(v) for v in state["model"].coef_.ravel()]
            fitted["intercept"] = float(state["model"].intercept_[0])
        test_rows.append(
            {
                "combiner": combiner.name,
                "description": combiner.description,
                "selected": combiner.name == selected_name,
                "test_auroc": metrics["auroc"],
                "test_auprc": metrics["auprc"],
                "test_brier": metrics["brier_score"],
                "test_ece": metrics["ece"],
                "fitted": fitted,
            }
        )
    return test_rows, test_predictions


def run_ensemble_study(root: Path, seed: int = 42, n_bootstrap: int = 2000) -> dict[str, Any]:
    """Select a combiner on dev, then score every candidate once on test."""
    frames, neural_seeds = load_official_arms(root, seed=seed)
    dev, test = frames["dev"], frames["test"]
    combiners = build_combiners()
    cv_table = cross_validate(combiners, dev, seed=seed)

    reference = {f"{arm}_only" for arm in DEFAULT_ARMS}
    selected_name, _ = select_combiner(cv_table, exclude=sorted(reference))

    test_rows, test_predictions = score_on_test(combiners, dev, test, selected_name)
    y_test = test["label"].to_numpy()
    test_table = pd.DataFrame(test_rows)

    # The honest headline: selected combiner against the stronger single arm.
    single = test_table[test_table["combiner"].isin(reference)]
    strongest_arm = str(single.loc[single["test_auroc"].idxmax(), "combiner"])
    comparison = learner_clustered_bootstrap(
        y_test,
        test_predictions[selected_name],
        test_predictions[strongest_arm],
        test["learner_id"].to_numpy(),
        metric="auroc",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    payload = {
        "protocol": "official SLAM train/dev/test",
        "selection": {
            "method": "learner-grouped 5-fold CV on the dev partition",
            "selected_combiner": selected_name,
            "test_partition_reads": 1,
            "note": "Reference arms are excluded from selection; they are scored for comparison only.",
        },
        "arms": {
            "neural": "results/raw_runs/official_eip_ktformer",
            "neural_seeds_averaged": neural_seeds,
            "boosted": "results/raw_runs/official_gbdt_benchmark",
            "probability_correlation_test": float(
                np.corrcoef(test["probability_neural"], test["probability_boosted"])[0, 1]
            ),
        },
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "n_learners_test": int(test["learner_id"].nunique()),
        "cross_validation": cv_table.to_dict(orient="records"),
        "test_results": test_rows,
        "headline": {
            "selected_combiner": selected_name,
            "strongest_single_arm": strongest_arm,
            "auroc_difference": comparison["model_a_minus_b"],
            "ci_lower": comparison["ci_lower"],
            "ci_upper": comparison["ci_upper"],
            "raw_p_value": comparison["raw_p_value"],
            "cluster_count": comparison["cluster_count"],
        },
    }

    output_dir = root / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ensemble_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cv_table.to_csv(metrics_dir / "ensemble_cross_validation.csv", index=False)
    test_table.drop(columns=["fitted"]).to_csv(metrics_dir / "ensemble_results.csv", index=False)
    frame = test[["row_index", "label", "learner_id"]].copy()
    frame["probability"] = test_predictions[selected_name]
    frame["probability_neural"] = test["probability_neural"].to_numpy()
    frame["probability_boosted"] = test["probability_boosted"].to_numpy()
    frame.to_csv(output_dir / "ensemble_test_predictions.csv", index=False)
    return payload


def run_diversity_ensemble_study(
    root: Path,
    arms: Sequence[str] = ("neural", "boosted", "bare"),
    criterion: str = "cv_auroc",
    correlation_penalty: float = 0.0,
    seed: int = 42,
    n_bootstrap: int = 2000,
    n_splits: int = 5,
    output_name: str = "ensemble_diversity_study.json",
) -> dict[str, Any]:
    """Diversity-oriented variant: more arms, subset search, richer stackers.

    Same protocol as :func:`run_ensemble_study` - candidates ranked by
    learner-grouped CV on dev, winner frozen, test read once - but the
    candidate family spans every arm *subset* and the selection rule can be
    switched to one that penalises correlated arms. It writes to its own output
    file so that running it never disturbs the published study.
    """
    arms = tuple(arms)
    extra = tuple(arm for arm in arms if arm in OPTIONAL_ARM_SOURCES)
    unknown = [arm for arm in arms if arm not in DEFAULT_ARMS and arm not in OPTIONAL_ARM_SOURCES]
    if unknown:
        raise KeyError(f"arms {unknown} have no official-partition source")
    frames, neural_seeds = load_official_arms(root, seed=seed, extra_arms=extra)
    # Meta-features are derived inside each partition, never across them.
    dev = add_meta_features(frames["dev"])
    test = add_meta_features(frames["test"])

    combiners = build_diversity_combiners(arms, seed=seed)
    cv_table = cross_validate(combiners, dev, n_splits=n_splits, seed=seed)
    dev_correlations = arm_correlation_matrix(dev, arms)

    reference = sorted(f"{arm}_only" for arm in arms)
    selected_name, selection_table = select_combiner(
        cv_table,
        correlations=dev_correlations,
        criterion=criterion,
        correlation_penalty=correlation_penalty,
        exclude=reference,
    )

    # Pooled out-of-fold AUROC for the winner, still on dev: the fold-mean above
    # is an average of five smaller samples, this is one score over the whole
    # partition, and both are computed from predictions made for learners the
    # combiner had never been fitted on.
    selected = next(c for c in combiners if c.name == selected_name)
    oof_predictions, _ = grouped_oof_predictions(selected, dev, n_splits=n_splits)
    oof_auroc = _auroc(dev["label"].to_numpy(), oof_predictions)

    test_rows, test_predictions = score_on_test(combiners, dev, test, selected_name)
    y_test = test["label"].to_numpy()
    test_table = pd.DataFrame(test_rows)
    single = test_table[test_table["combiner"].isin(reference)]
    strongest_arm = str(single.loc[single["test_auroc"].idxmax(), "combiner"])
    comparison = learner_clustered_bootstrap(
        y_test,
        test_predictions[selected_name],
        test_predictions[strongest_arm],
        test["learner_id"].to_numpy(),
        metric="auroc",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    payload = {
        "protocol": "official SLAM train/dev/test",
        "selection": {
            "method": f"learner-grouped {n_splits}-fold CV on the dev partition",
            "criterion": criterion,
            "correlation_penalty": correlation_penalty,
            "selected_combiner": selected_name,
            "selected_dev_oof_auroc": oof_auroc,
            "test_partition_reads": 1,
            "note": (
                "Arm subsets are candidates in the same CV ranking, so subset choice "
                "is part of one dev-only selection rather than a second pass."
            ),
        },
        "arms": {
            "used": list(arms),
            "sources": {
                "neural": "results/raw_runs/official_eip_ktformer",
                "boosted": "results/raw_runs/official_gbdt_benchmark",
                **{
                    arm: f"results/raw_runs/{OPTIONAL_ARM_SOURCES[arm]}"
                    for arm in extra
                },
            },
            "neural_seeds_averaged": neural_seeds,
            "dev_logit_correlation": dev_correlations.round(6).to_dict(),
            "test_logit_correlation": arm_correlation_matrix(test, arms).round(6).to_dict(),
            "test_probability_correlation": (
                arm_correlation_matrix(test, arms, space="probability").round(6).to_dict()
            ),
        },
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "n_learners_test": int(test["learner_id"].nunique()),
        "cross_validation": selection_table.to_dict(orient="records"),
        "test_results": test_rows,
        "headline": {
            "selected_combiner": selected_name,
            "strongest_single_arm": strongest_arm,
            "auroc_difference": comparison["model_a_minus_b"],
            "ci_lower": comparison["ci_lower"],
            "ci_upper": comparison["ci_upper"],
            "raw_p_value": comparison["raw_p_value"],
            "cluster_count": comparison["cluster_count"],
        },
    }

    output_dir = root / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / output_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stem = Path(output_name).stem
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selection_table.to_csv(metrics_dir / f"{stem}_cross_validation.csv", index=False)
    test_table.drop(columns=["fitted"]).to_csv(metrics_dir / f"{stem}_results.csv", index=False)
    return payload
