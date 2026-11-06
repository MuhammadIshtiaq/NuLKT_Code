"""N1 - recovering memory decay from data and from learned temporal attention.

Two independent strands are kept deliberately separate:

* **Learned kernel.** ``TemporalCausalSelfAttention`` biases attention scores by
  ``-softplus(log_decay) * log1p(dt_hours) ** softplus(log_power)``. Reading those
  parameters back describes the timescales the model *chose*, per head.
* **Empirical retention.** Observed accuracy as a function of the lag since a
  learner last met the same token, fitted with competing forgetting laws.

The second strand is the evidence for any claim about how learners forget. The
architecture's kernel lives in the power-law family by construction, so parameter
values alone could never adjudicate exponential versus power-law decay; that
comparison is made against data, and the learned kernel is then checked for
consistency with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.nulkt import NuLKT

# Retention laws compared on the empirical curve. Each maps lag in hours to a
# predicted accuracy, with an asymptote so curves need not decay to zero.
FORGETTING_LAWS = {
    "exponential": lambda t, a, b, c: c + a * np.exp(-b * t),
    "power_law": lambda t, a, b, c: c + a * np.power(1.0 + t, -b),
    "logarithmic": lambda t, a, b, c: c + a / (1.0 + b * np.log1p(t)),
}


# Initialization values of the temporal-attention parameters. A head still sitting
# on these after training received no usable gradient and must not be reported as
# a learned timescale.
INITIAL_LOG_DECAY = -2.0
INITIAL_LOG_POWER = 0.0
INITIALIZATION_TOLERANCE = 1e-4


def learned_temporal_kernel(model: NuLKT) -> pd.DataFrame:
    """Extract per-head decay rate, exponent, and implied attention half-life.

    Heads whose parameters remain at initialization are flagged. Within a session
    the elapsed gaps are seconds to minutes, so ``log1p(dt_hours)`` is
    approximately zero and the recency bias vanishes regardless of the parameter
    values; the local blocks' timescales are therefore not identifiable from data
    and their nominal half-lives are meaningless.
    """

    initial_decay = float(F.softplus(torch.tensor(INITIAL_LOG_DECAY)))
    initial_power = float(F.softplus(torch.tensor(INITIAL_LOG_POWER)))
    rows: list[dict[str, Any]] = []
    groups = (("local", model.local_blocks), ("global", model.global_blocks))
    for level, blocks in groups:
        for layer_index, block in enumerate(blocks):
            attention = block.attention
            decay = F.softplus(attention.log_decay.detach()).cpu().numpy()
            power = F.softplus(attention.log_power.detach()).cpu().numpy()
            for head in range(len(decay)):
                lam, gamma = float(decay[head]), float(power[head])
                # exp(-lam * log1p(t)^gamma) = 1/2  =>  t = exp((ln2/lam)^(1/gamma)) - 1
                half_life = float("inf")
                if lam > 1e-8 and gamma > 1e-8:
                    exponent = (np.log(2.0) / lam) ** (1.0 / gamma)
                    half_life = float(np.expm1(np.clip(exponent, 0.0, 700.0)))
                at_initialization = bool(
                    abs(lam - initial_decay) < INITIALIZATION_TOLERANCE
                    and abs(gamma - initial_power) < INITIALIZATION_TOLERANCE
                )
                rows.append(
                    {
                        "level": level,
                        "layer": layer_index,
                        "head": head,
                        "decay_lambda": lam,
                        "power_gamma": gamma,
                        "attention_half_life_hours": half_life,
                        "trainable": bool(attention.log_decay.requires_grad),
                        "at_initialization": at_initialization,
                        "identifiable": not at_initialization,
                    }
                )
    return pd.DataFrame(rows)


def token_revisit_lags(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag in hours since the learner previously met the same surface token.

    Uses only earlier occurrences, so the feature is past-only and safe to
    condition on. Rows for a learner's first encounter with a token are dropped.
    """

    required = {"user_id", "token_id", "timestamp_days", "row_order", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for revisit analysis: {sorted(missing)}")
    ordered = frame.loc[:, sorted(required)].sort_values(
        ["user_id", "token_id", "timestamp_days", "row_order"], kind="stable"
    )
    grouped = ordered.groupby(["user_id", "token_id"], sort=False, observed=True)
    previous_time = grouped["timestamp_days"].shift(1)
    exposure_index = grouped.cumcount()
    lag_hours = (ordered["timestamp_days"] - previous_time) * 24.0
    result = pd.DataFrame(
        {
            "lag_hours": lag_hours,
            "prior_exposures": exposure_index,
            "label": ordered["label"].to_numpy(),
            "correct": 1 - ordered["label"].to_numpy(),
        },
        index=ordered.index,
    )
    return result[result["prior_exposures"] >= 1].dropna(subset=["lag_hours"])


def empirical_retention_curve(
    revisits: pd.DataFrame,
    *,
    n_bins: int = 24,
    min_bin_count: int = 200,
) -> pd.DataFrame:
    """Observed accuracy against revisit lag, on log-spaced bins."""

    usable = revisits[revisits["lag_hours"] > 0]
    if usable.empty:
        raise ValueError("No positive revisit lags are available")
    edges = np.unique(
        np.geomspace(
            max(usable["lag_hours"].min(), 1e-3),
            usable["lag_hours"].max(),
            n_bins + 1,
        )
    )
    assigned = pd.cut(usable["lag_hours"], bins=edges, include_lowest=True)
    grouped = usable.groupby(assigned, observed=True)
    curve = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_lag_hours=("lag_hours", "mean"),
        median_lag_hours=("lag_hours", "median"),
    ).reset_index(drop=True)
    curve["standard_error"] = np.sqrt(
        (curve["accuracy"] * (1 - curve["accuracy"]) / curve["n"]).clip(lower=0)
    )
    return curve[curve["n"] >= min_bin_count].reset_index(drop=True)


def compare_forgetting_laws(curve: pd.DataFrame) -> dict[str, Any]:
    """Fit competing retention laws by weighted least squares and rank them.

    Weights are inverse binned variance, so densely observed lags dominate.
    Model ranking uses AIC/BIC on the weighted residual sum of squares plus a
    leave-one-bin-out cross-validated RMSE, which is less forgiving of curves
    that fit only through their asymptote.
    """

    from scipy.optimize import curve_fit

    if len(curve) < 5:
        raise ValueError("At least five populated lag bins are required")
    t = curve["mean_lag_hours"].to_numpy(dtype=float)
    y = curve["accuracy"].to_numpy(dtype=float)
    weights = curve["n"].to_numpy(dtype=float)
    sigma = 1.0 / np.sqrt(weights)
    n = len(t)

    def _fit(name: str, subset_t: np.ndarray, subset_y: np.ndarray, subset_sigma: np.ndarray):
        law = FORGETTING_LAWS[name]
        initial = [max(subset_y.max() - subset_y.min(), 1e-3), 0.1, float(subset_y.min())]
        bounds = ([0.0, 1e-6, 0.0], [1.0, 50.0, 1.0])
        parameters, _ = curve_fit(
            law, subset_t, subset_y, p0=initial, bounds=bounds, sigma=subset_sigma,
            maxfev=200_000,
        )
        return parameters

    results: dict[str, Any] = {}
    for name, law in FORGETTING_LAWS.items():
        try:
            parameters = _fit(name, t, y, sigma)
        except (RuntimeError, ValueError) as exc:
            results[name] = {"status": "fit_failed", "reason": str(exc)}
            continue
        predicted = law(t, *parameters)
        residual = np.sum(weights * (y - predicted) ** 2)
        k = len(parameters)
        aic = n * np.log(max(residual / n, 1e-300)) + 2 * k
        bic = n * np.log(max(residual / n, 1e-300)) + k * np.log(n)

        errors: list[float] = []
        for held in range(n):
            keep = np.arange(n) != held
            try:
                fold = _fit(name, t[keep], y[keep], sigma[keep])
            except (RuntimeError, ValueError):
                continue
            errors.append(float(law(t[held], *fold) - y[held]))
        cv_rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")

        results[name] = {
            "status": "fitted",
            "parameters": {"amplitude": float(parameters[0]), "rate": float(parameters[1]), "asymptote": float(parameters[2])},
            "weighted_residual_sum_of_squares": float(residual),
            "aic": float(aic),
            "bic": float(bic),
            "leave_one_bin_out_rmse": cv_rmse,
        }

    fitted = {name: value for name, value in results.items() if value.get("status") == "fitted"}
    ranking = sorted(fitted, key=lambda name: fitted[name]["aic"]) if fitted else []
    best = ranking[0] if ranking else None
    delta_aic = None
    if len(ranking) >= 2:
        delta_aic = float(fitted[ranking[1]]["aic"] - fitted[ranking[0]]["aic"])
    return {
        "laws": results,
        "ranking_by_aic": ranking,
        "best_law": best,
        "delta_aic_to_runner_up": delta_aic,
        "interpretation_note": (
            "A delta AIC below 2 does not distinguish the laws. The model's own temporal "
            "kernel is power-law-family by construction and therefore cannot be used as "
            "independent evidence for this comparison."
        ),
    }


def run(
    root: Path,
    *,
    seed: int = 42,
    device: str | None = None,
    sequence_length: int = 200,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Execute N1 and write machine-readable evidence."""

    from src.analysis.inference import load_proposed_model
    from src.data.torch_dataset import load_processed_frame

    output_dir = root / "results" / "analysis" / "forgetting"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    kernel = learned_temporal_kernel(model)
    kernel.to_csv(output_dir / "learned_temporal_kernel.csv", index=False)

    frame, _ = load_processed_frame(root, row_limit=max_rows)
    # Retention is estimated on training-partition behaviour only, so the curve
    # is never fitted on evaluation outcomes.
    train = frame[frame["chronological_split"].astype(str) == "train"]
    revisits = token_revisit_lags(train)
    curve = empirical_retention_curve(revisits)
    curve.to_csv(output_dir / "empirical_retention_curve.csv", index=False)
    comparison = compare_forgetting_laws(curve)

    identifiable = kernel[kernel["identifiable"] & np.isfinite(kernel["attention_half_life_hours"])]
    unidentifiable = kernel[~kernel["identifiable"]]
    spread = (
        float(identifiable["attention_half_life_hours"].max() / max(identifiable["attention_half_life_hours"].min(), 1e-9))
        if len(identifiable) else None
    )
    payload = {
        "provenance": provenance,
        "learned_kernel": {
            "heads_total": int(len(kernel)),
            "heads_identifiable": int(len(identifiable)),
            "heads_at_initialization": int(len(unidentifiable)),
            "levels_at_initialization": sorted(set(unidentifiable["level"])) if len(unidentifiable) else [],
            "decay_lambda_median": float(identifiable["decay_lambda"].median()) if len(identifiable) else None,
            "power_gamma_median": float(identifiable["power_gamma"].median()) if len(identifiable) else None,
            "power_gamma_min": float(identifiable["power_gamma"].min()) if len(identifiable) else None,
            "power_gamma_max": float(identifiable["power_gamma"].max()) if len(identifiable) else None,
            "half_life_hours_median": float(identifiable["attention_half_life_hours"].median()) if len(identifiable) else None,
            "half_life_hours_min": float(identifiable["attention_half_life_hours"].min()) if len(identifiable) else None,
            "half_life_hours_max": float(identifiable["attention_half_life_hours"].max()) if len(identifiable) else None,
            "half_life_spread_ratio": spread,
            "multi_timescale": bool(spread is not None and spread > 10),
            "identifiability_note": (
                "Local (within-session) heads span gaps of seconds to minutes, so log1p(dt_hours) "
                "is approximately zero and the recency bias is flat in the parameters. Heads left "
                "at initialization are excluded from every reported timescale; their nominal "
                "half-lives are artifacts of the initial values, not learned quantities."
            ),
        },
        "empirical_retention": {
            "revisit_observations": int(len(revisits)),
            "populated_bins": int(len(curve)),
            "accuracy_at_shortest_lag": float(curve["accuracy"].iloc[0]),
            "accuracy_at_longest_lag": float(curve["accuracy"].iloc[-1]),
            "partition": "chronological train",
        },
        "law_comparison": comparison,
    }
    (output_dir / "forgetting_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
