"""N2 - measuring personalization instead of asserting it.

Knowledge-tracing papers routinely describe their models as personalized on the
strength of having a learner embedding in the diagram. This module tests the
claim directly by intervening on learner identity while holding the entire
interaction sequence fixed:

* **permuted identity** - each sequence keeps its behaviour but is attributed to a
  different learner drawn from the same batch;
* **anonymous identity** - identity is replaced by the padding index, the state an
  unseen learner is in.

If the learner embedding is doing real work, both interventions move predictions
and degrade discrimination. If it is decorative, they do not. The comparison is
paired at token level, which makes it far more sensitive than an ablation that
retrains the whole model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.inference import scored_positions


@torch.inference_mode()
def identity_interventions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired predictions under true, permuted, and anonymous learner identity."""

    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames: list[pd.DataFrame] = []
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        valid = scored_positions(moved)
        observed = torch.sigmoid(model(moved, return_diagnostics=False)["logits"])

        learner = moved["learner"]
        batch_size = learner.shape[0]
        if batch_size > 1:
            # Derangement-style shift guarantees no sequence keeps its own identity.
            shift = int(torch.randint(1, batch_size, (1,), generator=generator).item())
            permuted_learner = torch.roll(learner, shifts=shift, dims=0)
        else:
            permuted_learner = torch.zeros_like(learner)
        permuted_batch = dict(moved)
        permuted_batch["learner"] = permuted_learner
        permuted = torch.sigmoid(model(permuted_batch, return_diagnostics=False)["logits"])

        anonymous_batch = dict(moved)
        anonymous_batch["learner"] = torch.zeros_like(learner)
        anonymous = torch.sigmoid(model(anonymous_batch, return_diagnostics=False)["logits"])

        frames.append(
            pd.DataFrame(
                {
                    "row_index": moved["row_index"][valid].cpu().numpy(),
                    "label": moved["responses"][valid].cpu().numpy(),
                    "learner_id": learner[valid].cpu().numpy(),
                    "probability": observed[valid].float().cpu().numpy(),
                    "probability_permuted_identity": permuted[valid].float().cpu().numpy(),
                    "probability_anonymous_identity": anonymous[valid].float().cpu().numpy(),
                }
            )
        )
    collected = pd.concat(frames, ignore_index=True)
    return collected[collected["row_index"] >= 0].reset_index(drop=True)


def summarize_interventions(frame: pd.DataFrame) -> dict[str, Any]:
    """Effect sizes and discrimination loss for each identity intervention."""

    def _describe(condition: str) -> dict[str, Any]:
        delta = frame[condition] - frame["probability"]
        absolute = delta.abs()
        return {
            "mean_absolute_delta": float(absolute.mean()),
            "median_absolute_delta": float(absolute.median()),
            "p90_absolute_delta": float(absolute.quantile(0.90)),
            "max_absolute_delta": float(absolute.max()),
            "auroc": float(roc_auc_score(frame["label"], frame[condition])),
            "auprc": float(average_precision_score(frame["label"], frame[condition])),
            "share_moved_over_0_05": float((absolute > 0.05).mean()),
        }

    observed_auroc = float(roc_auc_score(frame["label"], frame["probability"]))
    observed_auprc = float(average_precision_score(frame["label"], frame["probability"]))
    summary = {
        "observed": {"auroc": observed_auroc, "auprc": observed_auprc, "n": int(len(frame))},
        "permuted_identity": _describe("probability_permuted_identity"),
        "anonymous_identity": _describe("probability_anonymous_identity"),
    }
    for condition in ("permuted_identity", "anonymous_identity"):
        summary[condition]["auroc_loss"] = observed_auroc - summary[condition]["auroc"]
        summary[condition]["auprc_loss"] = observed_auprc - summary[condition]["auprc"]
    summary["interpretation"] = (
        "A near-zero mean absolute delta together with a near-zero AUROC loss would "
        "indicate that learner identity is not used by the trained model, regardless of "
        "the embedding being present in the architecture."
    )
    return summary


def sensitivity_by_history(
    frame: pd.DataFrame,
    *,
    history_column: str = "prior_interactions",
    edges: Iterable[float] = (0, 1, 3, 5, 10, 20, 50, 100, 200, np.inf),
    min_bin_count: int = 200,
) -> pd.DataFrame:
    """Does identity matter more once a learner has accumulated a track record?"""

    if history_column not in frame.columns:
        raise KeyError(f"Missing history column {history_column!r}")
    bins = pd.cut(frame[history_column], bins=list(edges), right=False, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for interval, group in frame.groupby(bins, observed=True):
        if len(group) < min_bin_count:
            continue
        permuted = (group["probability_permuted_identity"] - group["probability"]).abs()
        anonymous = (group["probability_anonymous_identity"] - group["probability"]).abs()
        rows.append(
            {
                "history_lower": float(interval.left),
                "history_upper": float(interval.right),
                "n": int(len(group)),
                "mean_absolute_delta_permuted": float(permuted.mean()),
                "mean_absolute_delta_anonymous": float(anonymous.mean()),
                "auroc": float(roc_auc_score(group["label"], group["probability"]))
                if group["label"].nunique() == 2 else float("nan"),
            }
        )
    if not rows:
        raise ValueError("No history bin met the minimum count requirement")
    return pd.DataFrame(rows)


def run(
    root: Path,
    *,
    seed: int = 42,
    split: str = "test",
    device: str | None = None,
    sequence_length: int = 200,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Execute N2 against the frozen proposed checkpoint."""

    from src.analysis.inference import indexed_dataloader, join_corpus_attributes, load_proposed_model

    output_dir = root / "results" / "analysis" / "personalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, resolved_device, provenance = load_proposed_model(
        root, seed=seed, sequence_length=sequence_length, device=device
    )
    loader, frame = indexed_dataloader(
        root, split=split, sequence_length=sequence_length, batch_size=batch_size
    )
    interventions = identity_interventions(model, loader, resolved_device, seed=seed)
    enriched = join_corpus_attributes(interventions, frame, ["prior_interactions", "prior_error_rate"])
    enriched.to_csv(output_dir / f"{split}_identity_interventions.csv.gz", index=False, compression="gzip")

    summary = summarize_interventions(enriched)
    by_history = sensitivity_by_history(enriched)
    by_history.to_csv(output_dir / "identity_sensitivity_by_history.csv", index=False)

    payload = {
        "provenance": provenance,
        "partition": split,
        "intervention_summary": summary,
        "sensitivity_by_history": by_history.to_dict(orient="records"),
        "design_note": (
            "Interventions are applied at inference on a fixed checkpoint, so the sequence, "
            "linguistic content, and temporal structure are held exactly constant and only "
            "learner identity varies."
        ),
    }
    (output_dir / "personalization_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
