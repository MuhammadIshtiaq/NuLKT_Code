"""Row-aligned diagnostic inference over a trained checkpoint.

The production dataset yields padded model tensors only. Scientific analysis
additionally needs to join each predicted token back to its corpus row so that
linguistic, temporal, and demographic attributes can condition the analysis.
``IndexedSequenceDataset`` adds that join key without altering model inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.torch_dataset import (
    EncodedSequenceDataset,
    load_processed_frame,
    vocab_sizes_for_model,
)
from src.models.nulkt import NuLKT, NuLKTConfig

# Kept identical to ``scripts.research_stages._model_settings['nulkt']``
# so a checkpoint is rebuilt with exactly the architecture that produced it.
PROPOSED_MODEL_SETTINGS: dict[str, Any] = {
    "hidden_dim": 128,
    "dropout": 0.2,
    "embedding_dim": 64,
    "num_heads": 4,
    "local_layers": 2,
    "global_layers": 2,
    "num_experts": 5,
    "time_embedding_dim": 16,
    "learner_embedding_dim": 32,
    "contrastive_max_samples": 2048,
    "within_exercise_bidirectional": True,
}

SCALAR_OUTPUTS = (
    "probability",
    "mastery",
    "uncertainty",
    "confidence",
    "forgetting_signal",
    "alpha",
    "beta",
    "routing_entropy",
)


class IndexedSequenceDataset(EncodedSequenceDataset):
    """``EncodedSequenceDataset`` that also emits each token's corpus row index."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(index)
        window = self.windows[index]
        # -1 marks padded positions so a real row index of 0 stays unambiguous.
        row_index = torch.full((self.max_length,), -1, dtype=torch.long)
        count = min(len(window.row_indices), self.max_length)
        row_index[:count] = torch.as_tensor(
            np.asarray(window.row_indices[:count], dtype=np.int64), dtype=torch.long
        )
        batch["row_index"] = row_index
        return batch


def scored_positions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Non-padded positions that this window is responsible for scoring."""

    valid = ~batch["padding_mask"]
    if "score_mask" in batch:
        valid = valid & batch["score_mask"].bool()
    return valid


def select_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_proposed_model(
    root: Path,
    *,
    sequence_length: int = 200,
    ablations: dict[str, bool] | None = None,
) -> NuLKT:
    """Instantiate the proposed architecture exactly as the training stage does."""

    settings = dict(PROPOSED_MODEL_SETTINGS)
    settings["max_sequence_length"] = sequence_length
    return NuLKT(
        NuLKTConfig(
            vocab_sizes=vocab_sizes_for_model(root),
            ablations=dict(ablations or {}),
            **settings,
        )
    )


def load_proposed_model(
    root: Path,
    *,
    seed: int = 42,
    sequence_length: int = 200,
    device: str | torch.device | None = None,
    tag: str = "proposed",
) -> tuple[NuLKT, torch.device, dict[str, Any]]:
    """Load the validation-selected checkpoint for one seed."""

    checkpoint_path = (
        root / "results" / "raw_runs" / tag / f"seed_{seed}" / "checkpoints" / f"best_seed_{seed}.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint_path}. Run `python run_all.py --stage proposed` first."
        )
    resolved = select_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)

    # Structural settings such as the attention-masking mode change what the model
    # computes without changing any parameter shape, so a mismatched default would
    # load cleanly and produce quietly wrong results. Prefer the configuration the
    # checkpoint was trained under.
    stored = (checkpoint.get("extra") or {}).get("model_config")
    if stored:
        recorded = dict(stored)
        recorded.pop("vocab_sizes", None)
        ablations = recorded.pop("ablations", {}) or {}
        recorded.pop("max_sequence_length", None)
        model = NuLKT(
            NuLKTConfig(
                vocab_sizes=vocab_sizes_for_model(root),
                ablations=dict(ablations),
                max_sequence_length=sequence_length,
                **recorded,
            )
        )
        config_source = "checkpoint"
    else:
        model = build_proposed_model(root, sequence_length=sequence_length)
        config_source = "module defaults (checkpoint predates config recording)"
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved).eval()
    provenance = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "selected_on": "validation loss",
        "architecture_config_source": config_source,
        "within_exercise_bidirectional": bool(model.config.within_exercise_bidirectional),
    }
    if not stored:
        provenance["warning"] = (
            "This checkpoint records no architecture configuration. It is being rebuilt "
            "from current module defaults, which may differ from how it was trained; "
            "retrain to remove this ambiguity."
        )
    return model, resolved, provenance


def indexed_dataloader(
    root: Path,
    *,
    split: str,
    protocol: str = "chronological_split",
    sequence_length: int = 200,
    batch_size: int = 64,
    row_limit: int | None = None,
    frame: pd.DataFrame | None = None,
    context_length: int = 0,
) -> tuple[DataLoader, pd.DataFrame]:
    """Build a deterministic, row-indexed loader for one partition."""

    if frame is None:
        frame, _ = load_processed_frame(
            root, row_limit=row_limit, derive_context=False,
            split_filter=(protocol, {split}),
        )
    if protocol not in frame.columns:
        raise KeyError(f"Unknown split protocol column: {protocol}")
    subset = frame[frame[protocol].astype(str) == split].copy()
    if subset.empty:
        raise ValueError(f"Partition {split!r} of protocol {protocol!r} is empty")
    dataset = IndexedSequenceDataset(
        subset, max_length=sequence_length, context_length=context_length
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, frame


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    outputs: Iterable[str] = SCALAR_OUTPUTS,
    with_expert_weights: bool = True,
) -> pd.DataFrame:
    """Run the model and return one row per non-padded token.

    Diagnostics are disabled so the O(L^2) attention tensors are never
    materialized for a full partition; :func:`collect_attention_parameters`
    handles attention separately on a bounded subsample.
    """

    model.eval()
    wanted = list(outputs)
    frames: list[pd.DataFrame] = []
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        result = model(moved, return_diagnostics=False)
        valid = scored_positions(moved)
        record: dict[str, np.ndarray] = {
            "row_index": moved["row_index"][valid].cpu().numpy(),
            "label": moved["responses"][valid].cpu().numpy(),
            "learner_id": moved["learner"][valid].cpu().numpy(),
            "logit_probability": torch.sigmoid(result["logits"])[valid].cpu().numpy(),
        }
        for name in wanted:
            if name in result:
                record[name] = result[name][valid].float().cpu().numpy()
        if with_expert_weights and "expert_weights" in result:
            weights = result["expert_weights"][valid].float().cpu().numpy()
            for index in range(weights.shape[-1]):
                record[f"expert_{index}"] = weights[:, index]
        frames.append(pd.DataFrame(record))
    if not frames:
        raise RuntimeError("The loader produced no batches")
    collected = pd.concat(frames, ignore_index=True)
    return collected[collected["row_index"] >= 0].reset_index(drop=True)


@torch.inference_mode()
def monte_carlo_dropout(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    passes: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Stochastic-forward-pass uncertainty, used as a control for the evidential head."""

    if passes < 2:
        raise ValueError("passes must be at least 2")
    model.eval()
    for module in model.modules():  # dropout only; normalization stays in eval mode
        if isinstance(module, nn.Dropout):
            module.train()
    torch.manual_seed(seed)
    row_indices: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        valid = scored_positions(moved)
        row_indices.append(moved["row_index"][valid].cpu().numpy())
        draws = [
            torch.sigmoid(model(moved, return_diagnostics=False)["logits"])[valid].float().cpu().numpy()
            for _ in range(passes)
        ]
        samples.append(np.stack(draws, axis=0))
    model.eval()
    stacked = np.concatenate(samples, axis=1)
    frame = pd.DataFrame(
        {
            "row_index": np.concatenate(row_indices),
            "mc_mean": stacked.mean(axis=0),
            "mc_std": stacked.std(axis=0, ddof=1),
        }
    )
    return frame[frame["row_index"] >= 0].reset_index(drop=True)


def join_corpus_attributes(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Attach corpus attributes to predictions through the retained row index."""

    available = [name for name in columns if name in frame.columns]
    missing = sorted(set(columns) - set(available))
    if missing:
        raise KeyError(f"Requested corpus columns are absent: {missing}")
    attributes = frame.loc[predictions["row_index"].to_numpy(), available].reset_index(drop=True)
    return pd.concat([predictions.reset_index(drop=True), attributes], axis=1)
