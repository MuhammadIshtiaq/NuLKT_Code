"""Stable public constructor used by experiment orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .baselines import AKT, DeepFM, DKT, DKVMN, GRUBaseline, HierarchicalTransformerBaseline, LSTMBaseline, MLPBaseline, SAKT, SAINTPlus
from .nulkt import NuLKT, NuLKTConfig


def create_model(name: str, vocab_sizes: Mapping[str, int], model_config: Mapping[str, Any] | None = None) -> nn.Module:
    """Create a neural model by its documented experiment name."""

    settings = dict(model_config or {})
    settings.pop("name", None)
    name = name.lower()
    if name in {"nulkt", "eip_ktformer"}:
        return NuLKT(NuLKTConfig(vocab_sizes=dict(vocab_sizes), **settings))
    constructors = {
        "mlp": lambda: MLPBaseline(vocab_sizes, **settings),
        "lstm": lambda: LSTMBaseline(vocab_sizes, **settings),
        "gru": lambda: GRUBaseline(vocab_sizes, **settings),
        "deepfm": lambda: DeepFM(vocab_sizes, **settings),
        "dkt": lambda: DKT(vocab_sizes["token"], **settings),
        "dkvmn": lambda: DKVMN(vocab_sizes["token"], **settings),
        "sakt": lambda: SAKT(vocab_sizes["token"], **settings),
        "akt": lambda: AKT(vocab_sizes["token"], **settings),
        "saint_plus": lambda: SAINTPlus(vocab_sizes, **settings),
        "hierarchical_transformer": lambda: HierarchicalTransformerBaseline(vocab_sizes, **settings),
    }
    try:
        return constructors[name]()
    except KeyError as error:
        available = constructors | {"nulkt": None, "eip_ktformer": None}
        raise ValueError(f"Unknown neural model {name!r}; available={sorted(available)}") from error
