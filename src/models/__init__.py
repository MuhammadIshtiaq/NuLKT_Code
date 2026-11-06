"""Neural models and shared model components."""

from .baselines import (
    AKT,
    DeepFM,
    DKT,
    DKVMN,
    GRUBaseline,
    HierarchicalTransformerBaseline,
    LSTMBaseline,
    MLPBaseline,
    SAKT,
    SAINTPlus,
)
from .nulkt import NuLKT, NuLKTConfig
from .classical import MajorityClassifier, build_classical_baseline, run_classical_baseline
from .utils import count_parameters, parameter_match_ratio
from .factory import create_model
from .batching import quick_batch, validate_model_batch

# Deprecated aliases retained for old notebooks and checkpoint-loading code.
EIPKTFormer = NuLKT
EIPKTFormerConfig = NuLKTConfig

__all__ = [
    "AKT",
    "DeepFM",
    "DKT",
    "DKVMN",
    "EIPKTFormer",
    "EIPKTFormerConfig",
    "NuLKT",
    "NuLKTConfig",
    "GRUBaseline",
    "HierarchicalTransformerBaseline",
    "LSTMBaseline",
    "MajorityClassifier",
    "MLPBaseline",
    "SAKT",
    "SAINTPlus",
    "build_classical_baseline",
    "count_parameters",
    "create_model",
    "parameter_match_ratio",
    "run_classical_baseline",
    "quick_batch",
    "validate_model_batch",
]
