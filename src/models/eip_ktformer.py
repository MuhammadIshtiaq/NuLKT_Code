"""Deprecated import compatibility for the former EIP-KTFormer name.

NuLKT is the canonical model. This wrapper intentionally remains so historical
experiment scripts and checkpoint-loading code continue to import successfully.
"""

from .nulkt import NuLKT, NuLKTConfig

EIPKTFormer = NuLKT
EIPKTFormerConfig = NuLKTConfig

__all__ = ["EIPKTFormer", "EIPKTFormerConfig", "NuLKT", "NuLKTConfig"]
