"""Post-hoc scientific analyses over trained NuLKT checkpoints.

Every module here consumes an already-trained checkpoint and produces
machine-readable evidence. Nothing in this package trains on, tunes against, or
selects using the final test partition; analyses that read test predictions read
artifacts that were already produced under the frozen protocol.
"""

from __future__ import annotations
