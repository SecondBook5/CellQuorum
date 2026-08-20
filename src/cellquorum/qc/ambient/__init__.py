"""Backward-compatible re-export shim.

The ambient-RNA correction stage moved to the top-level
``cellquorum.ambient_correction`` package (CellQuorum consolidation, Move 2).
New code should import from there; this module keeps existing
``from cellquorum.qc.ambient import ...`` imports working.
"""

from __future__ import annotations

from cellquorum.ambient_correction import AmbientCorrectionStage

__all__ = ["AmbientCorrectionStage"]
