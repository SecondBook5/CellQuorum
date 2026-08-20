"""Backward-compatible re-export shim.

The QC diagnostic figures moved to ``cellquorum.visualization.qc.diagnostics``
(CellQuorum consolidation, Move 2). New code should import from there.
"""

from __future__ import annotations

from cellquorum.visualization.qc.diagnostics import (
    QCVisualizationError,
    QCVisualizationResult,
    write_qc_figures,
)

__all__ = ["QCVisualizationError", "QCVisualizationResult", "write_qc_figures"]
