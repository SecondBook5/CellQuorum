"""Backward-compatible re-export shim.

The publication QC figures moved to ``cellquorum.visualization.qc.publication``
(CellQuorum consolidation, Move 2). New code should import from there.
"""

from __future__ import annotations

from cellquorum.visualization.qc.publication import (
    QCPublicationFigureError,
    write_publication_qc_figures,
)

__all__ = ["QCPublicationFigureError", "write_publication_qc_figures"]
