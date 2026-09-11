# Pipeline step (order=20): qc — the stage's shared exception type.
"""The QC stage's error type, in a module with no dependencies of its own.

Its own file for one reason: ``stage.py``, ``_context.py``, and ``reporting.py`` all raise
it, and ``stage.py`` imports the other two. Leaving the class in ``stage.py`` makes that a
circular import. A dozen lines here is cheaper than a lazy import in every call site.
"""

from __future__ import annotations

from cellquorum.core.exceptions import CellQuorumDataError


class QCStageError(CellQuorumDataError):
    """Report QC stage execution failures."""


__all__ = ["QCStageError"]
