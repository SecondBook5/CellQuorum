# Pipeline step (order=20): qc — the stage's shared exception type.
"""The QC stage's error type, in a module with no dependencies of its own.

Its own file for one reason: ``stage.py``, ``_context.py``, ``_annotate.py`` and ``_report.py``
all raise it, and ``stage.py`` imports the other three. Leaving the class in ``stage.py`` makes
that a circular import. Fifteen lines here is cheaper than a lazy import in four call sites.
"""

from __future__ import annotations

from cellquorum.core.exceptions import CellQuorumDataError


class QCStageError(CellQuorumDataError):
    """Report QC stage execution failures.

    The QC stage is the first full analysis stage. Errors here should explain whether the
    failure came from missing context state, invalid QC configuration, metric calculation,
    thresholding, decision construction, filtering, or artifact writing.
    """


__all__ = ["QCStageError"]
