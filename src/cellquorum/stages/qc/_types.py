# Pipeline step (order=20): qc — shared type aliases for the QC module.
"""Type aliases the QC module shares, so a matrix is never annotated as ``object``.

``matrix: object`` appeared eleven times across ``metrics.py``, ``validation.py`` and
``lineage.py``. It type-checks trivially and documents nothing: a reader cannot tell whether a
function wants counts or lognorm, dense or sparse, and a type checker cannot tell that
``matrix.sum(axis=0)`` is valid — which is most of the ``"object" has no attribute`` errors the
module reported.

The union is spelled out rather than hidden behind ``Any`` because the dense/sparse distinction
is load-bearing throughout QC. Sparse column indexing over a cohort-scale CSR once overflowed
int32 and segfaulted the interpreter, so "which of these two is it" is a question the signatures
should answer.

``sparray`` is included alongside ``spmatrix`` because scipy is mid-migration between them and
both appear in practice depending on which call produced the matrix.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

from cellquorum.backends.base import BackendStatus

#: A cells x genes expression matrix, dense or sparse. Says nothing about *which* values are in
#: it — that is the layer contract's job (see :mod:`cellquorum.core.contracts`).
type ExpressionMatrix = np.ndarray | sp.spmatrix | sp.sparray

#: A backend that runs a helper script inside an isolated environment.
__all__ = ["ExpressionMatrix", "IsolatedBackend"]


@runtime_checkable
class IsolatedBackend(Protocol):
    """Structural type for an isolated-environment subprocess backend.

    A ``Protocol`` rather than a base class so a test can pass a stub with just these two
    methods — which the archetype audit's tests do, to exercise the "environment absent" path
    without building a micromamba environment in CI.

    The return types are the real ones rather than ``object``. That matters: annotating the
    parameter as ``object`` type-checks but tells a checker nothing, so ``status().available``
    and ``result.returncode`` go unverified, and those are exactly the attributes a caller
    depends on.
    """

    def status(self) -> BackendStatus:
        """Availability of the isolated environment."""
        ...

    def run_helper(
        self,
        script_path: str | Path,
        args: list[str] | None = None,
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a helper script inside the environment."""
        ...
