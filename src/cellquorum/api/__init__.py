"""Public Python API for CellQuorum.

This package is the single home for the user-facing surface: the
:func:`run_pipeline` entry point plus the thin notebook namespaces
``tl`` (tools), ``pp`` (preprocessing), ``diag`` (diagnostics), and the
reserved ``evidence`` namespace. The top-level :mod:`cellquorum` package
re-exports these so ``cq.run_pipeline``, ``cq.tl``, ``cq.pp``, ``cq.diag``,
and ``cq.evidence`` remain the canonical access paths.

The former top-level modules (``cellquorum.tl``, ``cellquorum.pp``,
``cellquorum.diag``, ``cellquorum.evidence``, ``cellquorum._notebook``, and
the ``cellquorum.api`` *module*) now live here; thin re-export shims at those
old paths keep every prior import working unchanged.
"""

from __future__ import annotations

# Bind the notebook namespaces as attributes so `cellquorum.api.tl` (and the
# top-level `cq.tl` re-export) resolve to the package submodules.
from cellquorum.api import diag, evidence, pp, tl  # noqa: F401

# The main public pipeline entry point (and its result type).
from cellquorum.api.pipeline import PipelineRunResult, run_pipeline

# The public surface of the `cellquorum.api` namespace itself is the pipeline
# entry — the notebook namespaces are exposed as attributes (and re-exported by
# the top-level package) but are not part of this module's declared `__all__`,
# matching the frozen public-API contract.
__all__ = ["PipelineRunResult", "run_pipeline"]
