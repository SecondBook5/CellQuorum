"""Diagnostics namespace: ``cq.diag.*``.

Thin wrappers over the registered read-only diagnostic stages. Each returns a
:class:`~cellquorum.api._notebook.NotebookStageOutput`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellquorum.api._notebook import NotebookStageOutput, run_stage

if TYPE_CHECKING:
    import anndata as ad


def annotation(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the annotation-diagnostics stage (entropy, scDiagnostics)."""

    return run_stage("annotation_diagnostics", adata, **kwargs)


def integration(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the integration-benchmark stage (scIB-style metrics)."""

    return run_stage("integration_benchmark", adata, **kwargs)


__all__ = ["annotation", "integration"]
