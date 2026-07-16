"""Tools namespace: ``cq.tl.*``.

Thin wrappers over the registered analysis stages. Each returns a
:class:`~cellquorum._notebook.NotebookStageOutput` (use ``.adata`` for the
updated object, ``.result`` for artifacts/metrics/warnings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellquorum._notebook import NotebookStageOutput, run_stage

if TYPE_CHECKING:
    import anndata as ad


def reduce_dimensions(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the dimensionality-reduction (PCA) stage."""

    return run_stage("dimensionality", adata, **kwargs)


def cluster(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the clustering (neighbors + Leiden) stage."""

    return run_stage("clustering", adata, **kwargs)


def integrate(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the batch-integration stage (e.g. Harmony, scVI)."""

    return run_stage("integration", adata, **kwargs)


def annotate(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the cell-type annotation stage."""

    return run_stage("annotation", adata, **kwargs)


def reference_map(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the reference-mapping (scArches/scANVI) stage."""

    return run_stage("reference_mapping", adata, **kwargs)


def subcluster(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the principled subclustering stage."""

    return run_stage("subclustering", adata, **kwargs)


def adjudicate(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the cluster/state adjudication stage."""

    return run_stage("adjudication", adata, **kwargs)


def population_identity(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the population/state identity evidence stage."""

    return run_stage("population_identity", adata, **kwargs)


__all__ = [
    "adjudicate",
    "annotate",
    "cluster",
    "integrate",
    "population_identity",
    "reduce_dimensions",
    "reference_map",
    "subcluster",
]
