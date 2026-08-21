"""Preprocessing namespace: ``cq.pp.*``.

Thin wrappers over the registered preprocessing stages. Each returns a
:class:`~cellquorum.api._notebook.NotebookStageOutput` (use ``.adata`` for the
updated object, ``.result`` for artifacts/metrics/warnings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellquorum.api._notebook import NotebookStageOutput, run_stage

if TYPE_CHECKING:
    import anndata as ad


def qc(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the QC stage (metrics, thresholds, decisions, figures)."""

    return run_stage("qc", adata, **kwargs)


def normalize(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the preprocessing/normalization stage."""

    return run_stage("preprocessing", adata, **kwargs)


def select_features(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the feature-selection (HVG) stage."""

    return run_stage("feature_selection", adata, **kwargs)


def correct_ambient(adata: ad.AnnData, **kwargs: Any) -> NotebookStageOutput:
    """Run the ambient-RNA correction stage."""

    return run_stage("ambient_correction", adata, **kwargs)


__all__ = ["correct_ambient", "normalize", "qc", "select_features"]
