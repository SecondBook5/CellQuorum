"""Tests for CellQuorum stage result contracts."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from cellquorum.core.stage import StageArtifact, StageResult


def test_stage_result_accepts_artifacts_notes_warnings_and_metrics() -> None:
    """
    Verify that StageResult stores all required stage execution outputs.

    This test protects the most important early CellQuorum design rule: every
    stage must return the updated data object together with explicit artifacts,
    notes, warnings, and structured metrics. That contract prevents future
    stages from silently writing files or hiding important execution details.
    """

    # Create a tiny AnnData object so the stage result has a real data payload.
    adata = ad.AnnData(X=np.ones((2, 3)))

    # Create a representative artifact entry.
    artifact = StageArtifact(
        name="qc_summary",
        path=Path("results/qc/qc_summary.csv"),
        kind="csv",
        description="Cell-level QC summary table.",
    )

    # Create a stage result with every supported metadata field populated.
    result = StageResult(
        adata=adata,
        artifacts=[artifact],
        notes=["QC completed."],
        warnings=["Example warning."],
        metrics={"n_cells": 2, "n_genes": 3},
    )

    # Confirm the AnnData object was retained.
    assert result.adata.n_obs == 2

    # Confirm the AnnData object retained the expected number of variables.
    assert result.adata.n_vars == 3

    # Confirm artifact metadata is accessible.
    assert result.artifacts[0].name == "qc_summary"

    # Confirm artifact paths are stored as Path objects.
    assert result.artifacts[0].path == Path("results/qc/qc_summary.csv")

    # Confirm notes are preserved.
    assert result.notes == ["QC completed."]

    # Confirm warnings are preserved.
    assert result.warnings == ["Example warning."]

    # Confirm structured metrics are preserved.
    assert result.metrics["n_cells"] == 2

    # Confirm structured metrics can store gene counts.
    assert result.metrics["n_genes"] == 3
