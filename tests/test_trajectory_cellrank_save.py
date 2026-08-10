"""Artifact-writer test for CellRank fate-mapping h5ad."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.core.stage import StageArtifact
from cellquorum.trajectory.save import write_cellrank_h5ad


def test_write_cellrank_h5ad(tmp_path):
    a = ad.AnnData(np.ones((5, 3), dtype="float32"))
    artifact, note = write_cellrank_h5ad(a, tmp_path)
    assert isinstance(artifact, StageArtifact)
    assert artifact.kind == "h5ad"
    assert artifact.path.exists()
    assert artifact.path.name == "fate_mapping.h5ad"
    assert "cellrank" in note.lower()


def test_write_cellrank_h5ad_write_failure_returns_note(tmp_path):
    a = ad.AnnData(np.ones((5, 3), dtype="float32"))
    missing = tmp_path / "does" / "not" / "exist"  # parent absent → write fails
    artifact, note = write_cellrank_h5ad(a, missing)
    assert artifact is None
    assert "fail" in note.lower()
