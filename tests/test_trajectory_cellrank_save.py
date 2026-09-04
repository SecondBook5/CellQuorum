"""Artifact-writer test for CellRank fate-mapping h5ad."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.core.stage import StageArtifact
from cellquorum.stages.trajectory.save import write_cellrank_h5ad


def test_write_cellrank_h5ad(tmp_path):
    a = ad.AnnData(np.ones((5, 3), dtype="float32"))
    artifact, note = write_cellrank_h5ad(a, tmp_path)
    assert isinstance(artifact, StageArtifact)
    assert artifact.kind == "h5ad"
    assert artifact.path.exists()
    assert artifact.path.name == "fate_mapping.h5ad"
    assert "cellrank" in note.lower()
    # Default label reflects the whole atlas.
    assert "whole atlas" in artifact.description
    assert "subsampled" not in artifact.description


def test_write_cellrank_h5ad_subsampled_label(tmp_path):
    a = ad.AnnData(np.ones((5, 3), dtype="float32"))
    artifact, _ = write_cellrank_h5ad(a, tmp_path, subsampled=True)
    assert isinstance(artifact, StageArtifact)
    assert "subsampled" in artifact.description
    assert "whole atlas" not in artifact.description


def test_write_cellrank_h5ad_write_failure_returns_note(tmp_path):
    a = ad.AnnData(np.ones((5, 3), dtype="float32"))
    # A missing output directory is created now, so provoke a real failure: put a
    # regular file where the directory would have to be.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    artifact, note = write_cellrank_h5ad(a, blocked)
    assert artifact is None
    assert "fail" in note.lower()
