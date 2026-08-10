"""Artifact-writer tests for the pseudotime methods."""

from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.trajectory.save import write_pseudotime_h5ad


def _adata():
    a = ad.AnnData(np.zeros((5, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(5)]
    return a


def test_write_pseudotime_h5ad_ok(tmp_path):
    artifact, note = write_pseudotime_h5ad(_adata(), tmp_path, "dpt")
    assert artifact is not None
    assert artifact.kind == "h5ad"
    assert (tmp_path / "dpt_pseudotime.h5ad").exists()
    assert "dpt" in artifact.description


def test_write_pseudotime_h5ad_failure_returns_note(tmp_path):
    missing = tmp_path / "does_not_exist"
    artifact, note = write_pseudotime_h5ad(_adata(), missing, "palantir")
    assert artifact is None
    assert "failed" in note.lower()


def test_write_pseudotime_h5ad_labels_scope_honestly(tmp_path):
    """The description reflects whether the written object is a subset."""
    whole, _ = write_pseudotime_h5ad(_adata(), tmp_path, "dpt", subset=False)
    assert "whole object" in whole.description
    assert "subset" not in whole.description
    sub, _ = write_pseudotime_h5ad(_adata(), tmp_path, "palantir", subset=True)
    assert "subset" in sub.description
    assert "whole object" not in sub.description
