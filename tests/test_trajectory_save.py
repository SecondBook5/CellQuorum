from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.trajectory.save import safe_name, write_velocity_h5ad


def test_safe_name_sanitizes():
    assert safe_name("CD8+ T / naive") == "CD8__T___naive"
    assert safe_name("Fibroblast") == "Fibroblast"


def test_write_velocity_h5ad_writes_and_returns_artifact(tmp_path):
    a = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    results = tmp_path / "trajectory" / "velocity"
    results.mkdir(parents=True)
    artifact, note = write_velocity_h5ad(a, results, "Fibroblast")
    assert artifact is not None
    assert artifact.kind == "h5ad"
    assert (results / "Fibroblast.h5ad").exists()
    assert "Fibroblast" in note


def test_write_velocity_h5ad_write_failure_is_skip_not_crash(tmp_path):
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    missing = tmp_path / "does" / "not" / "exist"  # parent absent → write fails
    artifact, note = write_velocity_h5ad(a, missing, "grp")
    assert artifact is None
    assert "failed" in note.lower()
