from __future__ import annotations

import anndata as ad
import numpy as np

from cellquorum.stages.trajectory.save import record_write, safe_name, write_velocity_h5ad


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
    # A missing output directory is created now, so provoke a real failure: put a
    # regular file where the directory would have to be.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    artifact, note = write_velocity_h5ad(a, blocked, "grp")
    assert artifact is None
    assert "failed" in note.lower()


def test_record_write_files_a_failed_write_as_a_warning(tmp_path):
    # The reason this matters: a run whose h5ad writes ALL failed reported
    # status=success with an empty warnings list, because every writer reported
    # failure as a note and nobody reads 26 notes.
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    notes: list[str] = []
    warnings: list[str] = []

    artifact = record_write(write_velocity_h5ad(a, blocked, "grp"), notes=notes, warnings=warnings)

    assert artifact is None
    assert notes == []
    assert len(warnings) == 1
    assert "failed" in warnings[0]


def test_record_write_files_a_successful_write_as_a_note(tmp_path):
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    notes: list[str] = []
    warnings: list[str] = []

    artifact = record_write(write_velocity_h5ad(a, tmp_path, "grp"), notes=notes, warnings=warnings)

    assert artifact is not None
    assert warnings == []
    assert len(notes) == 1


def test_safe_name_collisions_produce_distinct_files(tmp_path):
    """Groups differing only in non-alphanumeric chars must write distinct files."""
    results = tmp_path / "trajectory" / "velocity"
    results.mkdir(parents=True)
    a1 = ad.AnnData(X=np.ones((2, 3), dtype="float32"))
    a2 = ad.AnnData(X=np.ones((3, 3), dtype="float32"))

    # These two groups have colliding safe_names: both → "CD8_"
    # Caller must pass distinct stems (simulating the disambiguation logic).
    artifact1, _ = write_velocity_h5ad(a1, results, "CD8+", stem="CD8_")
    artifact2, _ = write_velocity_h5ad(a2, results, "CD8 ", stem="CD8__1")

    assert artifact1 is not None
    assert artifact2 is not None
    # The two artifacts must have DISTINCT paths.
    assert artifact1.path != artifact2.path
    assert artifact1.path.exists()
    assert artifact2.path.exists()
