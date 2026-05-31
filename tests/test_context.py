"""Tests for CellQuorum pipeline context objects."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.context import PipelineContext, PipelinePaths


def test_pipeline_paths_from_output_dir_builds_standard_layout(tmp_path: Path) -> None:
    """
    Verify that PipelinePaths creates the standardized CellQuorum run layout.

    CellQuorum stages should not invent arbitrary output locations. This test
    ensures that a single output directory expands into the predictable results,
    figures, reports, objects, provenance, logs, and scratch directories used by
    every pipeline stage.
    """

    # Build a temporary output directory for the test run.
    output_dir = tmp_path / "cellquorum_run"

    # Create standardized pipeline paths from the output directory.
    paths = PipelinePaths.from_output_dir(output_dir)

    # Confirm the root path resolves to the expected output directory.
    assert paths.root == output_dir.resolve()

    # Confirm the results directory follows the standard layout.
    assert paths.results == output_dir.resolve() / "results"

    # Confirm the figures directory follows the standard layout.
    assert paths.figures == output_dir.resolve() / "figures"

    # Confirm the reports directory follows the standard layout.
    assert paths.reports == output_dir.resolve() / "reports"

    # Confirm the objects directory follows the standard layout.
    assert paths.objects == output_dir.resolve() / "objects"

    # Confirm the provenance directory follows the standard layout.
    assert paths.provenance == output_dir.resolve() / "provenance"

    # Confirm the logs directory follows the standard layout.
    assert paths.logs == output_dir.resolve() / "logs"

    # Confirm the scratch directory follows the standard layout.
    assert paths.scratch == output_dir.resolve() / "scratch"


def test_pipeline_paths_ensure_directories_creates_standard_layout(tmp_path: Path) -> None:
    """
    Verify that PipelinePaths can create all standard output directories.

    Pipeline setup should create the run layout once so individual stages can
    write artifacts without repeatedly checking whether shared directories
    exist. This test ensures that directory creation is centralized and reliable.
    """

    # Build standardized pipeline paths inside the temporary directory.
    paths = PipelinePaths.from_output_dir(tmp_path / "cellquorum_run")

    # Create every standard directory in the run layout.
    paths.ensure_directories()

    # Confirm the root output directory exists.
    assert paths.root.exists()

    # Confirm the results directory exists.
    assert paths.results.exists()

    # Confirm the figures directory exists.
    assert paths.figures.exists()

    # Confirm the reports directory exists.
    assert paths.reports.exists()

    # Confirm the objects directory exists.
    assert paths.objects.exists()

    # Confirm the provenance directory exists.
    assert paths.provenance.exists()

    # Confirm the logs directory exists.
    assert paths.logs.exists()

    # Confirm the scratch directory exists.
    assert paths.scratch.exists()


def test_pipeline_context_require_adata_returns_existing_adata(tmp_path: Path) -> None:
    """
    Verify that PipelineContext.require_adata returns the active AnnData object.

    Stages that require expression data should call this helper instead of
    directly accessing context.adata. That gives failures a clear message when
    AnnData has not yet been loaded.
    """

    # Create a tiny AnnData object for the test context.
    adata = ad.AnnData(X=np.ones((2, 3)))

    # Build standardized paths for the test context.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create a pipeline context containing the AnnData object.
    context = PipelineContext(config={}, paths=paths, adata=adata)

    # Confirm require_adata returns the same object.
    assert context.require_adata() is adata


def test_pipeline_context_require_adata_raises_clear_error_when_missing(tmp_path: Path) -> None:
    """
    Verify that PipelineContext.require_adata raises a clear error when missing.

    This protects future stages from vague NoneType failures and makes pipeline
    failures easier for users to understand.
    """

    # Build standardized paths for the test context.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create a context without an AnnData object.
    context = PipelineContext(config={}, paths=paths)

    # Confirm a clear runtime error is raised when AnnData is missing.
    with pytest.raises(RuntimeError, match="does not contain an AnnData object"):
        context.require_adata()


def test_pipeline_context_require_manifest_returns_existing_manifest(tmp_path: Path) -> None:
    """
    Verify that PipelineContext.require_manifest returns the active manifest.

    Several stages need sample-level metadata. This helper gives those stages a
    consistent way to access the manifest and fail clearly if it has not been
    loaded.
    """

    # Build a tiny manifest table.
    manifest = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "condition": ["control", "case"],
        }
    )

    # Build standardized paths for the test context.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create a pipeline context containing the manifest.
    context = PipelineContext(config={}, paths=paths, manifest=manifest)

    # Confirm require_manifest returns the same manifest object.
    assert context.require_manifest() is manifest


def test_pipeline_context_require_manifest_raises_clear_error_when_missing(tmp_path: Path) -> None:
    """
    Verify that PipelineContext.require_manifest raises a clear error when missing.

    This protects future stages from unclear pandas or metadata lookup errors
    when the manifest has not been loaded.
    """

    # Build standardized paths for the test context.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create a context without a manifest table.
    context = PipelineContext(config={}, paths=paths)

    # Confirm a clear runtime error is raised when the manifest is missing.
    with pytest.raises(RuntimeError, match="does not contain a manifest table"):
        context.require_manifest()


def test_pipeline_context_with_adata_preserves_runtime_state(tmp_path: Path) -> None:
    """
    Verify that PipelineContext.with_adata returns a context with updated data.

    This supports clean stage chaining: one stage can update the AnnData object
    while preserving config, paths, manifest, backend registry, run identifiers,
    random seed, and metadata.
    """

    # Create the original AnnData object.
    original_adata = ad.AnnData(X=np.ones((2, 3)))

    # Create the updated AnnData object.
    updated_adata = ad.AnnData(X=np.zeros((4, 5)))

    # Build a tiny manifest table.
    manifest = pd.DataFrame({"sample_id": ["S1"]})

    # Build standardized paths for the test context.
    paths = PipelinePaths.from_output_dir(tmp_path / "run")

    # Create the original pipeline context.
    context = PipelineContext(
        config={"profile": "test"},
        paths=paths,
        adata=original_adata,
        manifest=manifest,
        backend_registry={"python": "available"},
        run_id="test-run",
        random_seed=42,
        metadata={"key": "value"},
    )

    # Create a new context with the updated AnnData object.
    updated_context = context.with_adata(updated_adata)

    # Confirm the updated context contains the new AnnData object.
    assert updated_context.adata is updated_adata

    # Confirm the original context still contains the original AnnData object.
    assert context.adata is original_adata

    # Confirm the config was preserved.
    assert updated_context.config == {"profile": "test"}

    # Confirm paths were preserved.
    assert updated_context.paths is paths

    # Confirm the manifest was preserved.
    assert updated_context.manifest is manifest

    # Confirm the backend registry was preserved.
    assert updated_context.backend_registry == {"python": "available"}

    # Confirm the run ID was preserved.
    assert updated_context.run_id == "test-run"

    # Confirm the random seed was preserved.
    assert updated_context.random_seed == 42

    # Confirm metadata was preserved.
    assert updated_context.metadata == {"key": "value"}

    # Confirm metadata was copied rather than aliased.
    assert updated_context.metadata is not context.metadata
