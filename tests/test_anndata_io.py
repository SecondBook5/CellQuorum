"""Tests for CellQuorum AnnData I/O utilities."""

from __future__ import annotations

# Import Path for pytest tmp_path annotations.
from pathlib import Path

# Import AnnData for test object construction.
import anndata as ad

# Import NumPy for deterministic test matrices.
import numpy as np

# Import pandas for AnnData obs/var metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import AnnData I/O utilities under test.
from cellquorum.io import (
    AnnDataLoadError,
    load_adata,
    normalize_adata_path,
    validate_adata_path,
)


def make_test_adata() -> ad.AnnData:
    """
    Build a small AnnData object for I/O tests.

    Returns:
        Small AnnData object with deterministic names and values.
    """

    # Build a deterministic matrix.
    matrix = np.array(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(
        {
            "sample": ["sample_a", "sample_b"],
        },
        index=["cell_1", "cell_2"],
    )

    # Build variable metadata.
    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])

    # Return the AnnData object.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_normalize_adata_path_accepts_string_path(tmp_path: Path) -> None:
    """
    Verify AnnData path normalization accepts string paths.

    User-facing config values commonly arrive as strings.
    """

    # Build a string path.
    path = tmp_path / "example.h5ad"

    # Normalize the path.
    normalized = normalize_adata_path(str(path))

    # Confirm the normalized path is a Path object.
    assert normalized == path


def test_normalize_adata_path_accepts_path_object(tmp_path: Path) -> None:
    """
    Verify AnnData path normalization accepts Path objects.

    Programmatic callers often pass pathlib paths directly.
    """

    # Build a Path object.
    path = tmp_path / "example.h5ad"

    # Normalize the path.
    normalized = normalize_adata_path(path)

    # Confirm the path is preserved.
    assert normalized == path


def test_normalize_adata_path_rejects_empty_string() -> None:
    """
    Verify AnnData path normalization rejects empty string paths.

    Path('') becomes '.', so this must be rejected before Path conversion.
    """

    # Confirm empty paths fail clearly.
    with pytest.raises(AnnDataLoadError, match="cannot be empty"):
        normalize_adata_path("")


def test_validate_adata_path_accepts_existing_h5ad_file(tmp_path: Path) -> None:
    """
    Verify AnnData path validation accepts an existing h5ad file.

    The validator should only check path-level constraints, not read contents.
    """

    # Build a valid h5ad path.
    path = tmp_path / "input.h5ad"

    # Write a valid AnnData object.
    make_test_adata().write_h5ad(path)

    # Validate the path.
    validated = validate_adata_path(path)

    # Confirm the validated path is returned.
    assert validated == path


def test_validate_adata_path_rejects_missing_file(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects missing files.

    Missing inputs should fail before AnnData attempts to read them.
    """

    # Build a missing file path.
    path = tmp_path / "missing.h5ad"

    # Confirm missing files fail clearly.
    with pytest.raises(AnnDataLoadError, match="does not exist"):
        validate_adata_path(path)


def test_validate_adata_path_rejects_directory(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects directories.

    Users should provide an h5ad file, not a directory.
    """

    # Confirm directories fail clearly.
    with pytest.raises(AnnDataLoadError, match="not a file"):
        validate_adata_path(tmp_path)


def test_validate_adata_path_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects unsupported file suffixes.

    This first input loader intentionally supports h5ad only.
    """

    # Build an unsupported file path.
    path = tmp_path / "input.csv"

    # Write placeholder contents.
    path.write_text("not,h5ad\n", encoding="utf-8")

    # Confirm unsupported suffixes fail clearly.
    with pytest.raises(AnnDataLoadError, match="supports AnnData input only as '.h5ad'"):
        validate_adata_path(path)


def test_load_adata_reads_h5ad_file(tmp_path: Path) -> None:
    """
    Verify load_adata reads an h5ad file into AnnData.

    This is the minimal data-loading path needed before pipeline execution can
    populate context.adata.
    """

    # Build a test AnnData object.
    expected = make_test_adata()

    # Write the object to h5ad.
    path = tmp_path / "input.h5ad"
    expected.write_h5ad(path)

    # Load the AnnData object.
    observed = load_adata(path)

    # Confirm the returned object type.
    assert isinstance(observed, ad.AnnData)

    # Confirm shape round-tripped.
    assert observed.shape == expected.shape

    # Confirm observation names round-tripped.
    assert list(observed.obs_names) == ["cell_1", "cell_2"]

    # Confirm variable names round-tripped.
    assert list(observed.var_names) == ["gene_1", "gene_2", "gene_3"]

    # Confirm observation metadata round-tripped.
    assert observed.obs["sample"].tolist() == ["sample_a", "sample_b"]

    # Confirm matrix values round-tripped.
    np.testing.assert_array_equal(observed.X, expected.X)


def test_load_adata_accepts_string_path(tmp_path: Path) -> None:
    """
    Verify load_adata accepts string paths.

    YAML/config values will typically pass file paths as strings.
    """

    # Build and write a test AnnData file.
    path = tmp_path / "input.h5ad"
    make_test_adata().write_h5ad(path)

    # Load through a string path.
    observed = load_adata(str(path))

    # Confirm the file was loaded.
    assert observed.shape == (2, 3)


def test_load_adata_rejects_corrupt_h5ad_file(tmp_path: Path) -> None:
    """
    Verify load_adata wraps AnnData/HDF5 read failures.

    Corrupt h5ad files should raise a CellQuorum-specific data error.
    """

    # Build a corrupt h5ad path.
    path = tmp_path / "corrupt.h5ad"

    # Write invalid h5ad contents.
    path.write_text("this is not a valid h5ad file", encoding="utf-8")

    # Confirm corrupt files fail clearly.
    with pytest.raises(AnnDataLoadError, match="Failed to read AnnData file"):
        load_adata(path)


def test_load_adata_rejects_non_h5ad_path_before_reading(tmp_path: Path) -> None:
    """
    Verify load_adata rejects unsupported suffixes before reading.

    This keeps error messages clear for common user mistakes.
    """

    # Build an unsupported file path.
    path = tmp_path / "input.txt"

    # Write placeholder contents.
    path.write_text("not h5ad", encoding="utf-8")

    # Confirm unsupported files fail at validation time.
    with pytest.raises(AnnDataLoadError, match="supports AnnData input only"):
        load_adata(path)
