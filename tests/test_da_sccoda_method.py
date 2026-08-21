"""Tests for SccodaMethod differential abundance."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.backends.registry import build_default_backend_registry
from cellquorum.backends.sccoda_backend import build_sccoda_backend
from cellquorum.comparative.differential_abundance.sccoda_method import SccodaMethod
from cellquorum.methods.base import MethodSkip

# Check sccoda_env availability once.
_SCCODA_AVAILABLE = build_sccoda_backend().status().available


@pytest.fixture
def synthetic_adata():
    """
    Build a synthetic cohort with 3 cell types across 6 donors (3 Normal, 3 Disease).

    Cell type 2 is enriched in Disease (more cells in Disease samples).
    """

    np.random.seed(42)

    # Generate counts for each donor and cell type (Poisson-like distribution)
    # Donors: N1, N2, N3 (Normal) and D1, D2, D3 (Disease)
    # Cell types: Type0, Type1, Type2
    # Type2 is enriched in Disease samples

    donor_ids = [
        "N1",
        "N1",
        "N1",
        "N2",
        "N2",
        "N2",
        "N3",
        "N3",
        "N3",
        "D1",
        "D1",
        "D1",
        "D2",
        "D2",
        "D2",
        "D3",
        "D3",
        "D3",
    ]
    conditions = ["Normal"] * 9 + ["Disease"] * 9
    cell_types = ["Type0", "Type1", "Type2"] * 6

    # Normal: roughly [100, 50, 30] cells per type per donor
    # Disease: roughly [60, 90, 30] cells per type per donor (Type1 enriched)
    n_cells_list = []
    for _donor, cond in zip(donor_ids, conditions, strict=False):
        ct = cell_types[len(n_cells_list)]
        if cond == "Normal":
            if ct == "Type0":
                n = np.random.poisson(100)
            elif ct == "Type1":
                n = np.random.poisson(50)
            else:  # Type2
                n = np.random.poisson(30)
        else:  # Disease
            if ct == "Type0":
                n = np.random.poisson(60)
            elif ct == "Type1":
                n = np.random.poisson(90)
            else:  # Type2
                n = np.random.poisson(30)
        n_cells_list.append(n)

    # Build per-cell obs
    obs_rows = []
    for donor, cond, ct, n_cells in zip(
        donor_ids, conditions, cell_types, n_cells_list, strict=False
    ):
        for _ in range(n_cells):
            obs_rows.append({"donor": donor, "condition": cond, "cell_type": ct})

    obs = pd.DataFrame(obs_rows)
    n_obs = len(obs)

    # Dummy X matrix (doesn't matter for DA)
    X = np.zeros((n_obs, 10))

    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def mock_context(tmp_path):
    """Build a mock stage context with paths and backend registry."""

    class Paths:
        scratch = tmp_path / "scratch"
        results = tmp_path / "results"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    return Context()


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_happy_path_auto_only(synthetic_adata, mock_context):
    """Run scCODA with auto-reference only and verify the output."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "seed": 0,
        "num_iterations": 2000,  # Fast for testing
    }

    result = method.run(synthetic_adata, config, mock_context)

    # Should not be a skip
    assert not isinstance(result, MethodSkip)

    # Check artifacts
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.name == "da_results"
    assert artifact.path.name == "da_sccoda.csv"
    assert artifact.kind == "csv"

    # Read the CSV and verify structure
    df = pd.read_csv(artifact.path)
    assert set(df.columns) == {
        "cell_type",
        "log2_fold_change",
        "inclusion_probability",
        "credible_effect",
        "reference",
    }

    # Should contain "auto" reference
    assert "auto" in df["reference"].values

    # Should have 3 rows (3 cell types, auto reference only)
    assert len(df) == 3

    # Metrics should be populated
    assert result.metrics["case"] == "Disease"
    assert result.metrics["control"] == "Normal"
    assert result.metrics["n_samples"] == 6
    assert result.metrics["n_celltypes"] == 3


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_dual_reference(synthetic_adata, mock_context):
    """Run scCODA with explicit reference and verify both auto and explicit appear."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "reference_celltype": "Type0",
        "seed": 0,
        "num_iterations": 2000,
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert not isinstance(result, MethodSkip)

    df = pd.read_csv(result.artifacts[0].path)

    # Should contain BOTH "auto" and "Type0" in reference column
    references = set(df["reference"].values)
    assert "auto" in references
    assert "Type0" in references

    # Should have two sets of results (one per reference)
    assert len(df) == 6  # 3 cell types × 2 references


@pytest.mark.skipif(not _SCCODA_AVAILABLE, reason="sccoda_env not available")
def test_sccoda_determinism(synthetic_adata, mock_context, tmp_path):
    """Run scCODA twice with the same seed and verify identical credible_effect."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
        "seed": 0,
        "num_iterations": 2000,
    }

    # Run 1
    context1 = mock_context
    result1 = method.run(synthetic_adata, config, context1)
    assert not isinstance(result1, MethodSkip)
    df1 = pd.read_csv(result1.artifacts[0].path)

    # Run 2 (fresh context with different paths)
    class Paths:
        scratch = tmp_path / "run2_scratch"
        results = tmp_path / "run2_results"

    class Context:
        paths = Paths()
        backend_registry = build_default_backend_registry()

    context2 = Context()
    result2 = method.run(synthetic_adata, config, context2)
    assert not isinstance(result2, MethodSkip)
    df2 = pd.read_csv(result2.artifacts[0].path)

    # credible_effect should be identical
    assert df1["credible_effect"].tolist() == df2["credible_effect"].tolist()


def test_sccoda_skip_missing_cell_type_col(synthetic_adata, mock_context):
    """SccodaMethod should skip when cell_type_col is missing."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "nonexistent_column",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "Disease",
        "control": "Normal",
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "cell_type_col" in result.reason or "nonexistent_column" in result.reason


def test_sccoda_skip_missing_case_control(synthetic_adata, mock_context):
    """SccodaMethod should skip when case or control is unset."""

    method = SccodaMethod()
    config = {
        "cell_type_col": "cell_type",
        "condition_col": "condition",
        "donor_col": "donor",
        # No case/control
    }

    result = method.run(synthetic_adata, config, mock_context)

    assert isinstance(result, MethodSkip)
    assert "case" in result.reason or "control" in result.reason
