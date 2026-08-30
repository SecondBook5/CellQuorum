"""Tests for the Harmony integration method."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.stages.integration.harmony import HarmonyMethod


def _adata_with_pca(n=120, n_pcs=10, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=rng.normal(size=(n, 20)).astype(np.float32))
    pca = rng.normal(size=(n, n_pcs)).astype(np.float32)
    # Two batches with an artificial offset Harmony should reduce.
    batch = np.array(["A", "B"] * (n // 2))
    pca[batch == "B", 0] += 8.0
    a.obsm["X_pca"] = pca
    a.obs["patient_id"] = batch
    return a


def test_harmony_writes_corrected_embedding():
    m = HarmonyMethod()
    a = _adata_with_pca()
    result = m.run(
        a,
        {
            "batch_key": "patient_id",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
            "random_state": 0,
        },
        context=None,
        donor_col="patient_id",
    )
    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    # Corrected embedding exists with the SAME shape as the input (n_cells, n_pcs).
    assert "X_pca_harmony" in result.adata.obsm
    assert result.adata.obsm["X_pca_harmony"].shape == a.obsm["X_pca"].shape
    # Harmony must actually REDUCE the injected batch offset on PC0, not merely
    # return an uncorrupted embedding (guards against the silent-fallback bug).
    batch = a.obs["patient_id"].to_numpy()
    corrected = result.adata.obsm["X_pca_harmony"]
    before_gap = abs(
        a.obsm["X_pca"][batch == "A", 0].mean() - a.obsm["X_pca"][batch == "B", 0].mean()
    )
    after_gap = abs(corrected[batch == "A", 0].mean() - corrected[batch == "B", 0].mean())
    # Require a MEANINGFUL reduction of the injected offset, but not a brittle
    # exact fraction: different harmonypy versions (0.2.0 vs 2.0.0) legitimately
    # correct the tiny synthetic offset to different degrees. A >=20% reduction
    # proves correction happened (vs the silent-fallback bug returning it intact)
    # while staying stable across harmonypy versions.
    assert after_gap < before_gap * 0.8
    # Provenance recorded.
    assert result.adata.uns["cellquorum"]["integration"]["method"] == "harmony"


def test_harmony_accepts_non_contiguous_embedding_view():
    """Harmony copies sliced PCA views before handing them to PyTorch-backed builds."""

    m = HarmonyMethod()
    a = _adata_with_pca(n=80, n_pcs=12)
    a.obsm["X_pca"] = a.obsm["X_pca"][:, :6]

    result = m.run(
        a,
        {
            "batch_key": "patient_id",
            "input_rep": "X_pca",
            "output_rep": "X_pca_harmony",
            "random_state": 0,
        },
        context=None,
        donor_col="patient_id",
    )

    from cellquorum.methods.base import MethodSkip

    assert not isinstance(result, MethodSkip)
    assert result.adata.obsm["X_pca_harmony"].shape == a.obsm["X_pca"].shape


def test_harmony_skips_when_batch_column_absent():
    """Harmony skips gracefully when the batch obs column is missing."""
    m = HarmonyMethod()
    a = ad.AnnData(X=np.zeros((10, 5), dtype=np.float32))
    # Add PCA but no batch column.
    a.obsm["X_pca"] = np.zeros((10, 3), dtype=np.float32)
    result = m.run(
        a,
        {"batch_key": "patient_id", "input_rep": "X_pca"},
        context=None,
        donor_col="patient_id",
    )
    from cellquorum.methods.base import MethodSkip

    assert isinstance(result, MethodSkip)
    assert "patient_id" in result.reason


def test_harmony_raises_when_embedding_absent():
    """Harmony raises via contract when the input embedding is missing."""
    m = HarmonyMethod()
    a = ad.AnnData(X=np.zeros((10, 5), dtype=np.float32))
    # Add batch column but no PCA.
    a.obs["patient_id"] = ["A", "B"] * 5
    with pytest.raises(CellQuorumContractError):
        m.run(
            a,
            {"batch_key": "patient_id", "input_rep": "X_pca"},
            context=None,
            donor_col="patient_id",
        )
