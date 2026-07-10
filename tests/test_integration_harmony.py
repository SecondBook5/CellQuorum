"""Tests for the Harmony integration method."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts import CellQuorumContractError
from cellquorum.integration.harmony import HarmonyMethod


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
    assert after_gap < before_gap * 0.5
    # Provenance recorded.
    assert result.adata.uns["cellquorum"]["integration"]["method"] == "harmony"


def test_harmony_requires_pca_and_batch():
    m = HarmonyMethod()
    a = ad.AnnData(X=np.zeros((10, 5), dtype=np.float32))  # no X_pca, no batch
    with pytest.raises(CellQuorumContractError):
        m.run(
            a,
            {"batch_key": "patient_id", "input_rep": "X_pca"},
            context=None,
            donor_col="patient_id",
        )
