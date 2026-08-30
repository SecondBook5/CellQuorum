"""Tests for the SubclusteringStage return contract.

The regression these lock down: subclustering must NOT replace the pipeline's
working object with the focus subset. Doing so stripped the integration
embedding (X_pca_harmony) that downstream stages need, and — when the focus
matched zero cells — propagated a 0-cell object that poisoned population_identity,
embeddings, DE, and CCC. The stage computes on the focus internally but must
return the ORIGINAL object (all cells, embeddings intact) with subcluster labels
projected back onto it.
"""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.clustering.subclustering.stage import SubclusteringStage


def _parent_adata(seed=0):
    """A clustered, integrated parent object as it reaches subclustering."""
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.random((n, 8)).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=[f"g{i}" for i in range(8)]))
    a.obs_names = [f"cell_{i}" for i in range(n)]
    a.layers["counts"] = (x * 10).astype(np.float32)
    # Integration embedding the downstream stages depend on.
    a.obsm["X_pca_harmony"] = rng.random((n, 5)).astype(np.float32)
    a.obsm["X_pca"] = rng.random((n, 5)).astype(np.float32)
    a.obs["leiden"] = pd.Categorical(np.repeat(["0", "1", "2"], n // 3))
    a.obs["cell_type"] = pd.Categorical(["Fibroblasts"] * n)
    a.obs["sample_id"] = pd.Categorical(np.tile(["s1", "s2", "s3"], n // 3))
    a.obs["donor_id"] = pd.Categorical(np.tile(["d1", "d2", "d3"], n // 3))
    return a


def _context(adata, focus_labels):
    """Minimal context with a real SubclusteringConfig and no R backend."""
    sc_config = SubclusteringConfig(
        enabled=True,
        counts_layer="counts",
        focus={"label_key": "cell_type", "labels": focus_labels},
        group_filter={"group_key": "sample_id", "min_cells": 5},
        partition={"method": "choir", "seeds": [0]},
        donor_gate={"group_key": "donor_id", "min_groups": 3},
        key_added="fibroblast_subcluster",
    )
    config = SimpleNamespace(
        subclustering=sc_config,
        cohort=SimpleNamespace(donor_key="donor_id", sample_key="sample_id", focus=None),
    )
    # backend_registry.get("rscript") must raise/return None so CHOIR skips.
    registry = SimpleNamespace(get=lambda name: None)
    paths = SimpleNamespace(figures="/tmp", scratch="/tmp", root="/tmp")
    return SimpleNamespace(config=config, adata=adata, paths=paths, backend_registry=registry)


def test_subclustering_preserves_parent_cells_and_embeddings():
    """Focus matches all cells: return the full object with X_pca_harmony intact."""
    parent = _parent_adata()
    n_before = parent.n_obs
    ctx = _context(parent, focus_labels=["Fibroblasts"])

    result = SubclusteringStage().run(ctx)

    # All cells retained (flag semantics), not shrunk to the focus subset.
    assert result.adata.n_obs == n_before
    # The integration embedding downstream stages need survives.
    assert "X_pca_harmony" in result.adata.obsm
    assert result.adata.obsm["X_pca_harmony"].shape[0] == n_before


def test_subclustering_zero_cell_focus_does_not_poison_object():
    """Focus matches 0 cells: skip cleanly, return the parent unchanged."""
    parent = _parent_adata()
    n_before = parent.n_obs
    ctx = _context(parent, focus_labels=["NoSuchType"])

    result = SubclusteringStage().run(ctx)

    # Never propagate a 0-cell object; the parent passes through intact.
    assert result.adata.n_obs == n_before
    assert "X_pca_harmony" in result.adata.obsm
    assert result.metrics.get("skipped") is True
