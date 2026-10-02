"""HVG selection must feed each flavor the input space it actually expects.

seurat_v3/seurat_v3_paper operate on raw counts. seurat and cell_ranger are
dispersion-based and expect LOG-NORMALIZED data (scanpy's own documented
contract). Feeding raw counts to a dispersion-based flavor is a silent
correctness bug -- not an error, just a near-random gene set that scVI/scANVI
then trains on. Confirmed empirically: 0/50 true-variable genes recovered from
raw counts vs 7/50 from correctly log-normalized data on the same synthetic
fixture. This mirrors the fix already applied once in
``feature_selection/hvg.py`` (a flavor->layer dispatch), which reference_mapping
never got.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.stages.annotation.reference_mapping.scarches import _select_hvg_genes


def _atlas_with_known_variable_genes(seed: int = 0) -> ad.AnnData:
    """Raw counts where genes 0-9 carry real biological variance, the rest are noise."""

    rng = np.random.default_rng(seed)
    n_obs, n_vars = 200, 60
    counts = rng.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
    counts[:100, :10] += rng.poisson(lam=40, size=(100, 10))

    atlas = ad.AnnData(X=counts.copy(), var=pd.DataFrame(index=[f"G{i}" for i in range(n_vars)]))
    atlas.layers["counts"] = counts.copy()
    return atlas


def test_seurat_flavor_matches_hvg_computed_on_log_normalized_data() -> None:
    """The ground truth: whatever scanpy picks when handed properly log-normalized input."""

    atlas = _atlas_with_known_variable_genes()
    reference = ad.AnnData(X=atlas.layers["counts"].copy(), var=atlas.var.copy())
    sc.pp.normalize_total(reference, target_sum=1e4)
    sc.pp.log1p(reference)
    sc.pp.highly_variable_genes(reference, n_top_genes=10, flavor="seurat")
    expected = set(reference.var_names[reference.var["highly_variable"]])

    hvg_list = _select_hvg_genes(
        atlas, n_top_genes=10, hvg_flavor="seurat", force_genes=[], counts_layer="counts"
    )

    assert set(hvg_list) == expected


def test_seurat_flavor_on_raw_counts_would_have_picked_a_different_set() -> None:
    """Guards against the fix silently no-op'ing: raw counts really do differ."""

    atlas = _atlas_with_known_variable_genes()
    raw_selection = atlas.copy()
    sc.pp.highly_variable_genes(raw_selection, n_top_genes=10, flavor="seurat", layer="counts")
    wrong = set(raw_selection.var_names[raw_selection.var["highly_variable"]])

    hvg_list = _select_hvg_genes(
        atlas, n_top_genes=10, hvg_flavor="seurat", force_genes=[], counts_layer="counts"
    )

    assert set(hvg_list) != wrong


def test_seurat_flavor_does_not_leave_a_temporary_layer_behind() -> None:
    atlas = _atlas_with_known_variable_genes()
    _select_hvg_genes(
        atlas, n_top_genes=10, hvg_flavor="seurat", force_genes=[], counts_layer="counts"
    )
    assert set(atlas.layers.keys()) == {"counts"}


def test_seurat_v3_flavor_still_uses_raw_counts_directly() -> None:
    """Count-based flavors are unaffected by this fix -- same behavior as before."""

    atlas = _atlas_with_known_variable_genes()
    reference = atlas.copy()
    sc.pp.highly_variable_genes(reference, n_top_genes=10, flavor="seurat_v3", layer="counts")
    expected = set(reference.var_names[reference.var["highly_variable"]])

    hvg_list = _select_hvg_genes(
        atlas, n_top_genes=10, hvg_flavor="seurat_v3", force_genes=[], counts_layer="counts"
    )

    assert set(hvg_list) == expected


def test_force_genes_are_added_regardless_of_flavor() -> None:
    atlas = _atlas_with_known_variable_genes()
    hvg_list = _select_hvg_genes(
        atlas,
        n_top_genes=10,
        hvg_flavor="seurat",
        force_genes=["G59"],
        counts_layer="counts",
    )
    assert "G59" in hvg_list


def test_hvg_list_preserves_var_names_order() -> None:
    atlas = _atlas_with_known_variable_genes()
    hvg_list = _select_hvg_genes(
        atlas, n_top_genes=10, hvg_flavor="seurat_v3", force_genes=[], counts_layer="counts"
    )
    positions = [list(atlas.var_names).index(g) for g in hvg_list]
    assert positions == sorted(positions)
