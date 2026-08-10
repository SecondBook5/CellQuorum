"""Kernel construction + graceful-degradation tests for CellRank."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellquorum.trajectory import _cellrank

cr = pytest.importorskip("cellrank")


def _make_adata(n=200, with_graph=True, with_pt=True):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, 40)).astype("float32")
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(40)]
    a.obs["cell_type"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2))
    if with_pt:
        a.obs["pseudotime"] = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=20)
    if with_graph:
        sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    return a


def test_build_kernel_combines_pseudotime_and_connectivity():
    a = _make_adata()
    kernel, info = _cellrank.build_kernel(
        a,
        pseudotime_key="pseudotime",
        cytotrace_key=None,
        use_rep=None,
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        weight_connectivities=0.2,
        seed=0,
    )
    assert kernel is not None
    assert "pseudotime" in info["kernels"]
    assert "connectivity" in info["kernels"]
    assert info["weight_connectivities"] == pytest.approx(0.2)


def test_build_kernel_connectivity_only_when_no_pseudotime():
    a = _make_adata(with_pt=True)
    kernel, info = _cellrank.build_kernel(
        a,
        pseudotime_key=None,
        cytotrace_key=None,
        use_rep=None,
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        weight_connectivities=0.2,
        seed=0,
    )
    assert kernel is not None
    assert info["kernels"] == ["connectivity"]
    assert any("connectivity-only" in n for n in info["notes"])


def test_build_kernel_builds_graph_when_connectivities_absent():
    a = _make_adata(with_graph=False)
    assert "connectivities" not in a.obsp
    kernel, info = _cellrank.build_kernel(
        a,
        pseudotime_key="pseudotime",
        cytotrace_key=None,
        use_rep=None,
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        weight_connectivities=0.2,
        seed=0,
    )
    assert "connectivities" in a.obsp
    assert kernel is not None


def test_build_kernel_raises_no_kernel_input():
    a = _make_adata(with_graph=False)
    del a.obsm["X_pca"]  # no graph and no usable rep
    with pytest.raises(_cellrank.NoKernelInput):
        _cellrank.build_kernel(
            a,
            pseudotime_key="pseudotime",
            cytotrace_key=None,
            use_rep=None,
            use_rep_fallback=["X_pca"],
            n_neighbors=15,
            weight_connectivities=0.2,
            seed=0,
        )


def test_build_kernel_drops_pseudotime_when_all_nan():
    a = _make_adata()
    a.obs["pseudotime"] = np.nan
    kernel, info = _cellrank.build_kernel(
        a,
        pseudotime_key="pseudotime",
        cytotrace_key=None,
        use_rep=None,
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        weight_connectivities=0.2,
        seed=0,
    )
    assert info["kernels"] == ["connectivity"]
