"""GPCCA estimator-chain tests for CellRank (runs on real cellrank 2.x)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from cellquorum.stages.trajectory import _cellrank

cr = pytest.importorskip("cellrank")


def _make_adata(n=300):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, 60)).astype("float32")
    a = ad.AnnData(X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(60)]
    a.obs["cell_type"] = pd.Categorical(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    a.obs["pseudotime"] = np.linspace(0, 1, n) + rng.normal(0, 0.01, n)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=20)
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
    return a


def _kernel(a):
    k, _ = _cellrank.build_kernel(
        a,
        pseudotime_key="pseudotime",
        cytotrace_key=None,
        use_rep=None,
        use_rep_fallback=["X_pca"],
        n_neighbors=15,
        weight_connectivities=0.2,
        seed=0,
    )
    return k


def test_run_gpcca_full_chain():
    a = _make_adata()
    res = _cellrank.run_gpcca(
        a,
        _kernel(a),
        cluster_key="cell_type",
        n_components=10,
        n_states=3,
        n_terminal_states=2,
        terminal_method="stability",
        predict_initial_states=False,
        n_initial_states=1,
        seed=0,
    )
    assert res["n_macrostates_actual"] >= 1
    assert isinstance(res["macrostate_names"], list)
    assert res["fate_prob"] is not None
    assert res["fate_prob"].shape[0] == a.n_obs
    assert res["fate_prob"].shape[1] == len(res["fate_names"])
    # Fate probability rows ~sum to 1.
    assert np.allclose(res["fate_prob"].sum(axis=1), 1.0, atol=1e-2)
    assert isinstance(res["drivers"], pd.DataFrame)
    # cellrank wrote its fixed forward keys onto the object.
    assert "macrostates_fwd" in a.obs
    assert "term_states_fwd" in a.obs
    assert "lineages_fwd" in a.obsm


def test_run_gpcca_coerces_non_categorical_cluster_key():
    a = _make_adata()
    a.obs["cell_type"] = a.obs["cell_type"].astype(str)  # plain string → would crash
    res = _cellrank.run_gpcca(
        a,
        _kernel(a),
        cluster_key="cell_type",
        n_components=10,
        n_states=3,
        n_terminal_states=2,
        terminal_method="stability",
        predict_initial_states=False,
        n_initial_states=1,
        seed=0,
    )
    assert res["n_macrostates_actual"] >= 1


def test_run_gpcca_reads_back_actual_macrostate_count():
    a = _make_adata()
    # Request a tiny count; cellrank may return >= requested. Code must not
    # assume the requested value.
    res = _cellrank.run_gpcca(
        a,
        _kernel(a),
        cluster_key="cell_type",
        n_components=10,
        n_states=2,
        n_terminal_states=2,
        terminal_method="stability",
        predict_initial_states=False,
        n_initial_states=1,
        seed=0,
    )
    assert res["n_macrostates_requested"] == 2
    assert res["n_macrostates_actual"] == len(res["macrostate_names"])


def test_run_gpcca_schur_failure_raises_typed(monkeypatch):
    a = _make_adata()
    k = _kernel(a)

    class _Boom:
        def __init__(self, *args, **kwargs):
            pass

        def compute_schur(self, *args, **kwargs):
            raise RuntimeError("schur boom")

    monkeypatch.setattr(cr.estimators, "GPCCA", _Boom)
    with pytest.raises(_cellrank.SchurFailed):
        _cellrank.run_gpcca(
            a,
            k,
            cluster_key="cell_type",
            n_components=10,
            n_states=3,
            n_terminal_states=2,
            terminal_method="stability",
            predict_initial_states=False,
            n_initial_states=1,
            seed=0,
        )


def test_run_gpcca_macrostates_failure_raises_typed(monkeypatch):
    a = _make_adata()
    k = _kernel(a)

    class _MacroBoom:
        def __init__(self, *args, **kwargs):
            pass

        def compute_schur(self, *args, **kwargs):
            return None

        def compute_macrostates(self, *args, **kwargs):
            raise ValueError("cannot split complex-conjugate eigenvalue pair")

    monkeypatch.setattr(cr.estimators, "GPCCA", _MacroBoom)
    with pytest.raises(_cellrank.MacrostatesFailed) as excinfo:
        _cellrank.run_gpcca(
            a,
            k,
            cluster_key="cell_type",
            n_components=10,
            n_states=3,
            n_terminal_states=2,
            terminal_method="stability",
            predict_initial_states=False,
            n_initial_states=1,
            seed=0,
        )
    # MacrostatesFailed is a recoverable CellRankComputeError → MethodSkip.
    assert isinstance(excinfo.value, _cellrank.CellRankComputeError)
