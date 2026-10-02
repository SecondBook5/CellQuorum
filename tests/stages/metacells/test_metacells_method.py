"""Tests for MetacellsMethod (fake backend; no real MetaCell2 install)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import cellquorum.stages.metacells.metacells_method as mm
from cellquorum.methods.base import MethodSkip
from cellquorum.stages.metacells.metacells_method import MetacellsMethod, _top_k_positive_edges


def _adata(
    n: int = 6000, g: int = 30, with_rep: bool = True, with_group: bool = True
) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    if with_group:
        obs["ref_cell_type_granular_coarse"] = [
            "Fibroblast" if i % 2 else "Endothelial" for i in range(n)
        ]
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = X.copy()
    if with_rep:
        a.obsm["X_pca_harmony"] = rng.normal(size=(n, 10)).astype("float32")
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(
        results=tmp_path / "res", scratch=tmp_path / "scr", figures=tmp_path / "figs"
    )
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    m = MetacellsMethod()
    res = m._run(_adata(n=10), {"min_cells": 5000}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_group_by_column_absent(tmp_path: Path) -> None:
    m = MetacellsMethod()
    res = m._run(
        _adata(with_group=False),
        {"group_by": "ref_cell_type_granular_coarse"},
        _ctx(tmp_path, object()),
    )
    assert isinstance(res, MethodSkip)
    assert "group_by" in res.reason.lower() or "absent" in res.reason.lower()


def test_skips_when_rep_key_absent(tmp_path: Path) -> None:
    m = MetacellsMethod()
    res = m._run(_adata(with_rep=False), {"rep_key": "X_pca_harmony"}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)
    assert "rep_key" in res.reason.lower() or "x_pca_harmony" in res.reason.lower()


def test_skips_when_launcher_missing(tmp_path: Path) -> None:
    m = MetacellsMethod()
    cfg = {"launcher": "definitely_not_a_real_launcher_xyz"}
    res = m._run(_adata(), cfg, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    m = MetacellsMethod()
    cfg = {"launcher": "python"}  # resolves on PATH
    res = m._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    cfg = {"launcher": "python"}
    m = MetacellsMethod()
    res = m._run(_adata(), cfg, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def _write_canned_outputs(out_dir: Path, adata: ad.AnnData) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = adata.n_obs
    metacell_ids = ["MC1"] * (n // 2) + ["MC2"] * (n - n // 2 - 2) + ["Outliers"] * 2
    groups = adata.obs["ref_cell_type_granular_coarse"].to_numpy()
    assignments = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "metacell_id": metacell_ids,
            "group": groups,
            "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
        }
    )
    assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)

    mc_obs = pd.DataFrame(index=["MC1", "MC2"])
    mc_var = pd.DataFrame(index=adata.var_names)
    mc_X = np.random.default_rng(1).normal(size=(2, adata.n_vars)).astype("float32")
    mc_adata = ad.AnnData(X=mc_X, obs=mc_obs, var=mc_var)
    mc_adata.write_h5ad(out_dir / "metacell_adata.h5ad")


def test_success_path_writes_metacell_id_and_artifacts(tmp_path: Path) -> None:
    adata = _adata()

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[1])
            _write_canned_outputs(out_dir, adata)
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    m = MetacellsMethod()
    cfg = {"launcher": "python", "group_by": "ref_cell_type_granular_coarse"}
    result = m._run(adata, cfg, _ctx(tmp_path, FakeBackend()))

    assert not isinstance(result, MethodSkip)
    assert "metacell_id" in result.adata.obs.columns
    assert set(result.adata.obs["metacell_id"].unique()) == {"MC1", "MC2", "Outliers"}
    artifact_names = {a.name for a in result.artifacts}
    assert "metacell_adata" in artifact_names
    assert "metacell_assignments" in artifact_names
    assert result.metrics["n_metacells"] == 2
    assert result.metrics["n_outlier_cells"] == 2
    assert result.metrics["group_by"] == "ref_cell_type_granular_coarse"


# ---------------------------------------------------------------------------
# Finding 1: numeric-looking cell_id CSV dtype inference silently voids the
# cell-to-metacell join. Reproduced with obs_names "1000".."1199".
# ---------------------------------------------------------------------------


def _numeric_id_adata(n: int = 200, g: int = 10) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(index=[str(1000 + i) for i in range(n)])
    obs["ref_cell_type_granular_coarse"] = [
        "Fibroblast" if i % 2 else "Endothelial" for i in range(n)
    ]
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = X.copy()
    a.obsm["X_pca_harmony"] = rng.normal(size=(n, 10)).astype("float32")
    return a


def test_numeric_looking_cell_ids_join_correctly(tmp_path: Path) -> None:
    """Before the dtype fix, pandas would infer cell_id as int64 for these
    numeric-looking barcodes, the join against str obs_names would silently
    match nothing, and every cell would end up "Outliers" with status
    success and no warning. With the fix, genuinely-matching numeric-looking
    IDs join correctly."""

    adata = _numeric_id_adata()
    n = adata.n_obs

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            metacell_ids = ["MC1"] * (n // 2) + ["MC2"] * (n - n // 2 - 2) + ["Outliers"] * 2
            assignments = pd.DataFrame(
                {
                    "cell_id": adata.obs_names,  # genuinely matching, numeric-looking
                    "metacell_id": metacell_ids,
                    "group": adata.obs["ref_cell_type_granular_coarse"].to_numpy(),
                    "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
                }
            )
            assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    m = MetacellsMethod()
    cfg = {"launcher": "python", "group_by": "ref_cell_type_granular_coarse", "min_cells": 100}
    result = m._run(adata, cfg, _ctx(tmp_path, FakeBackend()))

    assert not isinstance(result, MethodSkip)
    assert set(result.adata.obs["metacell_id"].unique()) == {"MC1", "MC2", "Outliers"}
    assert (result.adata.obs["metacell_id"] != "Outliers").sum() == n - 2


def test_coverage_guard_skips_when_assignment_ids_dont_match_adata(tmp_path: Path) -> None:
    """A fake backend that writes real (non-Outliers) assignments for cell_id
    values completely disjoint from adata.obs_names -- the join would
    silently produce 100% "Outliers" without the guard. The guard must
    detect this and skip loudly instead."""

    adata = _numeric_id_adata()
    n = adata.n_obs

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            fake_ids = [str(9000 + i) for i in range(n)]  # disjoint from adata.obs_names
            metacell_ids = ["MC1"] * (n // 2) + ["MC2"] * (n - n // 2)
            assignments = pd.DataFrame(
                {
                    "cell_id": fake_ids,
                    "metacell_id": metacell_ids,
                    "group": adata.obs["ref_cell_type_granular_coarse"].to_numpy(),
                    "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
                }
            )
            assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    m = MetacellsMethod()
    cfg = {"launcher": "python", "group_by": "ref_cell_type_granular_coarse", "min_cells": 100}
    result = m._run(adata, cfg, _ctx(tmp_path, FakeBackend()))

    assert isinstance(result, MethodSkip)
    assert "match" in result.reason.lower() or "join" in result.reason.lower()


# ---------------------------------------------------------------------------
# Finding 2: metacell-network edges must be thresholded (top-k positive per
# node), not the full unthresholded O(n^2) pairwise correlation graph.
# ---------------------------------------------------------------------------


def test_top_k_positive_edges_drops_nonpositive_and_caps_per_node() -> None:
    labels = [f"MC{i}" for i in range(6)]
    vals = np.array(
        [
            [1.0, 0.9, -0.5, 0.2, 0.8, -0.1],
            [0.9, 1.0, 0.3, -0.2, 0.1, 0.05],
            [-0.5, 0.3, 1.0, 0.7, -0.3, 0.6],
            [0.2, -0.2, 0.7, 1.0, 0.4, 0.5],
            [0.8, 0.1, -0.3, 0.4, 1.0, 0.15],
            [-0.1, 0.05, 0.6, 0.5, 0.15, 1.0],
        ]
    )
    corr = pd.DataFrame(vals, index=labels, columns=labels)

    edges = _top_k_positive_edges(corr, k=2)

    assert (edges["weight"] > 0).all()
    assert not (edges["source"] == edges["target"]).any()
    assert len(edges) <= 2 * len(labels)
    # No undirected pair double-counted.
    pairs = list(zip(edges["source"], edges["target"], strict=False))
    seen = {frozenset(p) for p in pairs}
    assert len(seen) == len(pairs)
    # MC0's strongest two positive correlations are MC1 (0.9) and MC4 (0.8).
    assert {"MC0", "MC1"} in seen
    assert {"MC0", "MC4"} in seen


def test_top_k_positive_edges_empty_when_no_positive_correlation() -> None:
    labels = ["MC0", "MC1", "MC2"]
    corr = pd.DataFrame(
        [[1.0, -0.5, -0.2], [-0.5, 1.0, -0.3], [-0.2, -0.3, 1.0]], index=labels, columns=labels
    )
    edges = _top_k_positive_edges(corr, k=2)
    assert edges.empty


def test_method_success_path_thresholds_network_edges(tmp_path: Path, monkeypatch: object) -> None:
    """Full success-path regression: with enough metacells to have a real
    correlation structure, the edges DataFrame handed to plot_metacell_network
    must be far smaller than the full n*(n-1)/2 pairwise graph and must
    contain no negative-weight rows."""

    n_mc, g = 25, 40
    adata = _adata(n=6000, g=g)
    rng = np.random.default_rng(3)
    mc_ids = [f"MC{i}" for i in range(n_mc)]

    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            n = adata.n_obs
            metacell_ids = np.resize(np.array(mc_ids), n).tolist()
            assignments = pd.DataFrame(
                {
                    "cell_id": adata.obs_names,
                    "metacell_id": metacell_ids,
                    "group": adata.obs["ref_cell_type_granular_coarse"].to_numpy(),
                    "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
                }
            )
            assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)
            mc_obs = pd.DataFrame(index=mc_ids)
            mc_var = pd.DataFrame(index=adata.var_names)
            mc_X = rng.normal(size=(n_mc, g)).astype("float32")
            ad.AnnData(X=mc_X, obs=mc_obs, var=mc_var).write_h5ad(out_dir / "metacell_adata.h5ad")
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    captured: dict = {}

    def fake_plot_network(edges, nodes, out_dir):  # noqa: ANN001
        captured["edges"] = edges
        return []

    monkeypatch.setattr(mm, "plot_metacell_network", fake_plot_network)

    m = MetacellsMethod()
    cfg = {"launcher": "python", "group_by": "ref_cell_type_granular_coarse"}
    result = m._run(adata, cfg, _ctx(tmp_path, FakeBackend()))

    assert not isinstance(result, MethodSkip)
    edges = captured["edges"]
    assert (edges["weight"] > 0).all()
    full_pairwise = n_mc * (n_mc - 1) // 2
    assert len(edges) <= 8 * n_mc
    assert len(edges) < full_pairwise


# ---------------------------------------------------------------------------
# Finding 3 (sparse half): mc_adata.X may be scipy.sparse; expr construction
# must densify it rather than assume a dense array.
# ---------------------------------------------------------------------------


def test_sparse_mc_adata_x_produces_same_expr_as_dense(tmp_path: Path, monkeypatch: object) -> None:
    adata = _adata()
    g = adata.n_vars
    rng = np.random.default_rng(2)
    dense_mc_X = rng.normal(size=(2, g)).astype("float32")
    mc_ids = ["MC1", "MC2"]

    def _make_backend(*, sparse: bool):
        class FakeBackend:
            def _py_module_available(self, _m):  # noqa: ANN001
                return True

            def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
                out_dir = Path(args[1])
                out_dir.mkdir(parents=True, exist_ok=True)
                n = adata.n_obs
                metacell_ids = ["MC1"] * (n // 2) + ["MC2"] * (n - n // 2)
                assignments = pd.DataFrame(
                    {
                        "cell_id": adata.obs_names,
                        "metacell_id": metacell_ids,
                        "group": adata.obs["ref_cell_type_granular_coarse"].to_numpy(),
                        "n_umis": np.asarray(adata.layers["counts"].sum(axis=1)).ravel(),
                    }
                )
                assignments.to_csv(out_dir / "metacell_assignments.csv", index=False)
                mc_obs = pd.DataFrame(index=mc_ids)
                mc_var = pd.DataFrame(index=adata.var_names)
                X = sp.csr_matrix(dense_mc_X) if sparse else dense_mc_X.copy()
                ad.AnnData(X=X, obs=mc_obs, var=mc_var).write_h5ad(out_dir / "metacell_adata.h5ad")
                return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

        return FakeBackend()

    captured: dict = {}

    def fake_heatmap(expr, nodes, out_dir, **kwargs):  # noqa: ANN001
        captured[len(captured)] = expr
        return []

    monkeypatch.setattr(mm, "plot_metacell_gene_heatmap", fake_heatmap)

    cfg = {"launcher": "python", "group_by": "ref_cell_type_granular_coarse"}
    MetacellsMethod()._run(adata, cfg, _ctx(tmp_path, _make_backend(sparse=False)))
    MetacellsMethod()._run(adata, cfg, _ctx(tmp_path, _make_backend(sparse=True)))

    dense_expr, sparse_expr = captured[0], captured[1]
    pd.testing.assert_frame_equal(dense_expr, sparse_expr)
