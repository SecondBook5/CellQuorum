"""Tests for MetacellsMethod (fake backend; no real MetaCell2 install)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.metacells.metacells_method import MetacellsMethod


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
