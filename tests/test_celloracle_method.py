"""Tests for the CellOracle perturbation method (fake backend; no real celloracle)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.perturbation.celloracle_method import CellOracleMethod


def _adata(n: int = 300, g: int = 40, condition: bool = False) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = {"cell_type": ["A" if i % 2 else "B" for i in range(n)]}
    if condition:
        obs["condition"] = ["disease" if i % 2 else "healthy" for i in range(n)]
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(obs, index=[f"cell_{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(g)]),
    )
    a.layers["counts"] = X.copy()
    a.obsm["X_umap"] = rng.random((n, 2)).astype("float32")
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(results=tmp_path / "res", scratch=tmp_path / "scr")
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(_adata(n=10), {"min_cells_total": 200}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    cfg = {"launcher": "python"}  # resolves on PATH
    res = CellOracleMethod()._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _s, _a, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    res = CellOracleMethod()._run(_adata(), {"launcher": "python"}, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_input_contract_does_not_require_condition_or_cluster() -> None:
    m = CellOracleMethod()
    contract = m.input_contract({})
    assert contract.required_obs == []
    assert m.requires_obs({}) == []


def _fake_success_backend(direction: str):
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, args, *, timeout=None):  # noqa: ANN001, ANN003
            out_dir = Path(args[args.index("--out-dir") + 1])
            tag = args[args.index("--tag") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "tf": ["PROX1", "PIEZO1"],
                    "score": [0.9, 0.4],
                    "n_cells": [300, 300],
                    "direction": [direction, direction],
                }
            ).to_csv(out_dir / "perturbation_ranking.csv", index=False)
            # a shift-vector parquet so the shift-field figure has input
            idx = [f"cell_{i}" for i in range(300)]
            pd.DataFrame(
                {"dx": np.random.rand(300), "dy": np.random.rand(300)}, index=idx
            ).to_parquet(out_dir / "shift_vectors_PROX1.parquet")
            _ = tag
            return SimpleNamespace(returncode=0, stderr="")

    return FakeBackend()


def test_success_directional_builds_artifacts_and_metrics(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(condition=True),
        {"launcher": "python", "condition_key": "condition", "healthy_label": "healthy"},
        _ctx(tmp_path, _fake_success_backend("directional")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["n_tfs_screened"] == 2
    assert res.metrics["condition_scored"] is True
    assert res.metrics["cluster_key"] == "cell_type"
    assert res.metrics["n_obs"] == 300
    names = {a.name for a in res.artifacts}
    assert "ranking" in names


def test_success_magnitude_when_no_condition(tmp_path: Path) -> None:
    res = CellOracleMethod()._run(
        _adata(),
        {"launcher": "python"},
        _ctx(tmp_path, _fake_success_backend("magnitude")),
    )
    assert not isinstance(res, MethodSkip)
    assert res.metrics["condition_scored"] is False
