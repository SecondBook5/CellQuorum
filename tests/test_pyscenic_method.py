"""Tests for the pySCENIC GRN method (fake backend; no real pyscenic/DBs)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.grn.pyscenic_method import PyscenicMethod
from cellquorum.methods.base import MethodSkip


def _adata(n: int = 300, g: int = 40) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    obs = pd.DataFrame(
        {"cell_type": ["A" if i % 2 else "B" for i in range(n)]},
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(g)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = X.copy()
    return a


def _ctx(tmp_path: Path, backend):
    paths = SimpleNamespace(results=tmp_path / "res", scratch=tmp_path / "scr")
    reg = SimpleNamespace(get=lambda name: backend)
    return SimpleNamespace(paths=paths, backend_registry=reg, config=None)


def _db_config(tmp_path: Path) -> dict:
    # point the DB paths at real existing files so the pre-subprocess gate passes
    tfs = tmp_path / "tfs.txt"
    tfs.write_text("STAT1\n")
    motifs = tmp_path / "motifs.tbl"
    motifs.write_text("x\n")
    feather = tmp_path / "rank.feather"
    feather.write_text("x\n")
    return {
        "tfs_path": str(tfs),
        "motifs_path": str(motifs),
        "rankings_glob": str(feather),
    }


def test_skips_when_too_few_cells(tmp_path: Path) -> None:
    m = PyscenicMethod()
    res = m._run(
        _adata(n=10), {"min_cells_total": 200, **_db_config(tmp_path)}, _ctx(tmp_path, object())
    )
    assert isinstance(res, MethodSkip)
    assert "too few cells" in res.reason.lower()


def test_skips_when_db_paths_unset(tmp_path: Path) -> None:
    m = PyscenicMethod()
    res = m._run(_adata(), {}, _ctx(tmp_path, object()))
    assert isinstance(res, MethodSkip)


def test_skips_when_backend_missing(tmp_path: Path) -> None:
    m = PyscenicMethod()
    # launcher present (monkeypatch shutil.which via config launcher that exists) but backend None
    cfg = {"launcher": "python", **_db_config(tmp_path)}  # 'python' resolves on PATH
    res = m._run(_adata(), cfg, _ctx(tmp_path, None))
    assert isinstance(res, MethodSkip)


def test_skips_on_timeout(tmp_path: Path) -> None:
    class FakeBackend:
        def _py_module_available(self, _m):  # noqa: ANN001
            return True

        def run_script(self, _script, _args, *, timeout=None):  # noqa: ANN001, ANN003
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    cfg = {"launcher": "python", **_db_config(tmp_path)}
    m = PyscenicMethod()
    res = m._run(_adata(), cfg, _ctx(tmp_path, FakeBackend()))
    assert isinstance(res, MethodSkip)


def test_input_contract_does_not_require_group_by() -> None:
    m = PyscenicMethod()
    contract = m.input_contract({})
    assert "cell_type" not in contract.required_obs
    assert m.requires_obs({}) == []
