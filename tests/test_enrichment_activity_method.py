# tests/test_enrichment_activity_method.py
from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.enrichment.activity_method import ActivityMethod
from cellquorum.methods.base import MethodSkip


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata():
    rng = np.random.default_rng(0)
    n = 20
    X = rng.normal(size=(n, 8))
    obs = pd.DataFrame({"cell_type": (["T0"] * 10) + (["T1"] * 10)})
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f"G{i}" for i in range(8)]
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def _install_fake_decoupler(monkeypatch):
    dc = types.ModuleType("decoupler")

    def ulm(data, net, tmin=5, **kw):
        srcs = list(pd.unique(net["source"]))
        idx = list(data.index)
        rng = np.random.default_rng(2)
        es = pd.DataFrame(rng.normal(size=(len(idx), len(srcs))), index=idx, columns=srcs)
        pv = pd.DataFrame(np.full((len(idx), len(srcs)), 0.1), index=idx, columns=srcs)
        return es, pv

    dc.mt = types.SimpleNamespace(ulm=ulm)
    dc.op = types.SimpleNamespace(
        collectri=lambda **kw: pd.DataFrame(
            {"source": ["TF0"] * 4, "target": [f"G{i}" for i in range(4)], "weight": [1.0] * 4}
        )
    )
    dc.pp = types.SimpleNamespace(read_gmt=lambda p: pd.DataFrame({"source": [], "target": []}))
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def test_activity_runs_and_aggregates_per_celltype(tmp_path, monkeypatch):
    _install_fake_decoupler(monkeypatch)
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
        "min_size": 1,
    }
    out = ActivityMethod()._run(_adata(), cfg, _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_activity_collectri.csv")
    assert list(df.columns) == ["cell_type", "source", "mean_score"]
    assert set(df["cell_type"]) == {"T0", "T1"}


def test_activity_skips_when_decoupler_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "decoupler", None)
    cfg = {
        "activity_resources": ["collectri"],
        "cell_type_col": "cell_type",
        "layer": "cellquorum_normalized",
    }
    out = ActivityMethod()._run(_adata(), cfg, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)
