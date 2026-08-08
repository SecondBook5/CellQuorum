# tests/test_enrichment_gsva_method.py
from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.enrichment.gsva_method import GsvaMethod
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
    donors = ["d1", "d2", "d3", "d4", "d5", "d6"]
    rows, blocks = [], []
    for i, d in enumerate(donors):
        cond = "Normal" if i < 3 else "Disease"
        for _ in range(20):
            blocks.append(rng.poisson(5, size=10).astype(float))
            rows.append({"patient_id": d, "condition": cond, "cell_type": "T0"})
    X = np.vstack(blocks)
    a = ad.AnnData(X=X, obs=pd.DataFrame(rows))
    a.var_names = [f"G{i}" for i in range(10)]
    a.layers["counts"] = X.copy()  # raw counts for pseudobulk aggregation
    return a


def _install_fake_decoupler(monkeypatch):
    dc = types.ModuleType("decoupler")

    def gsva(data, net, tmin=5, **kw):
        srcs = list(pd.unique(net["source"]))
        idx = list(data.index)
        rng = np.random.default_rng(1)
        es = pd.DataFrame(rng.normal(size=(len(idx), len(srcs))), index=idx, columns=srcs)
        return es, None

    dc.mt = types.SimpleNamespace(gsva=gsva)
    dc.op = types.SimpleNamespace(
        hallmark=lambda **kw: pd.DataFrame(
            {"source": ["S0"] * 5, "target": [f"G{i}" for i in range(5)]}
        )
    )
    dc.pp = types.SimpleNamespace(read_gmt=lambda p: pd.DataFrame({"source": [], "target": []}))
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def test_gsva_skips_when_no_case_control(tmp_path):
    out = GsvaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)


def test_gsva_runs_and_writes_csv(tmp_path, monkeypatch):
    _install_fake_decoupler(monkeypatch)
    cfg = {
        "gene_set_collections": ["hallmark"],
        "condition_col": "condition",
        "donor_col": "patient_id",
        "case": "Disease",
        "control": "Normal",
        "paired": False,
        "min_size": 1,
        "counts_layer": "counts",
    }
    out = GsvaMethod()._run(_adata(), cfg, _Ctx(tmp_path))
    assert not isinstance(out, MethodSkip)
    scores = tmp_path / "results" / "enrichment_gsva_scores_hallmark.csv"
    contrast = tmp_path / "results" / "enrichment_gsva_contrast_hallmark.csv"
    assert scores.exists() and contrast.exists()
    cdf = pd.read_csv(contrast)
    assert list(cdf.columns) == [
        "source",
        "case_mean",
        "control_mean",
        "statistic",
        "pvalue",
        "padj",
        "collection",
    ]
