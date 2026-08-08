# tests/test_enrichment_gsea_method.py
from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.enrichment.gsea_method import GseaMethod
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
    return ad.AnnData(X=rng.normal(size=(6, 5)))


def _install_fake_decoupler(monkeypatch):
    dc = types.ModuleType("decoupler")

    def gsea(rank, net, tmin=5, times=1000, seed=42, **kw):
        srcs = list(pd.unique(net["source"]))
        es = pd.DataFrame([[0.5] * len(srcs)], index=["contrast"], columns=srcs)
        pv = pd.DataFrame([[0.01] * len(srcs)], index=["contrast"], columns=srcs)
        return es, pv

    dc.mt = types.SimpleNamespace(gsea=gsea)
    dc.op = types.SimpleNamespace(
        hallmark=lambda **kw: pd.DataFrame({"source": ["S0", "S0"], "target": ["A", "B"]}),
        resource=lambda name, **kw: pd.DataFrame({"source": ["R0"], "target": ["A"]}),
    )
    dc.pp = types.SimpleNamespace(read_gmt=lambda p: pd.DataFrame({"source": [], "target": []}))
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def _write_de(tmp):
    pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "logFC": [2.0, -1.0, 0.5],
            "logCPM": [1, 1, 1],
            "F": [1, 1, 1],
            "PValue": [0.01, 0.001, 0.1],
            "FDR": [0.02, 0.01, 0.2],
        }
    ).to_csv(tmp / "results" / "de_pseudobulk_edger.csv", index=False)


def test_gsea_skips_when_no_de_table(tmp_path):
    ctx = _Ctx(tmp_path)
    out = GseaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, ctx)
    assert isinstance(out, MethodSkip)
    assert "de results" in out.reason.lower()


def test_gsea_runs_and_writes_csv(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    _write_de(tmp_path)
    _install_fake_decoupler(monkeypatch)
    out = GseaMethod()._run(
        _adata(), {"gene_set_collections": ["hallmark"], "seed": 42, "min_size": 1}, ctx
    )
    assert not isinstance(out, MethodSkip)
    csv = tmp_path / "results" / "enrichment_gsea_hallmark.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == ["source", "score", "pvalue", "padj", "collection"]


def test_gsea_skips_when_decoupler_absent(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    _write_de(tmp_path)
    monkeypatch.setitem(sys.modules, "decoupler", None)
    out = GseaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, ctx)
    assert isinstance(out, MethodSkip)
