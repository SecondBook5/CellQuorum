# tests/test_enrichment_ora_method.py
from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.enrichment.ora_method import OraMethod
from cellquorum.methods.base import MethodSkip


class _Paths:
    def __init__(self, tmp):
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata():
    return ad.AnnData(X=np.random.default_rng(0).normal(size=(4, 3)))


def _write_de(tmp):
    # 8 genes; 3 clearly up, 2 clearly down
    (tmp / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(8)],
            "logFC": [2, 2, 2, -2, -2, 0.0, 0.1, -0.1],
            "logCPM": [1] * 8,
            "F": [1] * 8,
            "PValue": [0.001] * 8,
            "FDR": [0.001, 0.001, 0.001, 0.001, 0.001, 0.9, 0.9, 0.9],
        }
    ).to_csv(tmp / "results" / "de_pseudobulk_edger.csv", index=False)


def _install_fake_decoupler(monkeypatch):
    dc = types.ModuleType("decoupler")

    def ora(members, net, tmin=5, **kw):
        srcs = list(pd.unique(net["source"]))
        idx = list(members.index)
        es = pd.DataFrame(np.ones((len(idx), len(srcs))), index=idx, columns=srcs)
        pv = pd.DataFrame(np.full((len(idx), len(srcs)), 0.02), index=idx, columns=srcs)
        return es, pv

    dc.mt = types.SimpleNamespace(ora=ora)
    dc.op = types.SimpleNamespace(
        hallmark=lambda **kw: pd.DataFrame({"source": ["S0"] * 3, "target": ["G0", "G1", "G2"]})
    )
    dc.pp = types.SimpleNamespace(read_gmt=lambda p: pd.DataFrame({"source": [], "target": []}))
    monkeypatch.setitem(sys.modules, "decoupler", dc)


def test_ora_skips_when_no_de_table(tmp_path):
    out = OraMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)


def test_ora_runs_and_writes_csv(tmp_path, monkeypatch):
    _write_de(tmp_path)
    _install_fake_decoupler(monkeypatch)
    out = OraMethod()._run(
        _adata(),
        {"gene_set_collections": ["hallmark"], "min_size": 1, "min_foreground_genes": 1},
        _Ctx(tmp_path),
    )
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_ora_hallmark.csv")
    assert list(df.columns) == ["source", "direction", "score", "pvalue", "padj", "collection"]
    assert set(df["direction"]) <= {"up", "down"}
