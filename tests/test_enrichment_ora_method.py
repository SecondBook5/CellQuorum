# tests/test_enrichment_ora_method.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from cellquorum.enrichment.ora_method import OraMethod
from cellquorum.methods.base import MethodSkip

# ORA is now a direct hypergeometric test (no decoupler dc.mt.ora call), so only
# the network fetch is stubbed via get_net. The stub returns a REAL long-format
# net; the statistic is computed by the production code, not a constant.


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata():
    return ad.AnnData(X=np.random.default_rng(0).normal(size=(4, 3)))


def _write_de(tmp, genes, up_genes, down_genes):
    """DE table over `genes`; `up_genes`/`down_genes` are significant up/down."""
    (tmp / "results").mkdir(parents=True, exist_ok=True)
    rows = []
    for g in genes:
        if g in up_genes:
            lfc, fdr = 2.0, 0.001
        elif g in down_genes:
            lfc, fdr = -2.0, 0.001
        else:
            lfc, fdr = 0.0, 0.9
        rows.append({"gene": g, "logFC": lfc, "logCPM": 1, "F": 1, "PValue": 0.001, "FDR": fdr})
    pd.DataFrame(rows).to_csv(tmp / "results" / "de_pseudobulk_edger.csv", index=False)


def _patch_get_net(monkeypatch, net):
    monkeypatch.setattr(
        "cellquorum.enrichment.ora_method.get_net",
        lambda collection, **kw: net.copy(),
    )


def test_ora_skips_when_no_de_table(tmp_path):
    out = OraMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, _Ctx(tmp_path))
    assert isinstance(out, MethodSkip)


def test_ora_enriches_overlapping_set_only(tmp_path, monkeypatch):
    """C1 guard: the foreground set must drive the statistic against the tested-
    gene background — an ENRICHED set gets a small p, a disjoint set does not.

    The old code fed a 0/1 membership row to dc.mt.ora, which ranked the row and
    took the top 5%, then tested against a fixed n_bg=20000. That degenerate
    heuristic could not distinguish an overlapping set from a disjoint one on the
    foreground content; this test fails against it and passes with the direct
    hypergeometric over the real background.
    """
    # 30-gene universe; foreground (up) = SET_A exactly.
    genes = [f"g{i}" for i in range(30)]
    set_a = [f"g{i}" for i in range(6)]  # overlaps foreground fully
    set_b = [f"g{i}" for i in range(6, 12)]  # disjoint from foreground
    net = pd.DataFrame({"source": ["A"] * 6 + ["B"] * 6, "target": set_a + set_b})
    _write_de(tmp_path, genes, up_genes=set(set_a), down_genes=set())
    _patch_get_net(monkeypatch, net)

    out = OraMethod()._run(
        _adata(),
        {"gene_set_collections": ["hallmark"], "min_size": 3, "min_foreground_genes": 3},
        _Ctx(tmp_path),
    )
    assert not isinstance(out, MethodSkip)
    df = pd.read_csv(tmp_path / "results" / "enrichment_ora_hallmark.csv")
    assert list(df.columns) == [
        "source",
        "direction",
        "score",
        "pvalue",
        "padj",
        "significant",
        "collection",
        "count",
        "gene_ratio",
    ]

    up = df[df["direction"] == "up"].set_index("source")
    # Enriched set A: fully overlapping foreground → tiny p, significant.
    assert up.loc["A", "pvalue"] < 0.01
    assert bool(up.loc["A", "significant"])
    # Disjoint set B: no overlap → p == 1 (not significant).
    assert up.loc["B", "pvalue"] == 1.0
    assert not bool(up.loc["B", "significant"])
    # And A must be far more significant than B.
    assert up.loc["A", "pvalue"] < up.loc["B", "pvalue"]

    # Cross-check the reported raw p against a direct hypergeometric computation.
    expected_a = float(hypergeom.sf(6 - 1, 30, 6, 6))
    np.testing.assert_allclose(up.loc["A", "pvalue"], expected_a, rtol=0, atol=1e-9)


def test_ora_padj_is_single_bh_of_pvalue(tmp_path, monkeypatch):
    """I1 guard (ORA): padj is one BH pass over the raw hypergeometric p."""
    genes = [f"g{i}" for i in range(40)]
    fg = [f"g{i}" for i in range(8)]
    # Several sources with varying overlap so BH has work to do.
    rows = []
    for s in range(6):
        for t in genes[s * 4 : s * 4 + 8]:
            rows.append({"source": f"S{s}", "target": t})
    net = pd.DataFrame(rows)
    _write_de(tmp_path, genes, up_genes=set(fg), down_genes=set())
    _patch_get_net(monkeypatch, net)

    OraMethod()._run(
        _adata(),
        {"gene_set_collections": ["hallmark"], "min_size": 3, "min_foreground_genes": 3},
        _Ctx(tmp_path),
    )
    df = pd.read_csv(tmp_path / "results" / "enrichment_ora_hallmark.csv")
    up = df[df["direction"] == "up"]
    raw = up["pvalue"].to_numpy(dtype=float)
    padj = up["padj"].to_numpy(dtype=float)
    expected = multipletests(raw, method="fdr_bh")[1]
    np.testing.assert_allclose(np.sort(padj), np.sort(expected), rtol=0, atol=1e-9)
