# tests/stages/comparative/enrichment/test_enrichment_gsea_method.py
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.enrichment.gsea_method import GseaMethod

# These tests exercise the REAL decoupler package (gsea via its low-level
# building blocks) — never a constant stub — so the I1 fix (single, correctly
# labelled FDR) is genuinely guarded. Only the network fetch is stubbed, via
# get_net, to keep the test offline and deterministic.
dc = pytest.importorskip("decoupler")


class _Paths:
    def __init__(self, tmp):
        self.root = tmp
        self.results = tmp / "results"
        self.results.mkdir(parents=True, exist_ok=True)


class _Ctx:
    def __init__(self, tmp):
        self.paths = _Paths(tmp)


def _adata():
    rng = np.random.default_rng(0)
    return ad.AnnData(X=rng.normal(size=(6, 5)))


def _big_net():
    """Multi-source net over a large gene universe so BH has several sources to
    correct across (a single source makes double-correction invisible)."""
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(200)]
    rows = []
    for s in range(12):
        for t in rng.choice(genes, 15, replace=False):
            rows.append({"source": f"S{s}", "target": t})
    return genes, pd.DataFrame(rows)


def _write_de(tmp, genes):
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {
            "gene": genes,
            "logFC": rng.normal(size=len(genes)),
            "logCPM": [1] * len(genes),
            "F": [1] * len(genes),
            "PValue": rng.uniform(1e-4, 0.9, size=len(genes)),
            "FDR": rng.uniform(1e-3, 0.95, size=len(genes)),
        }
    ).to_csv(tmp / "results" / "de_pseudobulk_edger.csv", index=False)


def _patch_get_net(monkeypatch, net):
    monkeypatch.setattr(
        "cellquorum.stages.comparative.enrichment.gsea_method.get_net",
        lambda collection, **kw: net.copy(),
    )


def test_gsea_skips_when_no_de_table(tmp_path):
    ctx = _Ctx(tmp_path)
    out = GseaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, ctx)
    assert isinstance(out, MethodSkip)
    assert "de results" in out.reason.lower()


def test_gsea_runs_and_writes_csv(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    genes, net = _big_net()
    _write_de(tmp_path, genes)
    _patch_get_net(monkeypatch, net)
    out = GseaMethod()._run(
        _adata(),
        {"gene_set_collections": ["hallmark"], "seed": 42, "min_size": 5, "gsea_permutations": 200},
        ctx,
    )
    assert not isinstance(out, MethodSkip)
    csv = tmp_path / "results" / "enrichment_gsea_hallmark.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == [
        "source",
        "score",
        "es",
        "pvalue",
        "padj",
        "significant",
        "p_at_resolution_limit",
        "p_resolution_limit",
        "permutations",
        "collection",
        "set_size",
        "leading_edge_size",
        "leading_edge",
    ]

    _assert_leading_edge_genes(df, genes, net)
    _assert_raw_and_normalized_scores(df)
    _assert_single_bh_adjustment(df.set_index("source"), ctx, net)


def test_a_permutation_p_is_never_reported_as_zero(tmp_path, monkeypatch):
    """Cover both legacy zeros and corrected backend p-values at a known boundary."""
    import decoupler as dc

    actual_gsea = dc.mt.gsea.func

    def boundary_pvalues(*args, **kwargs):
        scores, pvalues = actual_gsea(*args, **kwargs)
        pvalues = pvalues.astype(float, copy=True)
        pvalues[0, :3] = [0.0, 1 / 11, 0.5]
        return scores, pvalues

    monkeypatch.setattr(dc.mt.gsea, "func", boundary_pvalues)
    ctx = _Ctx(tmp_path)
    genes, net = _big_net()
    _write_de(tmp_path, genes)
    _patch_get_net(monkeypatch, net)
    GseaMethod()._run(
        _adata(),
        {"gene_set_collections": ["hallmark"], "seed": 42, "min_size": 5, "gsea_permutations": 10},
        ctx,
    )
    df = pd.read_csv(tmp_path / "results" / "enrichment_gsea_hallmark.csv")

    assert (df["pvalue"] > 0).all(), "a p-value of exactly zero is not a p-value"
    assert (df["padj"] > 0).all()
    assert np.allclose(df["p_resolution_limit"], 1.0 / 11)
    assert (df["permutations"] == 10).all()
    # The flag is what keeps "p is at the limit" distinguishable from "p was measured this
    # small", which the floored value alone cannot say.
    at_limit = df["p_at_resolution_limit"].to_numpy(dtype=bool)
    assert at_limit.sum() >= 2
    assert 0.5 in df.loc[~at_limit, "pvalue"].to_numpy()
    assert np.allclose(df.loc[at_limit, "pvalue"], 1.0 / 11)
    assert (df.loc[~at_limit, "pvalue"] >= 1.0 / 11).all()


def _assert_leading_edge_genes(df, genes, net):
    assert df["leading_edge"].notna().all()
    assert (df["set_size"] > 0).all()
    universe = set(genes)
    by_source = net.groupby("source")["target"].agg(set)
    for row in df.itertuples():
        edge = str(row.leading_edge).split(";")
        assert len(edge) == row.leading_edge_size
        # The leading edge is a subset of the set, and never larger than it.
        assert set(edge) <= by_source[row.source] & universe
        assert row.leading_edge_size <= row.set_size


def _assert_raw_and_normalized_scores(df):
    assert df["es"].notna().all()
    assert (df["es"].abs() <= 1.0).all(), "a raw enrichment score cannot exceed 1"
    # Normalization divides by the mean of the same-sign permutation scores, so the two
    # never disagree about direction. If they do, one of them is not what it claims to be.
    assert (np.sign(df["es"]) == np.sign(df["score"])).all()
    # And they are genuinely different numbers, not the same column twice.
    assert not np.allclose(df["es"], df["score"])


def _assert_single_bh_adjustment(df, ctx, net):
    # Need several sources for BH to actually do something.
    assert len(df) >= 5

    raw = df["pvalue"].to_numpy(dtype=float)
    padj = df["padj"].to_numpy(dtype=float)
    # Raw p-values are plausible permutation p-values in [0, 1], not all identical.
    assert np.all((raw >= 0) & (raw <= 1))
    assert raw.min() < raw.max()

    # (1) padj is exactly one BH pass over the reported pvalue.
    expected = multipletests(raw, method="fdr_bh")[1]
    np.testing.assert_allclose(np.sort(padj), np.sort(expected), rtol=0, atol=1e-9)

    # (2) padj equals the high-level dc.mt.gsea output (its single internal BH of
    # the raw p) — computed here with REAL decoupler on the same ranking/net.
    from cellquorum.stages.comparative.enrichment.ranking import de_table_to_ranking

    de = pd.read_csv(ctx.paths.results / "de_pseudobulk_edger.csv")
    ranking = de_table_to_ranking(de)
    _, pv_hl = dc.mt.gsea(ranking, net.copy(), tmin=5, times=200, seed=42)
    # decoupler's internal BH returns float32, so compare at float32 precision.
    dc_adj = pv_hl.loc["contrast"].reindex(df.index).to_numpy(dtype=float)
    np.testing.assert_allclose(padj, dc_adj, rtol=0, atol=1e-6)

    # (3) The reported pvalue is the RAW p, NOT the adjusted value: it must differ
    # from decoupler's adjusted output on at least some sources.
    assert not np.allclose(raw, dc_adj, atol=1e-9)


def test_gsea_skips_when_decoupler_absent(tmp_path, monkeypatch):
    ctx = _Ctx(tmp_path)
    genes, _ = _big_net()
    _write_de(tmp_path, genes)
    monkeypatch.setitem(sys.modules, "decoupler", None)
    out = GseaMethod()._run(_adata(), {"gene_set_collections": ["hallmark"]}, ctx)
    assert isinstance(out, MethodSkip)
