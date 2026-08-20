"""The ORA CSV must carry count (overlap) and gene_ratio (overlap / foreground)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.enrichment.ora_method import OraMethod


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


def _write_gmt(tmp, gmt_data):
    """Write a simple GMT file from dict: {name: [genes]}."""
    gmt_path = tmp / "test.gmt"
    with open(gmt_path, "w") as f:
        for name, genes in gmt_data.items():
            f.write(f"{name}\t\t{chr(9).join(genes)}\n")
    return gmt_path


def _patch_get_net(monkeypatch, gmt_path):
    """Patch get_net to read from local GMT instead of fetching."""

    def mock_get_net(collection, **kw):
        # Parse GMT file and return long-format dataframe
        rows = []
        with open(gmt_path) as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) >= 3:
                    source = parts[0]
                    targets = parts[2:]
                    for target in targets:
                        rows.append({"source": source, "target": target})
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "cellquorum.enrichment.ora_method.get_net",
        mock_get_net,
    )


def test_ora_csv_has_count_and_gene_ratio(tmp_path, monkeypatch):
    """count (overlap size) and gene_ratio (overlap / foreground) columns are present and valid."""
    # Create a minimal universe: 20 genes total
    genes = [f"g{i}" for i in range(20)]
    # Foreground (up): genes 0-5
    up_genes = set(genes[:6])
    # Write DE table
    _write_de(tmp_path, genes, up_genes=up_genes, down_genes=set())

    # Create test gene sets
    # Set A: overlaps foreground fully (0-3)
    # Set B: partial overlap (2-5)
    # Set C: no overlap (10-13)
    gmt_data = {
        "SetA": genes[0:4],
        "SetB": genes[2:6],
        "SetC": genes[10:14],
    }
    gmt_path = _write_gmt(tmp_path, gmt_data)
    _patch_get_net(monkeypatch, gmt_path)

    # Run ORA
    OraMethod()._run(
        _adata(),
        {
            "gene_set_collections": ["test"],
            "gmt_path": str(gmt_path),
            "min_size": 1,
            "min_foreground_genes": 1,
        },
        _Ctx(tmp_path),
    )

    # Read result CSV
    csv_path = tmp_path / "results" / "enrichment_ora_test.csv"
    df = pd.read_csv(csv_path)

    # Check columns exist
    assert "count" in df.columns
    assert "gene_ratio" in df.columns

    # Check column order: first 7 are standard, then count, gene_ratio
    assert list(df.columns)[:7] == [
        "source",
        "direction",
        "score",
        "pvalue",
        "padj",
        "significant",
        "collection",
    ]
    assert list(df.columns)[7:] == ["count", "gene_ratio"]

    # Check count is nonnegative integer
    assert (df["count"] >= 0).all()
    assert all(isinstance(x, int | np.integer) for x in df["count"])

    # Check gene_ratio in [0, 1]
    assert ((df["gene_ratio"] >= 0) & (df["gene_ratio"] <= 1)).all()

    # Spot-check: SetA overlaps foreground (g0-g3) with foreground (g0-g5)
    # count should be 4, gene_ratio should be 4/6 ≈ 0.667
    set_a_row = df[df["source"] == "SetA"]
    assert len(set_a_row) == 1
    assert set_a_row.iloc[0]["count"] == 4
    np.testing.assert_allclose(set_a_row.iloc[0]["gene_ratio"], 4.0 / 6.0, rtol=1e-9)

    # Spot-check: SetB overlaps foreground (g2-g5) with foreground (g0-g5)
    # count should be 4, gene_ratio should be 4/6 ≈ 0.667
    set_b_row = df[df["source"] == "SetB"]
    assert len(set_b_row) == 1
    assert set_b_row.iloc[0]["count"] == 4
    np.testing.assert_allclose(set_b_row.iloc[0]["gene_ratio"], 4.0 / 6.0, rtol=1e-9)

    # Spot-check: SetC has no overlap with foreground
    # count should be 0, gene_ratio should be 0
    set_c_row = df[df["source"] == "SetC"]
    assert len(set_c_row) == 1
    assert set_c_row.iloc[0]["count"] == 0
    assert set_c_row.iloc[0]["gene_ratio"] == 0.0
