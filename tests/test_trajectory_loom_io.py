# tests/test_trajectory_loom_io.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.trajectory._loom_io import (
    loom_cellid_to_bare,
    read_loom_layers,
    reconcile_looms,
)

loompy = pytest.importorskip("loompy")


def _write_loom(path, stem, barcodes, genes, seed):
    """Write a tiny velocyto-style loom: genes×cells, spliced/unspliced layers."""
    rng = np.random.default_rng(seed)
    n_genes, n_cells = len(genes), len(barcodes)
    main = rng.integers(0, 5, size=(n_genes, n_cells)).astype("float32")
    spliced = rng.integers(0, 5, size=(n_genes, n_cells)).astype("float32")
    unspliced = rng.integers(0, 3, size=(n_genes, n_cells)).astype("float32")
    row_attrs = {"Gene": np.array(genes, dtype=object)}
    col_attrs = {"CellID": np.array([f"{stem}:{bc}x" for bc in barcodes], dtype=object)}
    loompy.create(
        str(path),
        layers={"": main, "spliced": spliced, "unspliced": unspliced},
        row_attrs=row_attrs,
        col_attrs=col_attrs,
    )


def test_cellid_to_bare():
    assert (
        loom_cellid_to_bare("Set1_norm_LE:AAACGCTAGGAAGAAC x".replace(" ", ""))
        == "AAACGCTAGGAAGAAC-1"
    )
    assert loom_cellid_to_bare("stem:ACGTACGTACGTACGTx") == "ACGTACGTACGTACGT-1"


def test_read_loom_layers_shapes(tmp_path):
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    bcs = ["AAAA", "CCCC", "GGGG", "TTTT"]
    p = tmp_path / "s1.loom"
    _write_loom(p, "s1", bcs, genes, seed=0)
    out = read_loom_layers(p)
    assert out is not None
    spliced, unspliced, gene_names, cell_ids = out
    # cells×genes after transpose
    assert spliced.shape == (4, 3)
    assert unspliced.shape == (4, 3)
    assert list(gene_names) == genes
    assert cell_ids[0] == "s1:AAAAx"


def test_read_loom_layers_missing_file_returns_none(tmp_path):
    assert read_loom_layers(tmp_path / "does_not_exist.loom") is None


def _atlas(sample_ids, barcodes_per_sample, genes):
    """Build an atlas AnnData with obs_names '<sample>_<BARCODE>-1'."""
    names, samples = [], []
    for s in sample_ids:
        for bc in barcodes_per_sample:
            names.append(f"{s}_{bc}-1")
            samples.append(s)
    n = len(names)
    X = np.ones((n, len(genes)), dtype="float32")
    a = ad.AnnData(X=X, obs=pd.DataFrame({"sample_id": samples}, index=names))
    a.var_names = genes
    return a


def test_reconcile_maps_barcodes_and_reindexes_genes(tmp_path):
    # Atlas has GENE_A/B/C/D; loom only has A/C — D must reindex to 0.
    atlas = _atlas(["s1", "s2"], ["AAAA", "CCCC"], ["GENE_A", "GENE_B", "GENE_C", "GENE_D"])
    _write_loom(tmp_path / "s1.loom", "s1", ["AAAA", "CCCC"], ["GENE_A", "GENE_C"], seed=1)
    _write_loom(tmp_path / "s2.loom", "s2", ["AAAA", "CCCC"], ["GENE_A", "GENE_C"], seed=2)
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "loom_path": [str(tmp_path / "s1.loom"), str(tmp_path / "s2.loom")],
        }
    )
    out, notes = reconcile_looms(atlas, manifest, sample_col="sample_id", loom_path_col="loom_path")
    assert out is not None
    assert "spliced" in out.layers and "unspliced" in out.layers
    # reindexed onto the atlas's 4 genes
    assert out.layers["spliced"].shape[1] == 4
    # GENE_B and GENE_D (index 1 and 3) absent from loom → all-zero columns
    import scipy.sparse as sp

    dense = (
        out.layers["spliced"].toarray()
        if sp.issparse(out.layers["spliced"])
        else np.asarray(out.layers["spliced"])
    )
    assert dense[:, 1].sum() == 0
    assert dense[:, 3].sum() == 0
    # all 4 cells reconciled (2 samples × 2 barcodes), no cross-sample collision
    assert out.n_obs == 4


def test_reconcile_skips_corrupt_sample_keeps_others(tmp_path):
    atlas = _atlas(["s1", "s2"], ["AAAA", "CCCC"], ["GENE_A", "GENE_C"])
    _write_loom(tmp_path / "s1.loom", "s1", ["AAAA", "CCCC"], ["GENE_A", "GENE_C"], seed=1)
    corrupt = tmp_path / "s2.loom"
    corrupt.write_bytes(b"not a real loom file")
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "loom_path": [str(tmp_path / "s1.loom"), str(corrupt)],
        }
    )
    out, notes = reconcile_looms(atlas, manifest, sample_col="sample_id", loom_path_col="loom_path")
    assert out is not None
    assert out.n_obs == 2  # only s1's cells
    assert any("s2" in n for n in notes)


def test_reconcile_returns_none_when_no_overlap(tmp_path):
    atlas = _atlas(["s1"], ["AAAA"], ["GENE_A"])
    # loom barcodes disjoint from atlas → no overlap
    _write_loom(tmp_path / "s1.loom", "s1", ["ZZZZ"], ["GENE_A"], seed=1)
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    out, notes = reconcile_looms(atlas, manifest, sample_col="sample_id", loom_path_col="loom_path")
    assert out is None
    assert notes  # a recorded reason


def test_reconcile_handles_duplicate_gene_symbols(tmp_path):
    """Real velocyto looms frequently have duplicate gene symbols."""
    atlas = _atlas(["s1"], ["AAAA", "CCCC"], ["GENE_A", "GENE_B", "GENE_C"])
    # loom has GENE_A twice (e.g. two Ensembl IDs map to same symbol)
    _write_loom(
        tmp_path / "s1.loom", "s1", ["AAAA", "CCCC"], ["GENE_A", "GENE_A", "GENE_B"], seed=1
    )
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    out, notes = reconcile_looms(atlas, manifest, sample_col="sample_id", loom_path_col="loom_path")
    # Must not raise, must return valid AnnData with atlas var_names and correct shape.
    assert out is not None
    assert list(out.var_names) == ["GENE_A", "GENE_B", "GENE_C"]
    assert out.layers["spliced"].shape == (2, 3)
    assert out.layers["unspliced"].shape == (2, 3)


def test_reconcile_returns_none_when_sample_col_absent(tmp_path):
    """If sample_col not in atlas obs, return (None, notes) with reason."""
    names = ["s1_AAAA-1", "s1_CCCC-1"]
    # Atlas has only 'other_col', not 'sample_id'
    a = ad.AnnData(
        X=np.ones((2, 1), dtype="float32"),
        obs=pd.DataFrame({"other_col": ["val1", "val2"]}, index=names),
    )
    a.var_names = ["GENE_A"]
    _write_loom(tmp_path / "s1.loom", "s1", ["AAAA", "CCCC"], ["GENE_A"], seed=1)
    # Manifest has sample_id column (so that check passes), but adata.obs doesn't.
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    out, notes = reconcile_looms(a, manifest, sample_col="sample_id", loom_path_col="loom_path")
    assert out is None
    assert any("sample_id" in n for n in notes)
