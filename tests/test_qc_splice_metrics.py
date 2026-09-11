"""Tests for the optional qc_splice_metrics stage (order=15)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.pipeline import build_pipeline_context
from cellquorum.stages.qc.splice_metrics import QCSpliceMetricsStage

loompy = pytest.importorskip("loompy")


def _write_loom(path, stem, barcodes, genes, spliced_value, unspliced_value):
    """Write a tiny velocyto-style loom with FIXED (not random) layer values."""
    n_genes, n_cells = len(genes), len(barcodes)
    main = np.zeros((n_genes, n_cells), dtype="float32")
    spliced = np.full((n_genes, n_cells), spliced_value, dtype="float32")
    unspliced = np.full((n_genes, n_cells), unspliced_value, dtype="float32")
    row_attrs = {"Gene": np.array(genes, dtype=object)}
    col_attrs = {"CellID": np.array([f"{stem}:{bc}x" for bc in barcodes], dtype=object)}
    loompy.create(
        str(path),
        layers={"": main, "spliced": spliced, "unspliced": unspliced},
        row_attrs=row_attrs,
        col_attrs=col_attrs,
    )


def _atlas(sample_id, barcodes, genes):
    names = [f"{sample_id}_{bc}-1" for bc in barcodes]
    adata = ad.AnnData(
        X=np.ones((len(names), len(genes)), dtype="float32"),
        obs=pd.DataFrame({"sample_id": [sample_id] * len(names)}, index=names),
    )
    adata.var_names = genes
    return adata


def _context(tmp_path, adata, manifest):
    config = CellQuorumConfig(compute={"prefer_gpu": False}, r={"enabled": False})
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)
    context.manifest = manifest
    return context


def test_stage_skips_when_no_manifest(tmp_path):
    adata = _atlas("s1", ["AAAA"], ["GENE_A"])
    config = CellQuorumConfig(compute={"prefer_gpu": False}, r={"enabled": False})
    context = build_pipeline_context(config, output_dir=tmp_path / "run").with_adata(adata)

    result = QCSpliceMetricsStage().run(context)

    assert result.status == "skipped"
    assert "qc_splice_intronic_fraction" not in result.adata.obs.columns


def test_stage_writes_intronic_fraction_for_reconciled_cells_only(tmp_path):
    genes = ["GENE_A", "GENE_B"]
    atlas = _atlas("s1", ["AAAA", "CCCC"], genes)
    # A third cell with no matching loom barcode, to prove it stays NaN.
    unmatched = _atlas("s1", ["GGGG"], genes)
    unmatched.obs_names = ["s1_unmatched-1"]
    atlas = ad.concat([atlas, unmatched])

    _write_loom(
        tmp_path / "s1.loom",
        "s1",
        ["AAAA", "CCCC"],
        genes,
        spliced_value=3.0,
        unspliced_value=1.0,
    )
    manifest = pd.DataFrame({"sample_id": ["s1"], "loom_path": [str(tmp_path / "s1.loom")]})
    context = _context(tmp_path, atlas, manifest)

    result = QCSpliceMetricsStage().run(context)

    assert result.status == "success"
    obs = result.adata.obs
    # Every gene x cell entry is spliced=3, unspliced=1 -> intronic_fraction = 1/4.
    assert obs.loc["s1_AAAA-1", "qc_splice_intronic_fraction"] == pytest.approx(0.25)
    assert obs.loc["s1_CCCC-1", "qc_splice_intronic_fraction"] == pytest.approx(0.25)
    assert np.isnan(obs.loc["s1_unmatched-1", "qc_splice_intronic_fraction"])
