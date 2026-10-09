"""TDD tests for the AnnData/file glue around the ambient specificity filter."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from optional_deps import requires_parquet

from cellquorum.stages.comparative.differential_expression.ambient_filter import (
    filter_ambient_genes,
    load_expression_profile,
)


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Keratinocytes": [5.0, 0.2],
            "Fibroblasts": [0.5, 3.0],
        },
        index=["KRT10", "ACTA2"],
    )


def _adata(genes: list[str]) -> ad.AnnData:
    rng = np.random.default_rng(0)
    n = 10
    a = ad.AnnData(rng.poisson(1.0, size=(n, len(genes))).astype("float32"))
    a.var_names = genes
    return a


def test_load_expression_profile_reads_csv(tmp_path):
    csv_path = tmp_path / "profile.csv"
    _profile().to_csv(csv_path)
    loaded = load_expression_profile(csv_path)
    assert list(loaded.columns) == ["Keratinocytes", "Fibroblasts"]
    assert list(loaded.index) == ["KRT10", "ACTA2"]


@requires_parquet()
def test_load_expression_profile_reads_parquet(tmp_path):
    parquet_path = tmp_path / "profile.parquet"
    _profile().to_parquet(parquet_path)
    loaded = load_expression_profile(parquet_path)
    assert list(loaded.columns) == ["Keratinocytes", "Fibroblasts"]


def test_filter_ambient_genes_drops_the_ambient_gene_and_keeps_the_rest():
    adata = _adata(["KRT10", "ACTA2"])
    filtered, info = filter_ambient_genes(
        adata, profile=_profile(), target_group="Fibroblasts", spec_threshold=0.25
    )
    assert list(filtered.var_names) == ["ACTA2"]
    assert info["n_dropped"] == 1
    assert info["dropped_genes"] == ["KRT10"]
    assert info["n_genes_before"] == 2
    assert info["n_genes_after"] == 1


def test_filter_ambient_genes_keeps_a_gene_absent_from_the_reference():
    """A gene the reference never covered cannot be judged ambient -> kept."""
    adata = _adata(["ACTA2", "NOT_IN_REFERENCE"])
    filtered, info = filter_ambient_genes(
        adata, profile=_profile(), target_group="Fibroblasts", spec_threshold=0.25
    )
    assert set(filtered.var_names) == {"ACTA2", "NOT_IN_REFERENCE"}
    assert info["n_dropped"] == 0


def test_filter_ambient_genes_preserves_cell_count():
    adata = _adata(["KRT10", "ACTA2"])
    filtered, _ = filter_ambient_genes(
        adata, profile=_profile(), target_group="Fibroblasts", spec_threshold=0.25
    )
    assert filtered.n_obs == adata.n_obs
