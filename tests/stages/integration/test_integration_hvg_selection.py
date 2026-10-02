"""Shared HVG-restriction logic used by both scVI and scANVI.

scVI already refused to train on every gene when feature_selection flagged HVGs; scANVI
had no such logic at all and trained on the full gene set regardless of config, silently
degrading its latent space the same way scVI's own history describes. This pins the
extracted, shared behavior directly so the two methods cannot drift on it again.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.core.exceptions import CellQuorumStageError
from cellquorum.stages.integration._hvg_selection import restrict_to_highly_variable


def _cohort(n_genes: int, n_hvg: int, n_cells: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    genes = [f"G{i}" for i in range(n_genes)]
    adata = ad.AnnData(X=counts, var=pd.DataFrame(index=genes))
    mask = np.zeros(n_genes, dtype=bool)
    mask[:n_hvg] = True
    adata.var["highly_variable"] = mask
    return adata


def test_restricts_to_highly_variable_genes_when_flagged_and_above_the_floor():
    work = _cohort(n_genes=600, n_hvg=520)

    restricted, note = restrict_to_highly_variable(work, {}, method_name="scANVI", min_hvg=500)

    assert restricted.n_vars == 520
    assert "520" in note


def test_falls_back_to_all_genes_when_too_few_hvg_flagged():
    work = _cohort(n_genes=600, n_hvg=10)

    restricted, note = restrict_to_highly_variable(work, {}, method_name="scANVI", min_hvg=500)

    assert restricted.n_vars == 600
    assert "10" in note
    assert "500" in note


def test_uses_all_genes_when_no_hvg_flag_present_and_not_requested():
    work = _cohort(n_genes=50, n_hvg=0)
    del work.var["highly_variable"]

    restricted, note = restrict_to_highly_variable(work, {}, method_name="scANVI", min_hvg=500)

    assert restricted.n_vars == 50
    assert "scANVI" in note


def test_raises_when_use_highly_variable_requested_but_flag_absent():
    work = _cohort(n_genes=50, n_hvg=0)
    del work.var["highly_variable"]

    with pytest.raises(CellQuorumStageError, match="use_highly_variable"):
        restrict_to_highly_variable(
            work, {"use_highly_variable": True}, method_name="scANVI", min_hvg=500
        )


def test_use_highly_variable_false_uses_all_genes_even_when_flag_present():
    work = _cohort(n_genes=600, n_hvg=520)

    restricted, note = restrict_to_highly_variable(
        work, {"use_highly_variable": False}, method_name="scANVI", min_hvg=500
    )

    assert restricted.n_vars == 600


def test_extra_required_genes_are_unioned_into_the_mask():
    work = _cohort(n_genes=600, n_hvg=520)
    # A gene outside the first 520 HVGs, that a caller (scVI's denoised layer) needs kept.
    extra_gene = work.var_names[550]

    restricted, note = restrict_to_highly_variable(
        work,
        {},
        method_name="scVI",
        min_hvg=500,
        extra_required_genes=[extra_gene],
    )

    assert restricted.n_vars == 521
    assert extra_gene in restricted.var_names
