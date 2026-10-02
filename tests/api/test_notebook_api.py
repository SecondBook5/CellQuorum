"""Tests for the thin notebook API (cq.pp / cq.tl / cq.diag / cq.evidence)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import cellquorum as cq
from cellquorum.api._notebook import NotebookStageOutput


def _counts_adata(n: int = 60, seed: int = 0) -> ad.AnnData:
    """Build a small counts AnnData with mito/ribo genes for QC."""

    rng = np.random.default_rng(seed)
    genes = ["MT-CO1", "MT-ND1", "RPS3", "RPL13"] + [f"G{i}" for i in range(40)]
    x = rng.poisson(2.0, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["counts"] = x.copy()
    a.obs_names = [f"cell_{i}" for i in range(n)]
    return a


def test_namespaces_are_exposed() -> None:
    """cq.pp / cq.tl / cq.diag / cq.evidence are importable."""

    assert hasattr(cq, "pp")
    assert hasattr(cq, "tl")
    assert hasattr(cq, "diag")
    assert hasattr(cq, "evidence")
    # run_pipeline remains the primary engine entry point.
    assert hasattr(cq, "run_pipeline")


def test_pp_qc_flag_no_drop_annotates_metrics() -> None:
    """cq.pp.qc runs the real QC stage and returns annotated adata + result."""

    adata = _counts_adata()
    out = cq.pp.qc(
        adata,
        floors={"min_genes_per_cell": None, "min_cells_per_gene": None},
        metrics={"layer": "counts"},
    )

    assert isinstance(out, NotebookStageOutput)
    # QC metric columns are present on the returned adata.
    assert "pct_counts_mito" in out.adata.obs.columns
    assert "total_counts" in out.adata.obs.columns
    # flag_no_drop never drops cells.
    assert out.adata.n_obs == adata.n_obs
    # The full StageResult is exposed for artifacts/metrics.
    assert isinstance(out.metrics, dict)


def test_pp_qc_accepts_base_config_and_kwargs() -> None:
    """A base config dict merges with call kwargs into the stage block."""

    adata = _counts_adata()
    out = cq.pp.qc(
        adata,
        config={"project": {"name": "nb"}},
        floors={"min_genes_per_cell": None, "min_cells_per_gene": None},
        metrics={"layer": "counts"},
    )
    assert "pct_counts_mito" in out.adata.obs.columns


def test_evidence_build_is_planned_not_silent() -> None:
    """cq.evidence.build raises a clear 'planned' error rather than missing."""

    with pytest.raises(NotImplementedError, match="planned"):
        cq.evidence.build()


def test_unknown_stage_raises() -> None:
    """run_stage rejects an unknown stage name with the available list."""

    from cellquorum.api._notebook import run_stage

    with pytest.raises(KeyError, match="Unknown stage"):
        run_stage("not_a_stage", _counts_adata())


def test_notebook_api_adds_no_new_stage_classes() -> None:
    """The namespaces wrap existing stages; they define no new stage classes."""

    for module in (cq.pp, cq.tl, cq.diag):
        for name in dir(module):
            obj = getattr(module, name)
            # No class whose name ends in "Stage" should be defined here.
            assert not (isinstance(obj, type) and name.endswith("Stage"))
