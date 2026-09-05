"""QC stage invokes doublet + cell-cycle add-ons when enabled."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.qc.cell_cycle import TIROSH_G2M_GENES, TIROSH_S_GENES
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.stage import QCStage


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    # Include more Tirosh genes + background genes for proper scoring.
    s = TIROSH_S_GENES[:15]
    g2m = TIROSH_G2M_GENES[:15]
    genes = s + g2m + [f"G{i}" for i in range(200)]  # More genes for control set.
    n = 120
    x = rng.poisson(1.0, size=(n, len(genes))).astype(np.float32)
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["counts"] = x.copy()
    a.layers["cellquorum_normalized"] = np.log1p(x)
    return a


class _Paths:
    def __init__(self, tmp):
        from pathlib import Path

        self.results = Path(tmp)
        self.figures = Path(tmp)


class _Config:
    """Match real context.config structure: config.qc reads QCConfig."""

    def __init__(self, qc):
        self.qc = qc
        self.stages = type("S", (), {"qc": True})()


class _Ctx:
    def __init__(self, adata, tmp, qc_config):
        self._adata = adata
        self.paths = _Paths(tmp)
        self.config = _Config(qc_config)
        self.run_id = "test-run"
        self.random_seed = 42

    def require_adata(self):
        return self._adata


def test_qc_stage_runs_cell_cycle_when_enabled(tmp_path):
    a = _adata()
    qc = QCConfig(
        cell_cycle={"enabled": True},
        floors={"min_genes_per_cell": None, "min_cells_per_gene": None},
    )
    ctx = _Ctx(a, tmp_path, qc)
    result = QCStage().run(ctx)
    assert "phase" in result.adata.obs


def test_qc_stage_flags_doublets_when_enabled(tmp_path):
    a = _adata()
    qc = QCConfig(
        doublets={"enabled": True, "methods": ["scrublet"]},
        floors={"min_genes_per_cell": None, "min_cells_per_gene": None},
    )
    ctx = _Ctx(a, tmp_path, qc)
    result = QCStage().run(ctx)
    assert "predicted_doublet" in result.adata.obs
    assert result.adata.n_obs == 120  # flag-only, no removal


def test_qc_stage_exposes_feature_family_metrics_to_plots(tmp_path):
    """QC plots should see computed mito/ribo/hemoglobin percentage columns."""

    genes = ["MT-CO1", "RPS3", "RPL13", "HBA1", "KRT14", "KRT10"]
    x = np.array(
        [
            [5, 1, 0, 0, 3, 1],
            [0, 3, 2, 1, 4, 0],
            [1, 0, 1, 4, 2, 2],
            [0, 0, 0, 0, 6, 3],
        ],
        dtype=np.float32,
    )
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.layers["counts"] = x.copy()
    qc = QCConfig(
        metrics={"layer": "counts", "percent_top": [20]},
        doublets={"enabled": False},
        ambient={"enabled": False},
        # A 6-gene fixture is below any real detection floor, so the floors are lifted
        # explicitly. This is what the deleted `mode: flag_no_drop` used to arrange.
        floors={"min_genes_per_cell": None, "min_cells_per_gene": None},
        outputs={"figure_dpi": 40},
    )
    ctx = _Ctx(a, tmp_path, qc)

    result = QCStage().run(ctx)

    for column in ("pct_counts_mito", "pct_counts_ribo", "pct_counts_hemoglobin"):
        assert column in result.adata.obs
    assert not any("pct_counts_mito not found" in warning for warning in result.warnings)
    assert not any("pct_counts_ribo not found" in warning for warning in result.warnings)
    assert not any("pct_counts_hemoglobin not found" in warning for warning in result.warnings)

    figure_names = {
        artifact.path.name
        for artifact in result.artifacts
        if artifact.kind == "file" and artifact.path.suffix == ".png"
    }

    # This used to assert three per-metric histogram filenames — the v1 diagnostics module's
    # way of proving the figure layer saw the feature-family columns. That module is gone, and
    # asserting filenames was always an indirect test of the wrong thing: a renamed figure broke
    # it while a silently-absent column did not.
    #
    # So assert the property instead. The figure source is built from the PRE-filter object, and
    # the panel writer raises if a metric it plots is missing, so a figure set reaching disk with
    # no warnings is evidence the columns arrived.
    assert figure_names, "the stage wrote no figures, so nothing consumed the metric columns"
    assert "qc_overview.png" in figure_names
    assert not [warning for warning in result.warnings if "could not be written" in warning]
