"""Tests for the differential_expression stage's per-subcluster stratification.

Real gap this closes: every other comparative stage (composition,
differential_abundance, coexpression, grn, enrichment) runs per-subcluster
(cell_type_col: fibroblast_subcluster); differential_expression pooled ALL
subclusters into one comparison. That is a real composition-confound risk --
a gene can look "differentially expressed LE vs Normal" purely because the
MIX of fibroblast states shifted, not because any one state's expression
changed. Confirmed on real data: one top hit (BCL2A1) lost ~25% of its
fold-change once two QC-confounded subclusters were excluded from the pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.contracts.layer_tags import set_layer_tag
from cellquorum.stages.comparative.differential_expression.config import (
    DifferentialExpressionConfig,
)
from cellquorum.stages.comparative.differential_expression.stage import (
    DifferentialExpressionStage,
)


def _de_adata(n_per_arm=6) -> ad.AnnData:
    """Two subclusters ('A', 'B'), each with 3 donors x 2 conditions x n_per_arm
    cells -- enough for edgeR's per-arm donor-replication floor in both the
    pooled comparison and each subcluster on its own."""
    rng = np.random.default_rng(0)
    donors = ["d1", "d2", "d3"]
    conditions = ["case", "control"]
    clusters = ["A", "B"]

    rows = []
    for cl in clusters:
        for donor in donors:
            for cond in conditions:
                for _ in range(n_per_arm):
                    rows.append({"cl": cl, "donor": donor, "condition": cond})
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]

    n = len(obs)
    n_genes = 10
    counts = rng.poisson(20, size=(n, n_genes)).astype(np.float32)
    a = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    a.layers["counts"] = counts.copy()
    set_layer_tag(a, "counts", kind="counts")
    return a


@dataclass
class _EdgeRCapturingBackend:
    """Stands in for the Rscript backend: writes a plausible edgeR-style CSV
    to whatever path the method passes as its output arg (args[2]), without
    running R. Records every invocation's output path so a test can tell how
    many times (and for what) the method actually ran."""

    calls: list = None

    def __post_init__(self):
        self.calls = []

    def _rscript_available(self) -> bool:
        return True

    def _r_package_available(self, name: str) -> bool:
        return True

    def run_script(self, script, args, timeout=None):
        out_csv = Path(args[2])
        self.calls.append(out_csv)
        pd.DataFrame(
            {
                "gene": ["g0", "g1"],
                "logFC": [1.5, -0.8],
                "logCPM": [4.0, 3.0],
                "F": [10.0, 5.0],
                "PValue": [0.001, 0.05],
                "FDR": [0.01, 0.1],
            }
        ).to_csv(out_csv, index=False)

        @dataclass
        class _Result:
            returncode: int = 0
            stdout: str = ""
            stderr: str = ""

        return _Result()


def _context(adata, *, cell_type_col=None, tmp_path):
    de_config = DifferentialExpressionConfig(
        enabled=True,
        cell_type_col=cell_type_col,
    )
    de_config = de_config.model_copy(
        update={
            "min_count": 0,
            "min_total_count": 0,
        }
    )
    config = SimpleNamespace(
        differential_expression=de_config,
        design=SimpleNamespace(
            donor_col="donor",
            condition_col="condition",
            case="case",
            control="control",
            paired=False,
        ),
        cohort=SimpleNamespace(donor_key=None, sample_key=None, focus=None),
    )
    backend = _EdgeRCapturingBackend()
    registry = SimpleNamespace(get=lambda name: backend)
    paths = SimpleNamespace(
        results=str(tmp_path), figures=str(tmp_path), scratch=str(tmp_path), root=str(tmp_path)
    )
    ctx = SimpleNamespace(
        config=config,
        adata=adata,
        paths=paths,
        backend_registry=registry,
        require_adata=lambda: adata,
    )
    return ctx, backend


def test_no_cell_type_col_runs_pooled_only_unchanged(tmp_path):
    """Default behavior (cell_type_col unset) must run exactly once, pooled."""
    adata = _de_adata()
    ctx, backend = _context(adata, cell_type_col=None, tmp_path=tmp_path)

    result = DifferentialExpressionStage().run(ctx)

    assert len(backend.calls) == 1
    assert (tmp_path / "de_pseudobulk_edger.csv").exists()
    assert not (tmp_path / "de_pseudobulk_edger_by_subcluster.csv").exists()
    assert result.status == "success"


def test_cell_type_col_runs_pooled_and_per_subcluster(tmp_path):
    """With cell_type_col set: the pooled table still exists, PLUS one combined
    per-subcluster table with a cell_type column identifying which subcluster
    each row's statistics came from."""
    adata = _de_adata()
    ctx, backend = _context(adata, cell_type_col="cl", tmp_path=tmp_path)

    result = DifferentialExpressionStage().run(ctx)

    # Pooled (1) + subcluster A (1) + subcluster B (1).
    assert len(backend.calls) == 3
    assert (tmp_path / "de_pseudobulk_edger.csv").exists()

    combined = pd.read_csv(tmp_path / "de_pseudobulk_edger_by_subcluster.csv")
    assert set(combined["cell_type"].unique()) == {"A", "B"}
    assert "gene" in combined.columns
    assert result.status == "success"


def test_a_subcluster_too_thin_for_edger_is_skipped_not_fatal(tmp_path):
    """A subcluster without enough donor replication per arm must be skipped
    with a warning, not crash the whole stage or block the other subclusters."""
    adata = _de_adata()
    # Give subcluster 'B' only one donor total, in only one condition -- fails
    # the per-arm donor-replication floor.
    mask = adata.obs["cl"] == "B"
    adata.obs.loc[mask, "donor"] = "d1"
    adata.obs.loc[mask, "condition"] = "case"

    ctx, backend = _context(adata, cell_type_col="cl", tmp_path=tmp_path)

    result = DifferentialExpressionStage().run(ctx)

    combined = pd.read_csv(tmp_path / "de_pseudobulk_edger_by_subcluster.csv")
    assert set(combined["cell_type"].unique()) == {"A"}
    assert any("B" in w for w in result.warnings)
