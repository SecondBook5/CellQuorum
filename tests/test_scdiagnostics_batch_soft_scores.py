"""Regression test: soft_scores_path must reach every R batch call, not just batch 0.

Found running Task 6.5 of the le_global rebuild: scdiag_entropy was only populated
for whichever cell type sorted into the first query batch (21.8% of cells on the
real 197,952-cell cohort) because the soft-scores CSV path was gated on
``index == 0`` when building each batch's R argv.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _write_reference_h5ad(path: Path, cell_types: list[str], n_genes: int = 10) -> None:
    n = len(cell_types)
    adata = ad.AnnData(X=np.random.default_rng(0).normal(size=(n, n_genes)).astype("float32"))
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    adata.layers["lognorm"] = adata.X.copy()
    adata.obsm["X_pca"] = np.random.default_rng(0).normal(size=(n, 5))
    adata.obs["cell_type"] = cell_types
    adata.write_h5ad(path)


class _RecordingBackend:
    """Fake Rscript backend: records every argv, writes a minimal valid result CSV."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _rscript_available(self) -> bool:
        return True

    def _r_package_available(self, pkg: str) -> bool:
        return True

    def run_script(self, script, args, timeout=None):
        self.calls.append(list(args))
        out_csv = Path(args[1])
        query_h5ad = Path(args[0])
        barcodes = ad.read_h5ad(query_h5ad, backed="r").obs_names
        pd.DataFrame({"barcode": barcodes}).to_csv(out_csv, index=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def two_batch_adata_and_backend(tmp_path: Path):
    """A query forced into 2 batches (one cell type per batch), each cell type
    carrying its own soft-score row -- so a batch call that never receives the
    soft-scores path is directly observable (missing scdiag_entropy).
    """
    n_genes = 10
    cell_types = ["TypeA"] * 3 + ["TypeB"] * 3
    barcodes = [f"CELL_{i}" for i in range(6)]

    adata = ad.AnnData(X=np.random.default_rng(1).normal(size=(6, n_genes)).astype("float32"))
    adata.obs_names = barcodes
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    adata.layers["lognorm"] = adata.X.copy()
    adata.obsm["X_pca"] = np.random.default_rng(1).normal(size=(6, 5))
    adata.obs["cell_type"] = cell_types
    adata.obsm["soft_scores"] = np.random.default_rng(1).dirichlet(np.ones(2), size=6)

    ref_path = tmp_path / "reference.h5ad"
    _write_reference_h5ad(ref_path, ["TypeA", "TypeB"], n_genes=n_genes)

    backend = _RecordingBackend()
    ctx = MagicMock()
    ctx.backend_registry.get.return_value = backend
    ctx.paths.scratch = tmp_path / "scratch"
    ctx.paths.scratch.mkdir(parents=True, exist_ok=True)

    config = {
        "cell_type_col": "cell_type",
        "reference_h5ad": str(ref_path),
        "soft_scores_obsm": "soft_scores",
        "pc_subset": [1, 2, 3, 4, 5],
        "n_tree": 10,
        "n_neighbor": 3,
        "timeout_seconds": 60,
        "max_query_cells_per_call": 3,  # forces one whole cell type per batch
        "max_reference_cells_per_label": 10,
    }
    return adata, ctx, backend, config


def test_soft_scores_path_reaches_every_batch(two_batch_adata_and_backend):
    from cellquorum.stages.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    adata, ctx, backend, config = two_batch_adata_and_backend

    ScdiagnosticsMethod()._run(adata, config, ctx)

    assert len(backend.calls) == 2, "expected the query split into 2 whole-cell-type batches"
    soft_scores_args = [call[4] for call in backend.calls]
    assert all(
        arg != "NONE" for arg in soft_scores_args
    ), f"soft_scores path must reach every batch, not just the first -- got {soft_scores_args}"
