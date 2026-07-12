"""Tests for the annotation_diagnostics stage and scDiagnostics method."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest

from cellquorum.contracts import CellQuorumContractError
from cellquorum.methods.base import MethodSkip


@pytest.fixture
def minimal_annotated_adata() -> ad.AnnData:
    """Return a minimal annotated AnnData with required structure."""
    n_cells = 100
    n_genes = 50
    # Build minimal annotated object with lognorm layer + X_pca + cell_type.
    adata = ad.AnnData(X=np.random.randn(n_cells, n_genes))
    adata.layers["lognorm"] = adata.X.copy()
    adata.obsm["X_pca"] = np.random.randn(n_cells, 10)
    adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB", "TypeC"], n_cells)
    return adata


@pytest.fixture
def adata_missing_pca() -> ad.AnnData:
    """Return AnnData without X_pca (contract violation fixture)."""
    n_cells = 100
    n_genes = 50
    adata = ad.AnnData(X=np.random.randn(n_cells, n_genes))
    adata.layers["lognorm"] = adata.X.copy()
    adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB"], n_cells)
    # Missing X_pca → contract should fail.
    return adata


@pytest.fixture
def stub_context_no_rscript(tmp_path: Path) -> MagicMock:
    """Return a stub context whose rscript backend is unavailable."""
    ctx = MagicMock()
    ctx.backend_registry.get.return_value = None
    ctx.paths.scratch = tmp_path / "scratch"
    ctx.paths.scratch.mkdir(parents=True, exist_ok=True)
    ctx.config = MagicMock()
    ctx.config.annotation_diagnostics = MagicMock()
    ctx.config.annotation_diagnostics.enabled = True
    ctx.config.annotation_diagnostics.cell_type_col = "cell_type"
    return ctx


def test_scdiagnostics_method_contract_requires_pca(adata_missing_pca: ad.AnnData) -> None:
    """Verify the method's input_contract requires X_pca in obsm."""
    from cellquorum.annotation_diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    method = ScdiagnosticsMethod()
    config_dict = {"cell_type_col": "cell_type"}
    contract = method.input_contract(config_dict)

    # Missing X_pca should raise CellQuorumContractError.
    with pytest.raises(CellQuorumContractError, match="X_pca"):
        contract.validate(adata_missing_pca)


def test_scdiagnostics_method_skip_when_rscript_unavailable(
    minimal_annotated_adata: ad.AnnData,
    stub_context_no_rscript: MagicMock,
) -> None:
    """Verify the method returns MethodSkip when Rscript backend is unavailable."""
    from cellquorum.annotation_diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    method = ScdiagnosticsMethod()
    config_dict = {
        "cell_type_col": "cell_type",
        "reference_h5ad": None,
        "soft_scores_obsm": None,
        "pc_subset": [1, 2, 3, 4, 5],
        "n_tree": 500,
        "n_neighbor": 15,
        "timeout_seconds": 1800,
    }

    # Run the method with unavailable backend → MethodSkip.
    result = method._run(minimal_annotated_adata, config_dict, stub_context_no_rscript)
    assert isinstance(result, MethodSkip)
    assert "rscript" in result.reason.lower()


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript unavailable",
)
def test_scdiagnostics_method_real_r_run(
    minimal_annotated_adata: ad.AnnData,
    tmp_path: Path,
) -> None:
    """Real R integration test (skippable when Rscript/scDiagnostics absent)."""
    from cellquorum.annotation_diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )
    from cellquorum.backends.rscript import build_rscript_backend

    # Try to import scDiagnostics via a quick R check.
    backend = build_rscript_backend(r_packages=["scDiagnostics"])
    status = backend.status()
    if not status.available:
        pytest.skip("scDiagnostics R package unavailable")

    # Build a stub context with real backend.
    ctx = MagicMock()
    ctx.backend_registry.get.return_value = backend
    ctx.paths.scratch = tmp_path / "scratch"
    ctx.paths.scratch.mkdir(parents=True, exist_ok=True)
    ctx.config = MagicMock()

    method = ScdiagnosticsMethod()

    # Test with soft_scores_obsm to ensure we get entropy diagnostic.
    adata_with_soft = minimal_annotated_adata.copy()
    # Add dummy soft scores (cells x 3 cell types).
    import numpy as np

    soft_scores = np.random.dirichlet(np.ones(3), size=adata_with_soft.n_obs)
    adata_with_soft.obsm["soft_scores"] = soft_scores

    config_dict = {
        "cell_type_col": "cell_type",
        "reference_h5ad": None,
        "soft_scores_obsm": "soft_scores",
        "pc_subset": [1, 2, 3, 4, 5],
        "n_tree": 500,
        "n_neighbor": 15,
        "timeout_seconds": 1800,
    }

    result = method._run(adata_with_soft, config_dict, ctx)

    # Real run should return StageResult (not MethodSkip).
    from cellquorum.core.stage import StageResult

    assert isinstance(result, StageResult)
    # Check that at least the entropy diagnostic was added.
    assert "scdiag_entropy" in result.adata.obs.columns
    assert result.metrics["n_diagnostics"] >= 1
