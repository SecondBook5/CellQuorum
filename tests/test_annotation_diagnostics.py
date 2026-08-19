"""Tests for the annotation_diagnostics stage and scDiagnostics method."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.contracts.layer_tags import set_layer_tag
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
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    method = ScdiagnosticsMethod()
    config_dict = {"cell_type_col": "cell_type"}
    contract = method.input_contract(config_dict)

    # Missing X_pca should raise CellQuorumContractError.
    with pytest.raises(CellQuorumContractError, match="X_pca"):
        contract.validate(adata_missing_pca)


def test_scdiagnostics_contract_uses_configured_expression_layer() -> None:
    """Verify scDiagnostics can audit CellQuorum-tagged normalized layers."""
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    adata = ad.AnnData(X=np.random.randn(8, 6))
    adata.layers["cellquorum_normalized"] = adata.X.copy()
    set_layer_tag(adata, "cellquorum_normalized", kind="lognorm")
    adata.obsm["X_pca"] = np.random.randn(8, 3)
    adata.obs["ref_state"] = ["KC 1", "KC 2"] * 4

    method = ScdiagnosticsMethod()
    contract = method.input_contract(
        {
            "cell_type_col": "ref_state",
            "expression_layer": "cellquorum_normalized",
        }
    )

    contract.validate(adata)


def test_scdiagnostics_method_skip_when_rscript_unavailable(
    minimal_annotated_adata: ad.AnnData,
    stub_context_no_rscript: MagicMock,
    monkeypatch,
) -> None:
    """Verify the method returns MethodSkip when Rscript is missing."""
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    # Monkeypatch shutil.which to simulate missing Rscript.
    monkeypatch.setattr("shutil.which", lambda x: None)

    method = ScdiagnosticsMethod()
    config_dict = {
        "cell_type_col": "cell_type",
        "reference_h5ad": None,
        "soft_scores_obsm": None,
        "pc_subset": [1, 2, 3, 4, 5],
        "n_tree": 500,
        "n_neighbor": 15,
        "timeout_seconds": 1800,
        "r_package": "scDiagnostics",
    }

    # Run the method with missing Rscript → MethodSkip.
    result = method._run(minimal_annotated_adata, config_dict, stub_context_no_rscript)
    assert isinstance(result, MethodSkip)
    assert "rscript" in result.reason.lower()


def test_scdiagnostics_entropy_from_soft_scores_without_r(
    minimal_annotated_adata: ad.AnnData,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Soft-score entropy should work without R when no reference is configured."""
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    monkeypatch.setattr("shutil.which", lambda x: None)
    adata = minimal_annotated_adata.copy()
    adata.obsm["soft_scores"] = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.25, 0.75], *([[0.9, 0.1]] * 97)]
    )
    ctx = MagicMock()
    ctx.paths.scratch = tmp_path

    result = ScdiagnosticsMethod()._run(
        adata,
        {
            "cell_type_col": "cell_type",
            "reference_h5ad": None,
            "soft_scores_obsm": "soft_scores",
        },
        ctx,
    )

    assert "scdiag_entropy" in result.adata.obs
    np.testing.assert_allclose(result.adata.obs["scdiag_entropy"].iloc[:2], [0.0, 1.0])
    assert result.metrics["diagnostics_computed"] == ["scdiag_entropy"]
    assert (tmp_path / "scdiag_results.csv").exists()


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript unavailable",
)
def test_scdiagnostics_method_real_r_run(
    minimal_annotated_adata: ad.AnnData,
    tmp_path: Path,
) -> None:
    """Real R integration test (skippable when Rscript/scDiagnostics absent)."""
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
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


def test_scdiagnostics_barcode_alignment_with_reordered_csv(tmp_path: Path) -> None:
    """Prove barcode-keyed join: reordered CSV still assigns correct values."""
    from cellquorum.annotation.diagnostics.scdiagnostics_method import (
        ScdiagnosticsMethod,
    )

    # Create a small test AnnData with known cell barcodes.
    n_cells = 5
    n_genes = 10
    barcodes = [f"CELL_{i}" for i in range(n_cells)]
    adata = ad.AnnData(X=np.random.randn(n_cells, n_genes))
    adata.obs_names = barcodes
    adata.layers["lognorm"] = adata.X.copy()
    adata.obsm["X_pca"] = np.random.randn(n_cells, 5)
    adata.obs["cell_type"] = ["TypeA"] * n_cells

    # Create a diagnostic CSV with REORDERED rows (different order from
    # adata.obs_names). Each cell gets a unique diagnostic value that
    # matches its barcode (so we can verify alignment).
    reordered_barcodes = ["CELL_2", "CELL_0", "CELL_4", "CELL_1", "CELL_3"]
    csv_path = tmp_path / "reordered_diag.csv"
    with open(csv_path, "w") as f:
        f.write("barcode,scdiag_test\n")
        for bc in reordered_barcodes:
            # Diagnostic value = cell index (e.g., CELL_2 → 2.0).
            cell_idx = int(bc.split("_")[1])
            f.write(f"{bc},{cell_idx}.0\n")

    # Read the CSV via the method's helper.
    method = ScdiagnosticsMethod()
    diag_df = method._read_diagnostic_csv(csv_path)

    # Verify the DataFrame is indexed by barcode.
    assert diag_df.index.name in ("barcode", "cell") or diag_df.index.tolist() == [
        "CELL_2",
        "CELL_0",
        "CELL_4",
        "CELL_1",
        "CELL_3",
    ]

    # Reindex to match adata.obs_names order.
    diag_df_aligned = diag_df.reindex(adata.obs_names)

    # Each cell should get ITS OWN value (CELL_0 → 0.0, CELL_1 → 1.0, etc).
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    actual_values = diag_df_aligned["scdiag_test"].to_numpy()
    np.testing.assert_array_equal(
        actual_values,
        expected_values,
        err_msg="Barcode alignment failed: cells did not get their own values",
    )
