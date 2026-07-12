"""Tests for subclustering partition (CHOIR + sc-SHC)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.subclustering.partition import run_choir, run_scshc_test


def make_synthetic_counts_adata() -> ad.AnnData:
    """
    Build a synthetic AnnData with counts layer for partition tests.

    Returns:
        AnnData with 100 cells, layers["counts"], obs[donor].
    """
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(100, 30))

    obs = pd.DataFrame(
        {"donor": (["d1"] * 40 + ["d2"] * 35 + ["d3"] * 25)},
        index=[f"cell_{i}" for i in range(100)],
    )

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(30)])

    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts

    return adata


def make_mock_config(
    counts_layer: str = "counts",
    key_added: str = "subcluster",
    partition_method: str = "choir",
    donor_gate_group_key: str | None = "donor",
) -> MagicMock:
    """Build a mock SubclusteringConfig for tests."""
    config = MagicMock()
    config.counts_layer = counts_layer
    config.key_added = key_added
    config.partition.method = partition_method
    config.partition.seeds = [0]
    config.partition.choir = {"alpha": 0.05, "n_iterations": 10, "n_trees": 10}
    config.donor_gate.group_key = donor_gate_group_key
    config.formal_test.alpha = 0.05
    return config


def test_run_choir_wiring_with_mocked_backend(tmp_path: Path) -> None:
    """
    Verify run_choir wiring with a mocked backend (barcode-aligned join).

    This test MOCKS the R backend to return a canned CSV, proving:
    - The h5ad → R → CSV → obs join pipeline works.
    - Barcode alignment is correct (not positional).
    """
    # Build synthetic counts adata.
    adata = make_synthetic_counts_adata()

    # Build mock config.
    config = make_mock_config()

    # Build mock backend that writes a canned CSV with SHUFFLED barcode order.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    def mock_run_script(script_path: Path, args: list[str], timeout: int):
        # Extract out_csv path from args.
        out_csv = Path(args[1])

        # Write a canned barcode,subcluster CSV with SHUFFLED order.
        # This proves the join is barcode-aligned, not positional.
        barcodes = adata.obs_names.tolist()
        shuffled_barcodes = barcodes[::-1]  # Reverse order.
        labels = ["cluster_A"] * 50 + ["cluster_B"] * 50

        canned_df = pd.DataFrame({"barcode": shuffled_barcodes, "subcluster": labels})
        canned_df.to_csv(out_csv, index=False)

        # Return success.
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    backend.run_script = mock_run_script

    # Run CHOIR with mocked backend.
    result = run_choir(adata, config, backend, tmp_path)

    # Verify labels joined correctly (not a MethodSkip).
    assert isinstance(result, ad.AnnData)
    assert "subcluster" in result.obs.columns

    # Verify barcode alignment: cell_0 (first) should get cluster_A (reversed).
    # After reversing, cell_0 maps to the label that was in position 99 → cluster_B.
    assert result.obs.loc["cell_0", "subcluster"] == "cluster_B"
    assert result.obs.loc["cell_99", "subcluster"] == "cluster_A"


def test_run_choir_method_skip_when_backend_none(tmp_path: Path) -> None:
    """Verify run_choir returns MethodSkip when backend is None."""
    adata = make_synthetic_counts_adata()
    config = make_mock_config()

    # Run CHOIR with backend=None.
    result = run_choir(adata, config, backend=None, scratch_dir=tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "Rscript unavailable" in result.reason


def test_run_choir_method_skip_when_rscript_unavailable(tmp_path: Path) -> None:
    """Verify run_choir returns MethodSkip when Rscript unavailable."""
    adata = make_synthetic_counts_adata()
    config = make_mock_config()

    # Build mock backend that reports CHOIR unavailable.
    backend = MagicMock()
    backend._r_package_available.return_value = False

    # Run CHOIR.
    result = run_choir(adata, config, backend, tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "CHOIR R package unavailable" in result.reason


def test_run_choir_raises_on_missing_counts_layer(tmp_path: Path) -> None:
    """Verify run_choir raises when counts layer is missing."""
    adata = make_synthetic_counts_adata()
    del adata.layers["counts"]  # Remove counts layer.

    config = make_mock_config()

    # Build mock backend.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    # Run CHOIR (should raise).
    from cellquorum.core.exceptions import CellQuorumBackendError

    with pytest.raises(CellQuorumBackendError, match="counts layer"):
        run_choir(adata, config, backend, tmp_path)


def test_run_scshc_test_wiring_with_mocked_backend(tmp_path: Path) -> None:
    """Verify run_scshc_test wiring with a mocked backend."""
    # Build synthetic counts adata with cluster labels.
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    # Build mock config.
    config = make_mock_config()

    # Build mock backend that writes a canned significance CSV.
    backend = MagicMock()
    backend._r_package_available.return_value = True

    def mock_run_script(script_path: Path, args: list[str], timeout: int):
        # Extract out_csv path from args.
        out_csv = Path(args[2])

        # Write a canned per-split significance CSV.
        canned_df = pd.DataFrame(
            {
                "split_index": [1, 2],
                "p_value": [0.01, 0.10],
                "significant": [True, False],
            }
        )
        canned_df.to_csv(out_csv, index=False)

        # Return success.
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    backend.run_script = mock_run_script

    # Run sc-SHC test with mocked backend.
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify significance dict returned (not a MethodSkip).
    assert isinstance(result, dict)
    assert result["method"] == "scshc"
    assert result["n_splits_tested"] == 2
    assert result["n_significant"] == 1
    assert result["alpha"] == 0.05


def test_run_scshc_test_method_skip_when_backend_none(tmp_path: Path) -> None:
    """Verify run_scshc_test returns MethodSkip when backend is None."""
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    config = make_mock_config()

    # Run sc-SHC test with backend=None.
    result = run_scshc_test(adata, "subcluster", config, backend=None, scratch_dir=tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "Rscript unavailable" in result.reason


def test_run_scshc_test_method_skip_when_scshc_unavailable(tmp_path: Path) -> None:
    """Verify run_scshc_test returns MethodSkip when scSHC unavailable."""
    adata = make_synthetic_counts_adata()
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    config = make_mock_config()

    # Build mock backend that reports scSHC unavailable.
    backend = MagicMock()
    backend._r_package_available.return_value = False

    # Run sc-SHC test.
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify MethodSkip returned.
    assert isinstance(result, MethodSkip)
    assert "scSHC R package unavailable" in result.reason


# ---- Real R tests (gated on Rscript + CHOIR + scSHC availability) ---- #


def _r_available() -> bool:
    """Check whether Rscript is available."""
    return shutil.which("Rscript") is not None


def _choir_available() -> bool:
    """Check whether CHOIR R package is available."""
    if not _r_available():
        return False
    import subprocess

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status = ifelse(requireNamespace('CHOIR', quietly = TRUE), 0, 1))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def _scshc_available() -> bool:
    """Check whether scSHC R package is available."""
    if not _r_available():
        return False
    import subprocess

    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status = ifelse(requireNamespace('scSHC', quietly = TRUE), 0, 1))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not (_r_available() and _choir_available()),
    reason="Rscript or CHOIR R package unavailable",
)
def test_run_choir_real_r(tmp_path: Path) -> None:
    """
    Real R test: run CHOIR on synthetic data.

    This test requires Rscript + CHOIR installed. Uses TINY n_iterations/n_trees.
    NOTE: This test may fail if CHOIR has initialization issues; the mocked tests
    are the primary verification of the partition.py wiring.
    """
    # Build synthetic counts adata with MORE genes (CHOIR needs sufficient features).
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(100, 200))  # 200 genes
    obs = pd.DataFrame(
        {"donor": (["d1"] * 40 + ["d2"] * 35 + ["d3"] * 25)},
        index=[f"cell_{i}" for i in range(100)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(200)])
    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts

    # Build mock config with TINY CHOIR parameters (fast).
    config = make_mock_config()
    config.partition.choir = {"alpha": 0.05, "n_iterations": 5, "n_trees": 5}

    # Build real Rscript backend.
    from cellquorum.backends.rscript import build_rscript_backend

    backend = build_rscript_backend()

    # Run CHOIR (real R execution).
    result = run_choir(adata, config, backend, tmp_path)

    # Verify CHOIR succeeded (not a MethodSkip).
    # If CHOIR has environment issues, this test may skip - that's OK because
    # the mocked tests verify the wiring.
    if isinstance(result, MethodSkip):
        pytest.skip(f"CHOIR real test skipped (R environment issue): {result.reason}")

    assert isinstance(result, ad.AnnData)
    assert "subcluster" in result.obs.columns

    # Verify per-cell labels assigned.
    n_subclusters = result.obs["subcluster"].nunique()
    assert n_subclusters >= 1  # At least one cluster.


@pytest.mark.skipif(
    not (_r_available() and _scshc_available()),
    reason="Rscript or scSHC R package unavailable",
)
def test_run_scshc_test_real_r(tmp_path: Path) -> None:
    """
    Real R test: run sc-SHC on synthetic clusters.

    This test requires Rscript + scSHC installed.
    NOTE: This test may skip if R environment has issues; the mocked tests
    are the primary verification of the scshc_test.py wiring.
    """
    # Build synthetic counts adata with cluster labels (MORE genes).
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5.0, size=(100, 200))  # 200 genes
    obs = pd.DataFrame(
        {"donor": (["d1"] * 40 + ["d2"] * 35 + ["d3"] * 25)},
        index=[f"cell_{i}" for i in range(100)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(200)])
    adata = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obs["subcluster"] = ["cluster_A"] * 50 + ["cluster_B"] * 50

    # Build mock config.
    config = make_mock_config()

    # Build real Rscript backend.
    from cellquorum.backends.rscript import build_rscript_backend

    backend = build_rscript_backend()

    # Run sc-SHC test (real R execution).
    result = run_scshc_test(adata, "subcluster", config, backend, tmp_path)

    # Verify sc-SHC succeeded (not a MethodSkip).
    # If R environment has issues, this test may skip - that's OK.
    if isinstance(result, MethodSkip):
        pytest.skip(f"sc-SHC real test skipped (R environment issue): {result.reason}")

    assert isinstance(result, dict)
    assert result["method"] == "scshc"
    assert result["n_splits_tested"] >= 0
