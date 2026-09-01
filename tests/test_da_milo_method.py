"""Tests for MiloMethod (neighborhood-level DA via miloR)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.comparative.differential_abundance.milo_method import MiloMethod


@pytest.fixture
def mock_context():
    """Mock execution context with paths and backend."""

    class MockBackend:
        """Mock Rscript backend."""

        def _rscript_available(self) -> bool:
            """Rscript is present; availability is gated on the R package below."""
            return True

        def _r_package_available(self, package: str) -> bool:
            """Check if an R package is available."""
            # miloR is installed in test env
            return package == "miloR"

        def run_script(self, script: Path, args: list[str], timeout: int):
            """Mock run_script — will actually run for happy-path tests."""
            import subprocess

            result = subprocess.run(
                ["Rscript", str(script), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result

    class MockRegistry:
        def get(self, backend_name: str):
            if backend_name == "rscript":
                return MockBackend()
            raise ValueError(f"Unknown backend: {backend_name}")

    class MockPaths:
        def __init__(self, tmp_path: Path):
            self.scratch = tmp_path / "scratch"
            self.results = tmp_path / "results"
            self.figures = tmp_path / "figures"
            self.scratch.mkdir(parents=True, exist_ok=True)
            self.results.mkdir(parents=True, exist_ok=True)

    class MockContext:
        def __init__(self, tmp_path: Path):
            self.paths = MockPaths(tmp_path)
            self.backend_registry = MockRegistry()

    return MockContext


def test_milo_happy_path(tmp_path, mock_context):
    """Test MiloMethod with a fixture that yields detectable DA signal."""

    # Build fixture: two separated Gaussian blobs in 5D, one enriched in case.
    # 3 control donors + 3 case donors, 100 cells/donor, unpaired design.
    np.random.seed(42)
    n_cells_per_donor = 100
    n_control_donors = 3
    n_case_donors = 3

    # Blob 1: control-enriched (80% control, 20% case)
    # Blob 2: case-enriched (20% control, 80% case)
    n_blob1_control = int(n_cells_per_donor * n_control_donors * 0.8)
    n_blob2_control = int(n_cells_per_donor * n_control_donors * 0.2)
    n_blob1_case = int(n_cells_per_donor * n_case_donors * 0.2)
    n_blob2_case = int(n_cells_per_donor * n_case_donors * 0.8)

    # Generate blobs: centered at [0,0,0,0,0] and [5,5,5,5,5] with noise
    blob1 = np.random.randn(n_blob1_control + n_blob1_case, 5) * 0.5
    blob2 = np.random.randn(n_blob2_control + n_blob2_case, 5) * 0.5 + 5.0

    # Assemble embedding
    X_pca = np.vstack([blob1, blob2])
    n_cells = X_pca.shape[0]

    # Build obs: condition and donor assignments
    conditions = (
        ["control"] * n_blob1_control
        + ["case"] * n_blob1_case
        + ["control"] * n_blob2_control
        + ["case"] * n_blob2_case
    )

    # Assign donors: 3 control (D1, D2, D3) and 3 case (D4, D5, D6)
    donors = []
    for i, cond in enumerate(conditions):
        if cond == "control":
            donors.append(f"D{(i % n_control_donors) + 1}")
        else:
            donors.append(f"D{(i % n_case_donors) + 4}")

    obs = {
        "condition": conditions,
        "donor": donors,
    }

    # Create AnnData with X_pca in obsm
    adata = ad.AnnData(
        X=np.zeros((n_cells, 1)),  # placeholder
        obs=obs,
        obsm={"X_pca": X_pca},
    )

    # Config: k=15, prop=0.2 for small fast fixture (not the default k=30/prop=0.1)
    config = {
        "use_rep": "X_pca",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
        "k": 15,
        "prop": 0.2,
        "spatial_fdr": 0.2,
        "paired": False,
    }

    context = mock_context(tmp_path)
    method = MiloMethod()

    # Check if miloR and Rscript are available; skip test if not
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")

    result = method._run(adata, config, context)

    # If miloR not available, expect MethodSkip
    if isinstance(result, MethodSkip):
        pytest.skip("miloR not available in test environment")

    # Assert the DA table is present and remains the primary artifact.
    assert len(result.artifacts) >= 1
    artifact = result.artifacts[0]
    assert artifact.name == "da_results"
    assert artifact.kind == "csv"
    assert Path(artifact.path).exists()

    # Read and validate output CSV
    df = pd.read_csv(artifact.path)
    expected_cols = [
        "nhood",
        "logFC",
        "PValue",
        "SpatialFDR",
        "nhood_size",
        "majority_celltype",
        "celltype_fraction",
    ]
    assert list(df.columns) == expected_cols

    # Assert at least one neighborhood has SpatialFDR < 0.2 (signal detection)
    assert (df["SpatialFDR"] < 0.2).any(), "Expected at least one nhood with SpatialFDR < 0.2"

    # Check metrics
    assert result.metrics["case"] == "case"
    assert result.metrics["control"] == "control"
    assert result.metrics["paired"] is False
    assert result.metrics["k"] == 15
    assert result.metrics["prop"] == 0.2
    assert result.metrics["use_rep"] == "X_pca"
    assert result.metrics["n_nhoods"] > 0
    assert "n_da" in result.metrics


def test_milo_missing_rep_skip(tmp_path, mock_context):
    """Test that missing reduced-dim rep returns MethodSkip."""

    # AnnData with no X_pca and use_rep not in obsm
    adata = ad.AnnData(
        X=np.zeros((100, 10)),
        obs={
            "condition": ["control"] * 50 + ["case"] * 50,
            "donor": [f"D{i % 3}" for i in range(100)],
        },
    )

    config = {
        "use_rep": "X_nonexistent",
        "condition_col": "condition",
        "donor_col": "donor",
        "case": "case",
        "control": "control",
    }

    context = mock_context(tmp_path)
    method = MiloMethod()

    result = method._run(adata, config, context)

    assert isinstance(result, MethodSkip)
    assert "no reduced-dim rep" in result.reason.lower()


def test_milo_missing_case_control_skip(tmp_path, mock_context):
    """Test that missing case/control labels returns MethodSkip."""

    # AnnData with X_pca but no case/control in config
    adata = ad.AnnData(
        X=np.zeros((100, 10)),
        obs={
            "condition": ["control"] * 50 + ["case"] * 50,
            "donor": [f"D{i % 3}" for i in range(100)],
        },
        obsm={"X_pca": np.random.randn(100, 5)},
    )

    config = {
        "use_rep": "X_pca",
        "condition_col": "condition",
        "donor_col": "donor",
        # Missing case/control
    }

    context = mock_context(tmp_path)
    method = MiloMethod()

    result = method._run(adata, config, context)

    assert isinstance(result, MethodSkip)
    assert "case/control" in result.reason.lower()


def test_milo_registration():
    """Test that MiloMethod is registered in METHOD_REGISTRY."""
    from cellquorum.methods.registry import METHOD_REGISTRY

    assert METHOD_REGISTRY.has("differential_abundance", "milo")
    method_class = METHOD_REGISTRY.get("differential_abundance", "milo")
    assert (
        method_class is MiloMethod
        or isinstance(method_class, type)
        and issubclass(method_class, MiloMethod)
    )


def _annotated_milo_df() -> pd.DataFrame:
    """A Milo DA table with majority cell types (renderable as a beeswarm)."""
    return pd.DataFrame(
        {
            "nhood": [1, 2, 3, 4],
            "logFC": [2.0, 1.5, -2.0, 0.1],
            "PValue": [0.001, 0.01, 0.001, 0.5],
            "SpatialFDR": [0.02, 0.20, 0.03, 0.60],
            "nhood_size": [50, 40, 45, 30],
            "majority_celltype": ["TypeA", "TypeA", "TypeB", "TypeB"],
            "celltype_fraction": [0.9, 0.8, 0.95, 0.7],
        }
    )


def test_milo_beeswarm_helper_emits_figure(tmp_path, mock_context):
    """The beeswarm helper renders a figure artifact to disk (no R needed)."""

    context = mock_context(tmp_path)
    artifacts = MiloMethod()._beeswarm_artifacts(
        _annotated_milo_df(),
        case="case",
        control="control",
        spatial_fdr=0.1,
        config={},
        context=context,
    )

    assert artifacts, "expected a beeswarm figure artifact"
    assert all(a.kind == "figure" and a.name == "da_milo_beeswarm" for a in artifacts)
    suffixes = set()
    for a in artifacts:
        assert Path(a.path).exists()
        suffixes.add(Path(a.path).suffix)
    # save_figure writes dual PDF + PNG.
    assert suffixes == {".pdf", ".png"}


def test_milo_beeswarm_helper_respects_disable_flag(tmp_path, mock_context):
    """Setting write_da_figure=False suppresses the beeswarm figure."""

    context = mock_context(tmp_path)
    artifacts = MiloMethod()._beeswarm_artifacts(
        _annotated_milo_df(),
        case="case",
        control="control",
        spatial_fdr=0.1,
        config={"write_da_figure": False},
        context=context,
    )
    assert artifacts == []


def test_milo_beeswarm_helper_skips_when_unannotated(tmp_path, mock_context):
    """No majority cell type → nothing to place on the axis → no figure."""

    df = _annotated_milo_df()
    df["majority_celltype"] = np.nan

    context = mock_context(tmp_path)
    artifacts = MiloMethod()._beeswarm_artifacts(
        df,
        case="case",
        control="control",
        spatial_fdr=0.1,
        config={},
        context=context,
    )
    assert artifacts == []
