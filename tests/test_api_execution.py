"""Tests for public API execution behavior."""

from __future__ import annotations

# Import Path for temporary output paths.
from pathlib import Path

# Import AnnData for h5ad fixture construction.
import anndata as ad

# Import NumPy for deterministic matrices.
import numpy as np

# Import pandas for AnnData metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import the public API entry point.
from cellquorum import run_pipeline

# Import backend primitives for deterministic test registries.
from cellquorum.backends.base import BaseBackend

# Import backend registry for deterministic test registries.
from cellquorum.backends.registry import BackendRegistry

# Import the validated top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import run result type for API assertions.
from cellquorum.core.pipeline import PipelineRunResult


def build_test_backend_registry() -> BackendRegistry:
    """
    Build a deterministic backend registry.

    Returns:
        BackendRegistry containing one available Python backend.
    """

    # Create an empty registry.
    registry = BackendRegistry()

    # Register a simple Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Return the deterministic registry.
    return registry


def make_test_adata() -> ad.AnnData:
    """
    Build a small deterministic AnnData object.

    Returns:
        AnnData object suitable for QC execution tests.
    """

    # Build a deterministic count matrix.
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])

    # Build variable metadata.
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "RPS3", "MALAT1"])

    # Return AnnData.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def write_test_h5ad(tmp_path: Path) -> Path:
    """
    Write a deterministic h5ad input file.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Path to the written h5ad file.
    """

    # Build the h5ad path.
    h5ad_path = tmp_path / "input.h5ad"

    # Write AnnData to disk.
    make_test_adata().write_h5ad(h5ad_path)

    # Return the h5ad path.
    return h5ad_path


def build_execution_config(h5ad_path: Path) -> CellQuorumConfig:
    """
    Build a deterministic config for public API execution tests.

    Args:
        h5ad_path: h5ad input path.

    Returns:
        Validated CellQuorumConfig.
    """

    # Return a config with deterministic QC behavior.
    return CellQuorumConfig(
        project={
            "name": "api_execution_project",
        },
        input={
            "h5ad": str(h5ad_path),
        },
        compute={
            "backend": "cpu",
            "prefer_gpu": False,
            "fallback_to_cpu": True,
        },
        r={
            "enabled": False,
        },
        qc={
            "mode": "report_only",
            "threshold_strategy": "fixed",
            "metrics": {
                "percent_top": [2],
            },
            "basic": {
                "min_genes_per_cell": 2,
                "min_cells_per_gene": 2,
                "max_mito_percent": 60.0,
            },
            "mad": {
                "enabled": False,
            },
            "outputs": {
                "write_h5ad": False,
                "write_figures": False,
            },
        },
    )


def test_run_pipeline_executes_by_default_for_config_model(tmp_path: Path) -> None:
    """
    Verify run_pipeline executes registered stages by default for config models.
    """

    # Write a deterministic h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a validated config.
    config = build_execution_config(h5ad_path)

    # Run through the public API.
    result = run_pipeline(
        config,
        output_dir=tmp_path / "api_execute_model",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm the public API returned a structured result.
    assert isinstance(result, PipelineRunResult)

    # Confirm execution happened.
    assert result.execution_result is not None
    assert "qc" in result.execution_result.succeeded_stage_names()

    # Confirm final AnnData contains QC annotations.
    assert isinstance(result.context.adata, ad.AnnData)
    assert "cellquorum_qc_keep" in result.context.adata.obs


def test_run_pipeline_can_bootstrap_only_for_config_model(tmp_path: Path) -> None:
    """
    Verify run_pipeline preserves bootstrap-only behavior when execute is False.
    """

    # Write a deterministic h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a validated config.
    config = build_execution_config(h5ad_path)

    # Run bootstrap-only through the public API.
    result = run_pipeline(
        config,
        output_dir=tmp_path / "api_bootstrap_model",
        backend_registry=build_test_backend_registry(),
        execute=False,
    )

    # Confirm no execution result is attached.
    assert result.execution_result is None

    # Confirm bootstrap did not load AnnData.
    assert result.context.adata is None

    # Confirm provenance was still written.
    assert (result.context.paths.provenance / "stage_execution_records.json").exists()


def test_run_pipeline_executes_by_default_for_dictionary(tmp_path: Path) -> None:
    """
    Verify run_pipeline executes registered stages by default for dict configs.
    """

    # Write a deterministic h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Run through the public API with a dictionary config.
    result = run_pipeline(
        {
            "project": {
                "name": "api_dict_execution_project",
            },
            "input": {
                "h5ad": str(h5ad_path),
            },
            "compute": {
                "backend": "cpu",
                "prefer_gpu": False,
                "fallback_to_cpu": True,
            },
            "r": {
                "enabled": False,
            },
            "qc": {
                "mode": "report_only",
                "threshold_strategy": "fixed",
                "metrics": {
                    "percent_top": [2],
                },
                "basic": {
                    "min_genes_per_cell": 2,
                    "min_cells_per_gene": 2,
                    "max_mito_percent": 60.0,
                },
                "mad": {
                    "enabled": False,
                },
                "outputs": {
                    "write_h5ad": False,
                    "write_figures": False,
                },
            },
        },
        output_dir=tmp_path / "api_execute_dict",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm execution happened.
    assert result.execution_result is not None
    assert "qc" in result.execution_result.succeeded_stage_names()


def test_run_pipeline_executes_by_default_for_yaml_path(tmp_path: Path) -> None:
    """
    Verify run_pipeline executes registered stages by default for YAML configs.
    """

    # Write a deterministic h5ad file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config path.
    config_path = tmp_path / "config.yaml"

    # Write a YAML config.
    config_path.write_text(
        f"""
project:
  name: api_yaml_execution_project
input:
  h5ad: {h5ad_path}
compute:
  backend: cpu
  prefer_gpu: false
  fallback_to_cpu: true
r:
  enabled: false
qc:
  mode: report_only
  threshold_strategy: fixed
  metrics:
    percent_top: [2]
  basic:
    min_genes_per_cell: 2
    min_cells_per_gene: 2
    max_mito_percent: 60.0
  mad:
    enabled: false
  outputs:
    write_h5ad: false
    write_figures: false
""",
        encoding="utf-8",
    )

    # Run through the public API with a YAML config path.
    result = run_pipeline(
        config_path,
        output_dir=tmp_path / "api_execute_yaml",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm execution happened.
    assert result.execution_result is not None
    assert "qc" in result.execution_result.succeeded_stage_names()

    # Confirm final AnnData contains QC annotations.
    assert isinstance(result.context.adata, ad.AnnData)
    assert "cellquorum_qc_keep" in result.context.adata.obs


def test_run_pipeline_rejects_unsupported_config_input(tmp_path: Path) -> None:
    """
    Verify run_pipeline rejects unsupported config inputs.
    """

    # Confirm unsupported config input raises a clear error.
    with pytest.raises(TypeError, match="path, CellQuorumConfig, or dictionary"):
        run_pipeline(
            12345,  # type: ignore[arg-type]
            output_dir=tmp_path / "bad_api_run",
            backend_registry=build_test_backend_registry(),
        )
