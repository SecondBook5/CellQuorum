"""Tests for executing registered stages through the pipeline run layer."""

from __future__ import annotations

# Import JSON to inspect written provenance records.
import json

# Import Path for temporary file annotations.
from pathlib import Path

# Import AnnData for h5ad fixture construction.
import anndata as ad

# Import NumPy for deterministic matrices.
import numpy as np

# Import pandas for AnnData metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import backend primitives for deterministic backend registry construction.
from cellquorum.backends.base import BaseBackend

# Import backend registry for deterministic tests.
from cellquorum.backends.registry import BackendRegistry

# Import top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import executor result type.
from cellquorum.core.executor import PipelineExecutionResult

# Import pipeline run helpers under test.
from cellquorum.core.pipeline import (
    bootstrap_pipeline_run,
    execute_pipeline_run,
    execute_pipeline_run_from_config_file,
)


def build_test_backend_registry() -> BackendRegistry:
    """
    Build a deterministic backend registry.

    Returns:
        BackendRegistry containing one available Python backend.
    """

    # Create an empty registry.
    registry = BackendRegistry()

    # Register a simple available Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Return the deterministic registry.
    return registry


def make_test_adata() -> ad.AnnData:
    """
    Build a small deterministic AnnData object.

    Returns:
        AnnData object suitable for QC execution tests.
    """

    # Build a count matrix matching the QC fixed-threshold test shape.
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
        Path to written h5ad file.
    """

    # Build the input file path.
    h5ad_path = tmp_path / "input.h5ad"

    # Write AnnData to disk.
    make_test_adata().write_h5ad(h5ad_path)

    # Return the h5ad path.
    return h5ad_path


def build_execution_config(h5ad_path: Path | None) -> CellQuorumConfig:
    """
    Build a deterministic config for pipeline execution tests.

    Args:
        h5ad_path: Optional h5ad input path.

    Returns:
        Validated CellQuorumConfig.
    """

    # Build the input block.
    input_block: dict[str, object] = {}

    # Store the h5ad path when supplied.
    if h5ad_path is not None:
        input_block["h5ad"] = str(h5ad_path)

    # Return a deterministic config.
    return CellQuorumConfig(
        project={
            "name": "execution_project",
        },
        input=input_block,
        compute={
            "backend": "cpu",
            "prefer_gpu": False,
            "fallback_to_cpu": True,
        },
        r={
            "enabled": False,
        },
        # Use the env-independent cp10k recipe: these tests exercise pipeline
        # threading, not the transform, and the scclr-backed PFlog1pPF default
        # requires the isolated scclr environment.
        preprocessing={
            "normalization": {"recipe": "cellquorum_log1p_cp10k_v1"},
        },
        qc={
            "metrics": {
                "percent_top": [2],
            },
            "floors": {
                "min_genes_per_cell": 2,
                "min_cells_per_gene": 2,
            },
            "outputs": {
                "write_h5ad": False,
                "write_figures": False,
            },
        },
    )


def load_stage_execution_records(run_dir: Path) -> list[dict[str, object]]:
    """
    Load stage execution records from provenance.

    Args:
        run_dir: Pipeline run directory.

    Returns:
        List of execution record dictionaries.
    """

    # Build the stage execution records path.
    records_path = run_dir / "provenance" / "stage_execution_records.json"

    # Load and return the records.
    return json.loads(records_path.read_text(encoding="utf-8"))


def test_bootstrap_pipeline_run_remains_bootstrap_only(tmp_path: Path) -> None:
    """
    Verify bootstrap_pipeline_run still does not execute stages.

    This preserves the existing execution-frame setup behavior.
    """

    # Write an h5ad input file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config pointing to the h5ad input.
    config = build_execution_config(h5ad_path)

    # Bootstrap the run without stage execution.
    result = bootstrap_pipeline_run(
        config,
        output_dir=tmp_path / "bootstrap_only",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm no executor result is attached.
    assert result.execution_result is None

    # Confirm bootstrap did not load input AnnData.
    assert result.context.adata is None

    # Confirm bootstrap provenance still exists.
    assert (tmp_path / "bootstrap_only" / "provenance" / "resolved_config.json").exists()


def test_execute_pipeline_run_loads_input_and_runs_qc(tmp_path: Path) -> None:
    """
    Verify execute_pipeline_run loads input AnnData and executes QCStage.

    This is the first full pipeline run path: config input to executed QC.
    """

    # Write an h5ad input file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config pointing to the h5ad input.
    config = build_execution_config(h5ad_path)

    # Execute the pipeline.
    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "executed_run",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm execution result exists.
    assert isinstance(result.execution_result, PipelineExecutionResult)

    # Confirm QC and preprocessing succeeded.
    assert "qc" in result.execution_result.succeeded_stage_names()
    assert "preprocessing" in result.execution_result.succeeded_stage_names()
    assert "qc" in result.execution_result.stage_results
    assert "preprocessing" in result.execution_result.stage_results

    # Confirm future stages were skipped explicitly (composition not yet implemented).
    assert "composition" in result.execution_result.skipped_stage_names()

    # Confirm final context contains AnnData.
    assert isinstance(result.context.adata, ad.AnnData)

    # Confirm QC annotations exist on final AnnData.
    assert "qc_floor_reason" in result.context.adata.obs
    assert "qc_state_initial" in result.context.adata.obs

    # Confirm provenance stage execution records were written. The first record
    # is always bootstrap; ambient_correction (disabled by default) is recorded
    # as a skip before qc, so locate the qc record by name rather than position.
    records = load_stage_execution_records(tmp_path / "executed_run")
    assert records[0]["stage_name"] == "bootstrap"
    qc_record = next(r for r in records if r["stage_name"] == "qc")
    assert qc_record["status"] == "success"


def test_execute_pipeline_run_records_failure_without_input(tmp_path: Path) -> None:
    """
    Verify execute_pipeline_run records QC failure when no AnnData is loaded.

    The executor should return structured failure records instead of producing
    unclear downstream errors.
    """

    # Build a config without an input file.
    config = build_execution_config(None)

    # Execute the pipeline.
    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "missing_input_run",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm execution result exists.
    assert result.execution_result is not None

    # Confirm QC failed because context.adata was missing. qc is the first
    # ENABLED executable stage (ambient_correction is disabled by default), so it
    # is the only failing stage.
    assert result.execution_result.failed_stage_names() == ["qc"]
    assert result.execution_result.has_failures() is True

    # Confirm provenance still records the failure. Locate qc by name (a disabled
    # ambient_correction skip record precedes it).
    records = load_stage_execution_records(tmp_path / "missing_input_run")
    assert records[0]["stage_name"] == "bootstrap"
    qc_record = next(r for r in records if r["stage_name"] == "qc")
    assert qc_record["status"] == "failed"
    assert qc_record["error"]["error_type"] in {"QCStageError", "CellQuorumDataError"}


def test_continue_on_stage_failure_attempts_stages_past_a_failure(tmp_path: Path) -> None:
    """
    Verify run.continue_on_stage_failure keeps executing after a failed stage.

    With no input, qc fails first. Under the default fail-fast policy the run
    stops there. With continue_on_stage_failure the executor must go on and
    attempt later mandatory stages, so more than one stage is recorded failed.
    This is the contract an unattended canary relies on to surface every break
    in a single pass instead of halting on the first.
    """

    # Build an input-less config and opt into continue-on-failure.
    config = build_execution_config(None)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"continue_on_stage_failure": True})}
    )

    # Execute the pipeline.
    result = execute_pipeline_run(
        config,
        output_dir=tmp_path / "continue_run",
        backend_registry=build_test_backend_registry(),
    )

    # qc still fails, but execution did not stop at it: at least one later stage
    # was attempted and also recorded (failed or otherwise), so the failure set
    # is not the single-element ["qc"] the fail-fast path produces.
    failed = result.execution_result.failed_stage_names()
    assert "qc" in failed
    assert len(failed) > 1
    """
    Verify config-file execution loads YAML and runs QC.

    This prepares the pipeline for future CLI execution.
    """

    # Write an h5ad input file.
    h5ad_path = write_test_h5ad(tmp_path)

    # Build a config path.
    config_path = tmp_path / "config.yaml"

    # Write a YAML config.
    config_path.write_text(
        f"""
project:
  name: yaml_execution_project
input:
  h5ad: {h5ad_path}
compute:
  backend: cpu
  prefer_gpu: false
  fallback_to_cpu: true
r:
  enabled: false
qc:
  metrics:
    percent_top: [2]
  floors:
    min_genes_per_cell: 2
    min_cells_per_gene: 2
  outputs:
    write_h5ad: false
    write_figures: false
""",
        encoding="utf-8",
    )

    # Execute from the YAML config.
    result = execute_pipeline_run_from_config_file(
        config_path,
        output_dir=tmp_path / "yaml_executed_run",
        backend_registry=build_test_backend_registry(),
    )

    # Confirm QC executed.
    assert result.execution_result is not None
    assert "qc" in result.execution_result.succeeded_stage_names()

    # Confirm input was loaded.
    assert isinstance(result.context.adata, ad.AnnData)

    # Confirm provenance exists.
    assert (tmp_path / "yaml_executed_run" / "provenance" / "stage_execution_records.json").exists()


def test_execute_pipeline_run_rejects_non_config(tmp_path: Path) -> None:
    """
    Verify execute_pipeline_run rejects invalid config objects.
    """

    # Confirm invalid config inputs fail clearly.
    with pytest.raises(TypeError, match="expected a CellQuorumConfig"):
        execute_pipeline_run(  # type: ignore[arg-type]
            object(),
            output_dir=tmp_path / "bad_run",
            backend_registry=build_test_backend_registry(),
        )
