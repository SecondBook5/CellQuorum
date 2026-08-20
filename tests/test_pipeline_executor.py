"""Tests for CellQuorum pipeline executor."""

from __future__ import annotations

# Import Path for temporary output path annotations.
from pathlib import Path

# Import AnnData for test stage return values and assertions.
import anndata as ad

# Import NumPy for deterministic matrices.
import numpy as np

# Import pandas for AnnData metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import backend primitives for deterministic context construction.
from cellquorum.backends.base import BaseBackend

# Import backend registry for deterministic context construction.
from cellquorum.backends.registry import BackendRegistry

# Import top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import pipeline context for helper return typing.
from cellquorum.core.context import PipelineContext

# Import executor objects under test.
from cellquorum.core.executor import (
    PipelineExecutionResult,
    PipelineExecutor,
    StageRegistry,
    build_default_stage_registry,
    validate_executor_inputs,
)

# Import pipeline context builder.
from cellquorum.core.pipeline import build_pipeline_context

# Import planning objects.
from cellquorum.core.planner import PipelinePlan, PlannedStage

# Import stage result for dummy test stages.
from cellquorum.core.stage import StageResult


def build_test_backend_registry() -> BackendRegistry:
    """
    Build a deterministic backend registry.

    Returns:
        BackendRegistry containing one available Python backend.
    """

    # Create an empty registry.
    registry = BackendRegistry()

    # Register an available Python backend.
    registry.register(BaseBackend(name="python", kind="python"))

    # Return the deterministic registry.
    return registry


def make_test_adata() -> ad.AnnData:
    """
    Build a small AnnData object for executor tests.

    Returns:
        AnnData object with deterministic values and names.
    """

    # Build a small count matrix.
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])

    # Build variable metadata with one mitochondrial gene.
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "RPS3", "MALAT1"])

    # Return AnnData.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def build_test_config(*, qc_mode: str = "flag_no_drop") -> CellQuorumConfig:
    """
    Build a deterministic CellQuorum config for executor tests.

    Args:
        qc_mode: QC execution mode.

    Returns:
        Validated CellQuorumConfig.
    """

    # Return a config with deterministic QC behavior.
    return CellQuorumConfig(
        project={
            "name": "executor_project",
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
            "mode": qc_mode,
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


def build_test_context(tmp_path: Path, *, qc_mode: str = "flag_no_drop") -> PipelineContext:
    """
    Build a PipelineContext containing AnnData.

    Args:
        tmp_path: Temporary test directory.
        qc_mode: QC execution mode.

    Returns:
        PipelineContext with AnnData loaded.
    """

    # Build a deterministic context without file input loading.
    context = build_pipeline_context(
        build_test_config(qc_mode=qc_mode),
        output_dir=tmp_path / "run",
        backend_registry=build_test_backend_registry(),
    )

    # Attach AnnData to the context.
    return context.with_adata(make_test_adata())


def make_plan(*planned_stages: PlannedStage) -> PipelinePlan:
    """
    Build a small PipelinePlan for executor tests.

    Args:
        planned_stages: PlannedStage objects in execution order.

    Returns:
        PipelinePlan containing supplied stages.
    """

    # Return a compact test plan.
    return PipelinePlan(
        profile="standard",
        stages=list(planned_stages),
        backend_status_table=[],
        warnings=[],
    )


def make_enabled_stage(name: str) -> PlannedStage:
    """
    Build an enabled planned stage.

    Args:
        name: Stage name.

    Returns:
        Enabled PlannedStage.
    """

    # Return enabled planned stage.
    return PlannedStage(
        name=name,
        enabled=True,
        status="enabled",
        reason="Enabled by test configuration.",
    )


def make_disabled_stage(name: str) -> PlannedStage:
    """
    Build a disabled planned stage.

    Args:
        name: Stage name.

    Returns:
        Disabled PlannedStage.
    """

    # Return disabled planned stage.
    return PlannedStage(
        name=name,
        enabled=False,
        status="disabled",
        reason="Disabled by test configuration.",
    )


class EchoStage:
    """
    Simple custom stage used to test registry-driven execution.
    """

    # Store the stage name.
    name = "echo"

    def run(self, context: object) -> StageResult:
        """
        Return the current AnnData object unchanged.

        Args:
            context: PipelineContext-like object.

        Returns:
            StageResult containing the existing AnnData.
        """

        # Return the existing AnnData with a test note.
        return StageResult(
            adata=context.require_adata(),  # type: ignore[attr-defined]
            notes=["Echo stage executed."],
            metrics={"echo": True},
        )


class FailingStage:
    """
    Simple custom stage used to test failure records.
    """

    # Store the stage name.
    name = "qc"

    def run(self, context: object) -> StageResult:
        """
        Raise a test failure.

        Args:
            context: PipelineContext-like object.

        Raises:
            RuntimeError: Always raised for failure-record tests.
        """

        # Raise a deterministic stage error.
        raise RuntimeError("Intentional executor test failure.")


def test_default_stage_registry_contains_qc() -> None:
    """
    Verify the default registry exposes the implemented stages.

    QC and preprocessing are the first real executable scientific stages.
    Phase-2A adds dimensionality and clustering.
    Phase-2B adds integration and annotation.
    Phase-2D adds ambient_correction.
    """

    # Build the default registry.
    registry = build_default_stage_registry()

    # Confirm all implemented stages are registered.
    assert registry.get("adjudication") is not None
    assert registry.get("ambient_correction") is not None
    assert registry.get("qc") is not None
    assert registry.get("preprocessing") is not None
    assert registry.get("dimensionality") is not None
    assert registry.get("clustering") is not None
    assert registry.get("integration") is not None
    assert registry.get("integration_benchmark") is not None
    assert registry.get("annotation") is not None
    assert registry.get("feature_selection") is not None
    assert registry.get("population_identity") is not None
    assert registry.get("reference_mapping") is not None
    # All implemented stages are registered (sorted alphabetically).
    assert registry.registered_stage_names() == [
        "adjudication",
        "ambient_correction",
        "annotation",
        "annotation_consensus",
        "annotation_diagnostics",
        "ccc_network",
        "ccc_viz",
        "cell_cell_communication",
        "clustering",
        "coexpression",
        "de_viz",
        "differential_abundance",
        "differential_expression",
        "dimensionality",
        "embeddings",
        "enrichment",
        "enrichment_viz",
        "feature_selection",
        "grn",
        "integration",
        "integration_benchmark",
        "multicellular_programs",
        "perturbation",
        "population_identity",
        "preprocessing",
        "qc",
        "reference_mapping",
        "subclustering",
        "trajectory",
        "trajectory_viz",
    ]


def test_stage_registry_with_stage_adds_custom_stage() -> None:
    """
    Verify StageRegistry can add custom stage implementations.

    This keeps the executor extensible for future preprocessing and analysis
    stages.
    """

    # Build a registry with a custom stage.
    registry = StageRegistry().with_stage(EchoStage())

    # Confirm the custom stage can be retrieved.
    assert isinstance(registry.get("echo"), EchoStage)
    assert registry.registered_stage_names() == ["echo"]


def test_stage_registry_rejects_stage_without_name() -> None:
    """
    Verify StageRegistry rejects invalid stage objects.

    Registered stages need stable names so the executor can match them to plans.
    """

    # Confirm invalid stage objects fail clearly.
    with pytest.raises(TypeError, match="string 'name'"):
        StageRegistry().with_stage(object())  # type: ignore[arg-type]


def test_validate_executor_inputs_rejects_invalid_context(tmp_path: Path) -> None:
    """
    Verify executor input validation rejects invalid contexts.
    """

    # Build a valid plan.
    plan = make_plan(make_enabled_stage("qc"))

    # Confirm invalid context fails clearly.
    with pytest.raises(TypeError, match="context to be a PipelineContext"):
        validate_executor_inputs(context=object(), plan=plan)  # type: ignore[arg-type]


def test_validate_executor_inputs_rejects_invalid_plan(tmp_path: Path) -> None:
    """
    Verify executor input validation rejects invalid plans.
    """

    # Build a valid context.
    context = build_test_context(tmp_path)

    # Confirm invalid plan fails clearly.
    with pytest.raises(TypeError, match="plan to be a PipelinePlan"):
        validate_executor_inputs(context=context, plan=object())  # type: ignore[arg-type]


def test_executor_runs_qc_stage_and_updates_context(tmp_path: Path) -> None:
    """
    Verify PipelineExecutor runs QCStage and propagates updated AnnData.

    This is the first real vertical slice: planned QC stage to executed QC stage.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a plan with QC enabled.
    plan = make_plan(make_enabled_stage("qc"))

    # Execute the plan.
    execution_result = PipelineExecutor().run(context=context, plan=plan)

    # Confirm result type.
    assert isinstance(execution_result, PipelineExecutionResult)

    # Confirm QC succeeded.
    assert execution_result.succeeded_stage_names() == ["qc"]
    assert execution_result.skipped_stage_names() == []
    assert execution_result.failed_stage_names() == []
    assert execution_result.has_failures() is False

    # Confirm QC result was stored.
    assert "qc" in execution_result.stage_results

    # Confirm context AnnData was preserved and updated.
    assert isinstance(execution_result.context.adata, ad.AnnData)
    assert execution_result.context.adata.shape == (3, 4)

    # Confirm QC annotations were added.
    assert "cellquorum_qc_keep" in execution_result.context.adata.obs
    assert "cellquorum_qc_keep" in execution_result.context.adata.var

    # Confirm the execution record captured artifacts.
    record = execution_result.stage_execution_records[0]
    assert record.status == "success"
    assert record.backend_used == "python"
    assert len(record.output_artifacts) > 0


def test_executor_records_disabled_stage_as_skipped(tmp_path: Path) -> None:
    """
    Verify disabled planned stages become skipped execution records.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a plan with QC disabled.
    plan = make_plan(make_disabled_stage("qc"))

    # Execute the plan.
    execution_result = PipelineExecutor().run(context=context, plan=plan)

    # Confirm QC was skipped.
    assert execution_result.succeeded_stage_names() == []
    assert execution_result.skipped_stage_names() == ["qc"]

    # Confirm skip reason came from the plan.
    record = execution_result.stage_execution_records[0]
    assert record.status == "skipped"
    assert record.skip_reason is not None
    assert record.skip_reason.reason == "Disabled by test configuration."

    # Confirm no stage result was stored.
    assert execution_result.stage_results == {}


def test_executor_records_unimplemented_enabled_stage_as_skipped(tmp_path: Path) -> None:
    """
    Verify enabled planned stages without implementations are skipped explicitly.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a plan with an enabled future stage (state_scoring not yet implemented).
    plan = make_plan(make_enabled_stage("state_scoring"))

    # Execute the plan.
    execution_result = PipelineExecutor().run(context=context, plan=plan)

    # Confirm state_scoring was skipped, not silently ignored.
    assert execution_result.succeeded_stage_names() == []
    assert execution_result.skipped_stage_names() == ["state_scoring"]

    # Confirm skip reason explains missing implementation.
    record = execution_result.stage_execution_records[0]
    assert record.status == "skipped"
    assert record.skip_reason is not None
    assert record.skip_reason.reason == "Stage is planned but no implementation is registered."


def test_executor_runs_custom_registered_stage(tmp_path: Path) -> None:
    """
    Verify PipelineExecutor can run a non-default registered stage.

    This proves the executor is a reusable spine, not hard-coded only to QC.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a custom registry.
    registry = StageRegistry().with_stage(EchoStage())

    # Build a plan for the custom stage.
    plan = make_plan(make_enabled_stage("echo"))

    # Execute the plan.
    execution_result = PipelineExecutor(registry=registry).run(
        context=context,
        plan=plan,
    )

    # Confirm the custom stage succeeded.
    assert execution_result.succeeded_stage_names() == ["echo"]
    assert execution_result.stage_results["echo"].notes == ["Echo stage executed."]
    assert execution_result.stage_results["echo"].metrics == {"echo": True}


def test_executor_records_failed_stage_and_stops_by_default(tmp_path: Path) -> None:
    """
    Verify PipelineExecutor records failed stages and stops by default.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a registry with a failing QC implementation.
    registry = StageRegistry(stages={"qc": FailingStage()})

    # Build a plan with failing QC followed by an unimplemented stage.
    plan = make_plan(
        make_enabled_stage("qc"),
        make_enabled_stage("preprocessing"),
    )

    # Execute the plan.
    execution_result = PipelineExecutor(registry=registry).run(
        context=context,
        plan=plan,
    )

    # Confirm only the failed QC record exists because execution stopped.
    assert execution_result.succeeded_stage_names() == []
    assert execution_result.skipped_stage_names() == []
    assert execution_result.failed_stage_names() == ["qc"]
    assert len(execution_result.stage_execution_records) == 1

    # Confirm the failure was structured.
    record = execution_result.stage_execution_records[0]
    assert record.status == "failed"
    assert record.error is not None
    assert record.error.error_type == "RuntimeError"
    assert record.error.message == "Intentional executor test failure."


def test_executor_can_continue_after_failure_when_configured(tmp_path: Path) -> None:
    """
    Verify PipelineExecutor can continue after failures when requested.
    """

    # Build context containing AnnData.
    context = build_test_context(tmp_path)

    # Build a registry with a failing QC implementation.
    registry = StageRegistry(stages={"qc": FailingStage()})

    # Build a plan with failing QC followed by an unimplemented stage.
    plan = make_plan(
        make_enabled_stage("qc"),
        make_enabled_stage("preprocessing"),
    )

    # Execute with stop_on_failure disabled.
    execution_result = PipelineExecutor(
        registry=registry,
        stop_on_failure=False,
    ).run(
        context=context,
        plan=plan,
    )

    # Confirm QC failed and preprocessing was still evaluated as skipped.
    assert execution_result.failed_stage_names() == ["qc"]
    assert execution_result.skipped_stage_names() == ["preprocessing"]
    assert len(execution_result.stage_execution_records) == 2
