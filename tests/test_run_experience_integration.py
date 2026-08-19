"""Integration tests for run experience (reporter + executor + pipeline wiring)."""

from __future__ import annotations

import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console

from cellquorum.backends.base import BaseBackend
from cellquorum.backends.registry import BackendRegistry
from cellquorum.config.models import CellQuorumConfig
from cellquorum.core.executor import PipelineExecutor
from cellquorum.core.pipeline import build_pipeline_context
from cellquorum.core.planner import PipelinePlan, PlannedStage
from cellquorum.core.run_reporter import RunReporter


def build_test_backend_registry() -> BackendRegistry:
    """Build a deterministic backend registry."""
    registry = BackendRegistry()
    registry.register(BaseBackend(name="python", kind="python"))
    return registry


def make_test_adata() -> ad.AnnData:
    """Build a small AnnData object for executor tests."""
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
    var = pd.DataFrame(index=["MT-ND1", "ACTB", "RPS3", "MALAT1"])
    return ad.AnnData(X=matrix, obs=obs, var=var)


def build_test_config(*, qc_mode: str = "flag_no_drop") -> CellQuorumConfig:
    """Build a deterministic CellQuorum config for executor tests."""
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


def test_executor_with_verbose_reporter_emits_stage_markers(tmp_path: Path):
    """Verify executor with verbose reporter emits per-stage ▶/✓ lines."""
    # Build a context + plan.
    context = build_pipeline_context(
        build_test_config(qc_mode="flag_no_drop"),
        output_dir=tmp_path / "run",
        backend_registry=build_test_backend_registry(),
    ).with_adata(make_test_adata())

    plan = PipelinePlan(
        profile="standard",
        stages=[
            PlannedStage(name="qc", enabled=True, status="enabled", reason="test"),
        ],
    )

    # Build a reporter writing to a captured buffer.
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    reporter = RunReporter(verbose=True, console=console)

    # Run the executor with the reporter.
    executor = PipelineExecutor()
    result = executor.run(context=context, plan=plan, reporter=reporter)

    # Check that execution succeeded.
    assert len(result.succeeded_stage_names()) == 1
    assert "qc" in result.succeeded_stage_names()

    # Check that the reporter emitted stage markers.
    output = buf.getvalue()
    assert "▶ qc" in output  # stage_start
    assert "✓ qc" in output  # stage_end


def test_executor_with_noop_reporter_produces_no_output(tmp_path: Path):
    """Verify executor with a no-op reporter produces zero output."""
    # Build a context + plan.
    context = build_pipeline_context(
        build_test_config(qc_mode="flag_no_drop"),
        output_dir=tmp_path / "run",
        backend_registry=build_test_backend_registry(),
    ).with_adata(make_test_adata())

    plan = PipelinePlan(
        profile="standard",
        stages=[
            PlannedStage(name="qc", enabled=True, status="enabled", reason="test"),
        ],
    )

    # Build a no-op reporter writing to a captured buffer.
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    reporter = RunReporter(verbose=False, console=console)

    # Run the executor with the no-op reporter.
    executor = PipelineExecutor()
    result = executor.run(context=context, plan=plan, reporter=reporter)

    # Check that execution succeeded.
    assert len(result.succeeded_stage_names()) == 1

    # Check that the reporter produced zero output.
    output = buf.getvalue()
    assert output == ""


def test_executor_results_identical_with_verbose_and_noop_reporter(tmp_path: Path):
    """Verify that verbose and no-op reporters produce identical results."""
    # Build a shared context + plan.
    config = build_test_config(qc_mode="flag_no_drop")
    plan = PipelinePlan(
        profile="standard",
        stages=[
            PlannedStage(name="qc", enabled=True, status="enabled", reason="test"),
        ],
    )

    # Run with verbose reporter.
    context_verbose = build_pipeline_context(
        config,
        output_dir=tmp_path / "run_verbose",
        backend_registry=build_test_backend_registry(),
    ).with_adata(make_test_adata())

    buf_verbose = io.StringIO()
    console_verbose = Console(file=buf_verbose, force_terminal=False, width=100)
    reporter_verbose = RunReporter(verbose=True, console=console_verbose)

    executor_verbose = PipelineExecutor()
    result_verbose = executor_verbose.run(
        context=context_verbose, plan=plan, reporter=reporter_verbose
    )

    # Run with no-op reporter.
    context_noop = build_pipeline_context(
        config,
        output_dir=tmp_path / "run_noop",
        backend_registry=build_test_backend_registry(),
    ).with_adata(make_test_adata())

    buf_noop = io.StringIO()
    console_noop = Console(file=buf_noop, force_terminal=False, width=100)
    reporter_noop = RunReporter(verbose=False, console=console_noop)

    executor_noop = PipelineExecutor()
    result_noop = executor_noop.run(context=context_noop, plan=plan, reporter=reporter_noop)

    # Check that results are identical (same stage names, same statuses).
    assert result_verbose.succeeded_stage_names() == result_noop.succeeded_stage_names()
    assert result_verbose.skipped_stage_names() == result_noop.skipped_stage_names()
    assert result_verbose.failed_stage_names() == result_noop.failed_stage_names()

    # Check that both produced an AnnData result.
    assert result_verbose.context.adata is not None
    assert result_noop.context.adata is not None

    # Check that output differs (verbose emits, noop doesn't).
    assert len(buf_verbose.getvalue()) > 0
    assert buf_noop.getvalue() == ""


def test_executor_without_reporter_arg_still_works(tmp_path: Path):
    """Verify backward compat: executor.run() with NO reporter arg works."""
    # Build a context + plan.
    context = build_pipeline_context(
        build_test_config(qc_mode="flag_no_drop"),
        output_dir=tmp_path / "run",
        backend_registry=build_test_backend_registry(),
    ).with_adata(make_test_adata())

    plan = PipelinePlan(
        profile="standard",
        stages=[
            PlannedStage(name="qc", enabled=True, status="enabled", reason="test"),
        ],
    )

    # Run executor with NO reporter argument (backward compat).
    executor = PipelineExecutor()
    result = executor.run(context=context, plan=plan)

    # Check that execution succeeded.
    assert len(result.succeeded_stage_names()) == 1
    assert "qc" in result.succeeded_stage_names()


def test_config_echo_with_planned_stage_names_shows_only_runnable_stages():
    """Verify config_echo shows only planned+registered stages, not reserved slots."""
    # Build a config with reference_mapping enabled.
    cfg = CellQuorumConfig.model_validate(
        {
            "project": {"name": "test"},
            "stages": {
                "reference_mapping": True,
                "state_scoring": True,  # RESERVED (not implemented).
            },
        }
    )

    # Build a reporter.
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    reporter = RunReporter(verbose=True, console=console)

    # Echo with planned_stage_names = only reference_mapping (registered).
    reporter.config_echo(cfg, planned_stage_names=["reference_mapping"])

    output = buf.getvalue()

    # Check that reference_mapping appears.
    assert "reference_mapping" in output

    # Check that state_scoring does NOT appear (reserved, unimplemented).
    assert "state_scoring" not in output
