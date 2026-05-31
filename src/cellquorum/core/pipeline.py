"""Execution-frame runner for CellQuorum."""

from __future__ import annotations

# Import dataclass for structured run result objects.
from dataclasses import dataclass

# Import datetime so run metadata can include a UTC timestamp.
from datetime import UTC, datetime

# Import Path for output directory handling.
from pathlib import Path

# Import Any for JSON-like runtime metadata typing.
from typing import Any

# Import pandas so stage and backend plans can be written as tables.
import pandas as pd

# Import the backend registry builder.
from cellquorum.backends.registry import BackendRegistry, build_default_backend_registry

# Import config loading and saving utilities.
from cellquorum.config.loader import load_config, save_resolved_config

# Import the validated top-level config model.
from cellquorum.config.models import CellQuorumConfig

# Import the artifact manager.
from cellquorum.core.artifacts import ArtifactManager

# Import pipeline context objects.
from cellquorum.core.context import PipelineContext, PipelinePaths

# Import planner utilities.
from cellquorum.core.planner import PipelinePlan, build_pipeline_plan

# Import stage lifecycle records.
from cellquorum.core.stage import StageExecutionRecord


@dataclass(frozen=True)
class PipelineRunResult:
    """
    Store the result of a CellQuorum execution-frame bootstrap.

    This object captures the validated configuration, stage plan, backend-aware
    execution context, and provenance artifacts generated before analysis stages
    begin. The purpose is not to run a toy workflow. The purpose is to create the
    reproducible infrastructure that every serious CellQuorum stage will depend
    on: standardized paths, backend discovery, validated configuration, stage
    planning, and machine-readable provenance.

    Args:
        config: Validated CellQuorum runtime configuration.
        plan: Pipeline plan generated from the validated configuration and backend registry.
        context: Pipeline context containing paths, config, backend registry, and metadata.
        artifacts: Artifact manager containing provenance artifacts written during bootstrap.
    """

    # Store the validated runtime configuration.
    config: CellQuorumConfig

    # Store the generated pipeline plan.
    plan: PipelinePlan

    # Store the initialized pipeline context.
    context: PipelineContext

    # Store the artifact manager used during bootstrap.
    artifacts: ArtifactManager


def resolve_output_dir(config: CellQuorumConfig, output_dir: str | Path | None = None) -> Path:
    """
    Resolve the output directory for a CellQuorum run.

    CellQuorum should never scatter outputs into arbitrary working directories.
    This function applies a clear priority order: explicit caller override first,
    config-level output directory second, and a project-named directory under
    `paths.run_root` third. If none of those are available, execution fails
    before any files are written.

    Args:
        config: Validated CellQuorum configuration.
        output_dir: Optional explicit output directory override.

    Returns:
        Absolute output directory path.

    Raises:
        TypeError: If config is not a CellQuorumConfig.
        ValueError: If no output directory can be resolved.
    """

    # Validate the config type early so downstream errors are clear.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "resolve_output_dir expected a CellQuorumConfig object. "
            f"Received: {type(config).__name__}"
        )

    # Use the explicit output directory when provided by the caller.
    if output_dir is not None:
        # Return the explicit output directory as an absolute Path.
        return Path(output_dir).expanduser().resolve()

    # Use the config output directory when provided.
    if config.paths.output_dir is not None:
        # Return the config output directory as an absolute Path.
        return config.paths.output_dir.expanduser().resolve()

    # Use the run root and project name when a run root is configured.
    if config.paths.run_root is not None:
        # Return a default run directory under the configured run root.
        return (config.paths.run_root / config.project.name).expanduser().resolve()

    # Raise a clear error when no output location exists.
    raise ValueError(
        "Could not resolve an output directory. Provide output_dir, set "
        "paths.output_dir, or set paths.run_root in the CellQuorum configuration."
    )


def build_pipeline_context(
    config: CellQuorumConfig,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
) -> PipelineContext:
    """
    Build the runtime context for a CellQuorum run.

    The context is the object every stage will receive. It centralizes validated
    configuration, standardized paths, backend availability, run identity, random
    seed, and runtime metadata. This keeps future QC, preprocessing, annotation,
    R-backed, GPU-backed, and report-generation stages from each inventing their
    own execution state.

    Args:
        config: Validated CellQuorum configuration.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for tests or custom execution.

    Returns:
        Initialized PipelineContext.

    Raises:
        TypeError: If config is not a CellQuorumConfig.
    """

    # Validate the config type early.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "build_pipeline_context expected a CellQuorumConfig object. "
            f"Received: {type(config).__name__}"
        )

    # Resolve the output directory for this run.
    resolved_output_dir = resolve_output_dir(config, output_dir)

    # Build the standardized run path layout.
    paths = PipelinePaths.from_output_dir(resolved_output_dir)

    # Create all standard run directories.
    paths.ensure_directories()

    # Use the supplied backend registry or build the default CellQuorum registry.
    resolved_backend_registry = backend_registry or build_default_backend_registry()

    # Choose the run identifier from config or project name.
    run_id = config.run.run_id or config.project.name

    # Build runtime metadata for provenance and reporting.
    metadata: dict[str, Any] = {
        "project_name": config.project.name,
        "profile": config.run.profile,
        "organism": config.project.organism,
        "species_id": config.project.species_id,
        "bootstrap_time_utc": datetime.now(UTC).isoformat(),
    }

    # Build and return the pipeline context.
    return PipelineContext(
        config=config,
        paths=paths,
        backend_registry=resolved_backend_registry,
        run_id=run_id,
        random_seed=config.run.random_seed,
        metadata=metadata,
    )


def _stage_plan_dataframe(plan: PipelinePlan) -> pd.DataFrame:
    """
    Convert the planned stage list into a DataFrame.

    Stage plans should be available as both JSON and CSV. JSON is better for
    structured provenance. CSV is better for quick inspection, reports, and
    downstream workflow wrappers.

    Args:
        plan: Pipeline plan generated by CellQuorum.

    Returns:
        DataFrame containing stage name, enabled flag, status, and reason.
    """

    # Convert each planned stage into a table row.
    rows = [
        {
            "name": stage.name,
            "enabled": stage.enabled,
            "status": stage.status,
            "reason": stage.reason,
        }
        for stage in plan.stages
    ]

    # Return the stage plan table.
    return pd.DataFrame(rows, columns=["name", "enabled", "status", "reason"])


def _backend_status_dataframe(plan: PipelinePlan) -> pd.DataFrame:
    """
    Convert backend status rows into a DataFrame.

    Backend status is written as a simple table so users can quickly see which
    execution families are available in the current environment.

    Args:
        plan: Pipeline plan containing backend status rows.

    Returns:
        DataFrame containing backend availability summary.
    """

    # Convert each backend row into a simplified table row.
    rows = [
        {
            "name": row["name"],
            "kind": row["kind"],
            "available": row["available"],
            "missing": "; ".join(str(item) for item in row["missing"]),
            "warnings": "; ".join(str(item) for item in row["warnings"]),
        }
        for row in plan.backend_status_table
    ]

    # Return the backend status table.
    return pd.DataFrame(rows, columns=["name", "kind", "available", "missing", "warnings"])


def _stage_execution_records_dataframe(
    records: list[StageExecutionRecord],
) -> pd.DataFrame:
    """
    Convert stage execution records into a compact DataFrame.

    The JSON artifact stores full nested execution records. The CSV artifact
    stores a flattened summary that is easy to inspect in reports, spreadsheets,
    and workflow logs.

    Args:
        records: Stage execution records to summarize.

    Returns:
        DataFrame containing one row per stage execution record.
    """

    # Convert each execution record into a compact table row.
    rows = [
        {
            "stage_name": record.stage_name,
            "status": record.status,
            "started_at_utc": record.started_at_utc.isoformat(),
            "ended_at_utc": record.ended_at_utc.isoformat(),
            "duration_seconds": record.duration_seconds,
            "backend_used": record.backend_used,
            "n_input_artifacts": len(record.input_artifacts),
            "n_output_artifacts": len(record.output_artifacts),
            "n_notes": len(record.notes),
            "n_warnings": len(record.warnings),
            "has_skip_reason": record.skip_reason is not None,
            "has_error": record.error is not None,
        }
        for record in records
    ]

    # Return the execution records table.
    return pd.DataFrame(
        rows,
        columns=[
            "stage_name",
            "status",
            "started_at_utc",
            "ended_at_utc",
            "duration_seconds",
            "backend_used",
            "n_input_artifacts",
            "n_output_artifacts",
            "n_notes",
            "n_warnings",
            "has_skip_reason",
            "has_error",
        ],
    )


def write_pipeline_provenance(
    *,
    config: CellQuorumConfig,
    plan: PipelinePlan,
    context: PipelineContext,
    stage_execution_records: list[StageExecutionRecord] | None = None,
) -> ArtifactManager:
    """
    Write initial CellQuorum provenance artifacts.

    Every run should begin with auditable provenance before heavy analysis
    starts. This function writes the validated config, full pipeline plan,
    stage plan table, backend status table, planner warnings, run metadata,
    stage execution records, and artifact manifest.

    Args:
        config: Validated CellQuorum configuration.
        plan: Generated pipeline plan.
        context: Initialized pipeline context.
        stage_execution_records: Optional stage lifecycle records to write.

    Returns:
        ArtifactManager containing the registered provenance artifacts.

    Raises:
        TypeError: If inputs are not the expected CellQuorum runtime objects.
    """

    # Validate the config type early.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "write_pipeline_provenance expected config to be a CellQuorumConfig. "
            f"Received: {type(config).__name__}"
        )

    # Validate the plan type early.
    if not isinstance(plan, PipelinePlan):
        raise TypeError(
            "write_pipeline_provenance expected plan to be a PipelinePlan. "
            f"Received: {type(plan).__name__}"
        )

    # Validate the context type early.
    if not isinstance(context, PipelineContext):
        raise TypeError(
            "write_pipeline_provenance expected context to be a PipelineContext. "
            f"Received: {type(context).__name__}"
        )

    # Normalize stage execution records.
    records = [] if stage_execution_records is None else list(stage_execution_records)

    # Validate every stage execution record.
    for record in records:
        # Raise a clear error if the list contains the wrong type.
        if not isinstance(record, StageExecutionRecord):
            raise TypeError(
                "stage_execution_records must contain StageExecutionRecord objects. "
                f"Received: {type(record).__name__}"
            )

    # Create an artifact manager rooted at the run directory.
    artifact_manager = ArtifactManager.from_root(context.paths.root)

    # Save the validated runtime config as JSON provenance.
    save_resolved_config(
        config,
        context.paths.provenance / "resolved_config.json",
    )

    # Register the resolved config artifact.
    artifact_manager.register(
        name="resolved_config",
        relative_path="provenance/resolved_config.json",
        kind="json",
        description="Validated CellQuorum runtime configuration.",
    )

    # Write the full pipeline plan as JSON.
    artifact_manager.write_json(
        plan.to_dict(),
        name="pipeline_plan",
        relative_path="provenance/pipeline_plan.json",
        description="Full stage-level execution plan and backend availability summary.",
    )

    # Write the stage plan as CSV.
    artifact_manager.write_dataframe(
        _stage_plan_dataframe(plan),
        name="stage_plan",
        relative_path="provenance/stage_plan.csv",
        description="Tabular stage-level execution plan.",
        index=False,
    )

    # Write the backend status as JSON.
    artifact_manager.write_json(
        plan.backend_status_table,
        name="backend_status_json",
        relative_path="provenance/backend_status.json",
        description="Structured backend availability report.",
    )

    # Write the backend status as CSV.
    artifact_manager.write_dataframe(
        _backend_status_dataframe(plan),
        name="backend_status_table",
        relative_path="provenance/backend_status.csv",
        description="Tabular backend availability report.",
        index=False,
    )

    # Write planner warnings as JSON.
    artifact_manager.write_json(
        {"warnings": plan.warnings},
        name="planner_warnings",
        relative_path="provenance/planner_warnings.json",
        description="Planner-level warnings generated before pipeline execution.",
    )

    # Write run metadata as JSON.
    artifact_manager.write_json(
        {
            "run_id": context.run_id,
            "random_seed": context.random_seed,
            "paths": {
                "root": str(context.paths.root),
                "results": str(context.paths.results),
                "figures": str(context.paths.figures),
                "reports": str(context.paths.reports),
                "objects": str(context.paths.objects),
                "provenance": str(context.paths.provenance),
                "logs": str(context.paths.logs),
                "scratch": str(context.paths.scratch),
            },
            "metadata": dict(context.metadata),
        },
        name="run_metadata",
        relative_path="provenance/run_metadata.json",
        description="Run identity, standardized paths, and runtime metadata.",
    )

    # Write the full stage execution records as JSON.
    artifact_manager.write_json(
        [record.to_dict() for record in records],
        name="stage_execution_records_json",
        relative_path="provenance/stage_execution_records.json",
        description="Structured lifecycle records for stage execution decisions.",
    )

    # Write the compact stage execution records table as CSV.
    artifact_manager.write_dataframe(
        _stage_execution_records_dataframe(records),
        name="stage_execution_records_table",
        relative_path="provenance/stage_execution_records.csv",
        description="Tabular lifecycle records for stage execution decisions.",
        index=False,
    )

    # Write the artifact manifest.
    artifact_manager.write_manifest()

    # Return the artifact manager for programmatic inspection.
    return artifact_manager


def bootstrap_pipeline_run(
    config: CellQuorumConfig,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
) -> PipelineRunResult:
    """
    Bootstrap a CellQuorum pipeline run.

    This function creates the execution frame for a serious CellQuorum analysis.
    It does not pretend to perform QC or downstream biology yet. Instead, it does
    the part that must be correct before any analysis is trustworthy: validate
    configuration, construct standardized paths, discover backends, build the
    stage plan, and write provenance artifacts.

    Args:
        config: Validated CellQuorum configuration.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for tests or custom execution.

    Returns:
        PipelineRunResult containing config, plan, context, and provenance artifacts.
    """

    # Mark the start of the bootstrap lifecycle.
    bootstrap_started_at = datetime.now(UTC)

    # Build the initialized pipeline context.
    context = build_pipeline_context(
        config,
        output_dir=output_dir,
        backend_registry=backend_registry,
    )

    # Build the pipeline plan using the context backend registry.
    plan = build_pipeline_plan(
        config,
        backend_registry=context.backend_registry,
    )

    # Mark the end of the bootstrap lifecycle before provenance writing.
    bootstrap_ended_at = datetime.now(UTC)

    # Build the bootstrap execution record.
    bootstrap_record = StageExecutionRecord(
        stage_name="bootstrap",
        status="success",
        started_at_utc=bootstrap_started_at,
        ended_at_utc=bootstrap_ended_at,
        duration_seconds=(bootstrap_ended_at - bootstrap_started_at).total_seconds(),
        backend_used="python",
        notes=["Initialized CellQuorum execution frame."],
        warnings=list(plan.warnings),
        metrics={
            "n_planned_stages": len(plan.stages),
            "n_enabled_stages": len(plan.enabled_stage_names()),
            "n_backend_status_rows": len(plan.backend_status_table),
        },
    )

    # Write initial provenance artifacts.
    artifacts = write_pipeline_provenance(
        config=config,
        plan=plan,
        context=context,
        stage_execution_records=[bootstrap_record],
    )

    # Return the pipeline run result.
    return PipelineRunResult(
        config=config,
        plan=plan,
        context=context,
        artifacts=artifacts,
    )


def bootstrap_pipeline_run_from_config_file(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
) -> PipelineRunResult:
    """
    Load a config file and bootstrap a CellQuorum pipeline run.

    This is the file-based entry point used by the CLI and future workflow
    wrappers. It loads and validates YAML configuration before creating the
    execution frame.

    Args:
        config_path: Path to a CellQuorum YAML configuration file.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for tests or custom execution.

    Returns:
        PipelineRunResult containing config, plan, context, and provenance artifacts.
    """

    # Load and validate the configuration file.
    config = load_config(config_path)

    # Bootstrap the pipeline run from the validated config.
    return bootstrap_pipeline_run(
        config,
        output_dir=output_dir,
        backend_registry=backend_registry,
    )
