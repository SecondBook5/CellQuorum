"""Execution-frame runner for CellQuorum."""

from __future__ import annotations

# Import importlib.metadata so dependency versions can be resolved from metadata.
import importlib.metadata

# Import logging so startup config/data-mismatch warnings surface where they occur.
import logging

# Import platform so run metadata can stamp the interpreter and OS environment.
import platform

# Import time for wall-clock measurement.
import time

# Import dataclass for structured run result objects.
from dataclasses import dataclass

# Import datetime so run metadata can include a UTC timestamp.
from datetime import UTC, datetime

# Import Path for output directory handling.
from pathlib import Path

# Import Any for JSON-like runtime metadata typing.
from typing import Any

# Import AnnData for configured input-loading return typing.
import anndata as ad

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

# Import the stage executor.
from cellquorum.core.executor import PipelineExecutionResult, PipelineExecutor

# Import the run-directory hygiene gate and the inherited-artifact report.
from cellquorum.core.output_hygiene import (
    InheritedArtifact,
    assert_output_dir_matches_config,
    find_inherited_artifacts,
    format_inherited_artifacts,
)

# Import planner utilities.
from cellquorum.core.planner import PipelinePlan, build_pipeline_plan

# Import runtime progress reporter.
from cellquorum.core.run_reporter import RunReporter

# Import stage lifecycle records.
from cellquorum.core.stage import StageExecutionRecord

# Import AnnData input loading utility.
from cellquorum.io import load_adata

# Import package version.
from cellquorum.version import __version__

logger = logging.getLogger(__name__)


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
        execution_result: Optional stage execution result for executed runs.
    """

    # Store the validated runtime configuration.
    config: CellQuorumConfig

    # Store the generated pipeline plan.
    plan: PipelinePlan

    # Store the initialized pipeline context.
    context: PipelineContext

    # Store the artifact manager used during bootstrap or execution.
    artifacts: ArtifactManager

    # Store stage execution results when stages have been executed.
    execution_result: PipelineExecutionResult | None = None


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


def load_input_adata_from_config(config: CellQuorumConfig) -> ad.AnnData | None:
    """
    Load the configured AnnData input file, if one is configured.

    This helper connects the validated top-level input config to the AnnData I/O
    layer. It intentionally returns None when no input path is configured so
    programmatic workflows can still inject AnnData directly into the context.

    Args:
        config: Validated CellQuorum configuration.

    Returns:
        Loaded AnnData object, or None when config.input.h5ad is omitted.

    Raises:
        TypeError: If config is not a CellQuorumConfig.
        AnnDataLoadError: If the configured h5ad path cannot be loaded.
    """

    # Validate the config type early so downstream errors are clear.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "load_input_adata_from_config expected a CellQuorumConfig object. "
            f"Received: {type(config).__name__}"
        )

    # Return None when no h5ad file is configured.
    if config.input.h5ad is None:
        return None

    # Apply a configured row restriction (e.g. cell_type == Fibroblasts) at load
    # time. The I/O layer reads it in backed mode so the full object is never
    # materialized; without a subset the whole object is read as before.
    subset = config.input.subset
    exclude = config.input.exclude
    if subset is None and exclude is None:
        return load_adata(config.input.h5ad)
    return load_adata(
        config.input.h5ad,
        subset_column=subset.column if subset else None,
        subset_values=subset.values if subset else None,
        agreement_column=subset.require_agreement if subset else None,
        exclude_column=exclude.column if exclude else None,
        exclude_values=exclude.values if exclude else None,
    )


def build_pipeline_context(
    config: CellQuorumConfig,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
    load_input: bool = False,
) -> PipelineContext:
    """
    Build the runtime context for a CellQuorum run.

    The context is the object every stage will receive. It centralizes validated
    configuration, standardized paths, optional AnnData input, backend
    availability, run identity, random seed, and runtime metadata. This keeps
    future QC, preprocessing, annotation,
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
    # Thread the configured Rscript path so R availability reflects where R
    # actually lives (e.g. a non-default container/HPC path).
    resolved_backend_registry = backend_registry or build_default_backend_registry(
        rscript_path=config.r.rscript_path
    )

    # Choose the run identifier from config or project name.
    run_id = config.run.run_id or config.project.name

    # Optionally load the configured AnnData input.
    loaded_adata = load_input_adata_from_config(config) if load_input else None

    # Validate the declared cohort keys against the loaded obs columns once, at
    # startup. This is warn-not-raise: a cohort block may declare keys only some
    # stages use, but a key that is present in config yet absent from obs (a
    # typo, or the wrong input file) would otherwise surface only as an obscure
    # per-stage fallback or failure. Surface it loudly here and record it for the
    # run report so the mismatch is visible at the top, not hunted for later.
    cohort_warnings: list[str] = []
    cohort_config = getattr(config, "cohort", None)
    if cohort_config is not None and loaded_adata is not None:
        from cellquorum.config.cohort import validate_cohort_against_obs

        cohort_warnings = validate_cohort_against_obs(
            list(loaded_adata.obs.columns), cohort=cohort_config
        )
        for warning in cohort_warnings:
            logger.warning("Cohort/config mismatch: %s", warning)

    # Load the sample manifest whenever one is configured. This is independent of
    # load_input: ambient_correction needs the manifest to locate CellRanger
    # libraries even when there is no pre-built input AnnData.
    loaded_manifest = None
    if config.paths.manifest is not None:
        from cellquorum.io.manifest import load_manifest

        manifest_obj = load_manifest(config.paths.manifest, data_root=config.paths.data_root)
        loaded_manifest = manifest_obj.to_dataframe()

    # Build runtime metadata for provenance and reporting.
    metadata: dict[str, Any] = {
        "project_name": config.project.name,
        "profile": config.run.profile,
        "organism": config.project.organism,
        "species_id": config.project.species_id,
        "input_h5ad": str(config.input.h5ad) if config.input.h5ad is not None else None,
        "input_counts_layer": config.input.counts_layer,
        "input_loaded": loaded_adata is not None,
        # The size of the matrix this run actually analysed, recorded
        # UNCONDITIONALLY. "input_subset" below is None when no filter applied, so
        # without these two keys a run with no subset has its cell count nowhere in
        # the run directory's JSON -- only inside the final .h5ad, which a run that
        # computes a table rather than an object has no reason to write. Every
        # figure caption in every downstream repo needs this number, and reading it
        # from the run beats each script hardcoding its own copy.
        "input_n_obs": int(loaded_adata.n_obs) if loaded_adata is not None else None,
        "input_n_vars": int(loaded_adata.n_vars) if loaded_adata is not None else None,
        # Record the applied row restriction (column, values, n_before/n_after,
        # plus the exclusion rule and what it dropped) so a cell-type subset or an
        # artifact-cluster drop is a visible provenance step, not a silent cut.
        "input_subset": (
            loaded_adata.uns.get("cellquorum_input_subset") if loaded_adata is not None else None
        ),
        "manifest_path": str(config.paths.manifest) if config.paths.manifest is not None else None,
        "manifest_n_samples": int(len(loaded_manifest)) if loaded_manifest is not None else 0,
        # Cohort keys declared in config but absent from obs (empty when clean).
        "cohort_warnings": cohort_warnings,
        "bootstrap_time_utc": datetime.now(UTC).isoformat(),
    }

    # Build and return the pipeline context.
    return PipelineContext(
        config=config,
        paths=paths,
        adata=loaded_adata,
        manifest=loaded_manifest,
        backend_registry=resolved_backend_registry,
        run_id=run_id,
        random_seed=config.run.random_seed,
        metadata=metadata,
    )


def _environment_stamp() -> dict[str, Any]:
    """
    Capture a run-level environment stamp for reproducibility provenance.

    Records the CellQuorum version, the interpreter, the OS platform, and the
    versions of key scientific dependencies. Each dependency import is guarded so
    a missing (or broken) package records ``None`` instead of crashing the run.

    Returns:
        Dict with keys: cellquorum_version, python_version, platform, and a
        nested "dependencies" map of package name to version (or None).
    """
    # Version-carrying scientific dependencies worth stamping into provenance.
    dependency_packages = ("scanpy", "anndata", "numpy", "scipy", "pandas")

    # Resolve each dependency version from installed metadata, tolerating any that
    # are absent (records None) so a missing dep never crashes the run.
    dependencies: dict[str, str | None] = {}
    for package_name in dependency_packages:
        try:
            dependencies[package_name] = importlib.metadata.version(package_name)
        except Exception:
            dependencies[package_name] = None

    # Assemble the stamp in the same explicit dict style as run metadata.
    return {
        "cellquorum_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


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


def _stage_completion_sidecar_payload(record: StageExecutionRecord) -> dict[str, object]:
    """
    Build the durable per-stage completion sidecar payload.

    The sidecar is intentionally redundant with ``stage_execution_records.json``.
    Resume logic should be able to inspect one stage's completion state without
    parsing the whole run log.

    Args:
        record: Successful stage execution record.

    Returns:
        JSON-safe completion sidecar payload.
    """

    # Build artifact existence summaries at sidecar-write time.
    artifact_status = [
        {
            **artifact.to_dict(),
            "exists": artifact.path.exists(),
        }
        for artifact in record.output_artifacts
    ]

    # Return a compact but complete completion marker.
    return {
        "schema_version": 1,
        "stage_name": record.stage_name,
        "status": record.status,
        "completed_at_utc": record.ended_at_utc.isoformat(),
        "duration_seconds": record.duration_seconds,
        "backend_used": record.backend_used,
        "method_version": record.method_version,
        "device": record.device,
        "input_fingerprint": record.input_fingerprint,
        "output_fingerprint": record.output_fingerprint,
        "checkpoint_path": None if record.checkpoint_path is None else str(record.checkpoint_path),
        "n_output_artifacts": len(record.output_artifacts),
        "output_artifacts": artifact_status,
        "metrics": dict(record.metrics),
        "warnings": list(record.warnings),
        "notes": list(record.notes),
    }


def _stage_completion_sidecar_path(stage_name: str) -> Path:
    """
    Return a stable relative sidecar path for a stage.

    Args:
        stage_name: Stage name from the execution record.

    Returns:
        Relative path under the run root.
    """

    # Stage names are registry-controlled, but keep this safe for custom stages.
    safe_stage_name = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in stage_name
    ).strip("_")
    if not safe_stage_name:
        safe_stage_name = "stage"
    return Path("provenance") / "stages" / safe_stage_name / "completion.json"


def _write_stage_completion_sidecars(
    artifact_manager: ArtifactManager,
    records: list[StageExecutionRecord],
) -> None:
    """
    Write one completion marker per successful stage execution.

    Args:
        artifact_manager: Run-root artifact manager.
        records: Stage lifecycle records to inspect.
    """

    # Write only successful stage markers; skipped/failed stages are not reusable.
    for record in records:
        if record.status != "success":
            continue
        artifact_manager.write_json(
            _stage_completion_sidecar_payload(record),
            name=f"stage_completion_{record.stage_name}",
            relative_path=_stage_completion_sidecar_path(record.stage_name),
            description=f"Durable completion marker for stage '{record.stage_name}'.",
        )


def _write_final_object(*, config: CellQuorumConfig, context: PipelineContext) -> Path | None:
    """
    Write the final in-memory AnnData to the run's objects directory.

    A from-scratch run (e.g. built by ambient_correction) threads one AnnData
    through every stage in memory. Without this write, that fully-annotated
    object is discarded when the run ends and only per-stage snapshots and
    provenance remain on disk. Controlled by ``run.write_final_object`` /
    ``run.final_object_name``.

    Args:
        config: Resolved run configuration.
        context: Final pipeline context (its ``adata`` is the object to write).

    Returns:
        The path written, or None when writing is disabled or there is no adata.
    """

    run_config = getattr(config, "run", None)
    if run_config is None or not getattr(run_config, "write_final_object", True):
        return None

    adata = getattr(context, "adata", None)
    if adata is None:
        return None

    objects_dir = context.paths.objects
    objects_dir.mkdir(parents=True, exist_ok=True)
    out_path = objects_dir / getattr(run_config, "final_object_name", "final_annotated.h5ad")

    # Everything that makes a real object refuse to serialize — '/' in keys,
    # mixed-type label columns, uns payloads with no HDF5 encoding — is handled by
    # the shared writer. See cellquorum.core.h5ad_io for what it changes and why
    # it changes only representations, never information.
    from cellquorum.core.h5ad_io import write_h5ad

    write_h5ad(adata, out_path)
    return out_path


def _write_run_report_after_provenance(
    *,
    config: CellQuorumConfig,
    context: PipelineContext,
    records: list[StageExecutionRecord],
    artifact_manager: ArtifactManager,
) -> None:
    """
    Render the run report as a post-provenance hook.

    Running here (rather than as a mid-plan stage) lets the report see the full
    record set and avoids editing the hard-coded planner order. Report failures
    are swallowed unless ``config.report.fail_on_report_error`` is set.

    Args:
        config: Resolved run configuration.
        context: Final pipeline context (unused today; reserved for embedding
            figures/tables in a richer report).
        records: All stage execution records, in order.
        artifact_manager: The run-root artifact manager to write through.
    """

    report_config = getattr(config, "report", None)
    if report_config is None or not getattr(report_config, "enabled", False):
        return

    try:
        from cellquorum.core.reports import write_run_report

        write_run_report(config=config, records=records, artifact_manager=artifact_manager)
    except Exception:
        # Honor the opt-in: only fail the run when explicitly requested.
        if getattr(report_config, "fail_on_report_error", False):
            raise


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
            "environment": _environment_stamp(),
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

    # Write one completion sidecar per successful stage. These files are the
    # durable primitive that future resume logic can read stage-by-stage.
    _write_stage_completion_sidecars(artifact_manager, records)

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


def _restrict_plan_from_stage(plan: PipelinePlan, from_stage: str) -> PipelinePlan:
    """Return `plan` with stages before `from_stage` removed.

    Their results already live in the checkpoint being loaded, so re-running them
    would both waste time and risk producing a different object than the one the
    checkpoint recorded.
    """
    import dataclasses

    from cellquorum.core.stages import stage_order_map

    order_of = stage_order_map()
    start = order_of.get(from_stage)
    if start is None:
        return plan
    # PlannedStage records enablement, not order, so the cut point comes from the
    # catalog. Stages missing from the catalog are kept rather than dropped: losing
    # a stage silently is worse than running one extra.
    kept = [s for s in plan.stages if order_of.get(s.name, start) >= start]
    return dataclasses.replace(plan, stages=kept)


def _write_inherited_artifacts_report(
    *, context: PipelineContext, artifacts: list[InheritedArtifact]
) -> Path:
    """
    Write the inventory of files this run did not produce.

    The run summary names only the largest few, because a resume can inherit
    thousands of files legitimately and a warning that scrolls is a warning nobody
    reads. The full list belongs on disk next to the rest of the provenance, so a
    reader assembling a figure panel can check whether the file they picked was
    actually produced by the run they think it was.

    Args:
        context: The pipeline context, for its provenance path.
        artifacts: The inventory, largest first.

    Returns:
        The path written.
    """

    provenance_dir = Path(context.paths.provenance)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "inherited_artifacts.csv"
    pd.DataFrame(
        [
            {
                "path": artifact.path,
                "size_bytes": artifact.size_bytes,
                "modified_utc": artifact.modified_utc,
            }
            for artifact in artifacts
        ],
        columns=["path", "size_bytes", "modified_utc"],
    ).to_csv(path, index=False)
    return path


def _config_disabled_stage_names(
    config: CellQuorumConfig, config_dict: dict[str, Any]
) -> list[str]:
    """
    Return the stages this configuration turns off.

    A stage is off if either gate says so: the top-level ``stages`` flag, or the
    stage's own ``enabled`` field. Both are honoured because they are independent --
    ``stages.qc: false`` and ``qc: {enabled: false}`` are different ways to say the
    same thing, and a check that read only one would miss half the cases.

    This deliberately ignores the plan. Plan membership also encodes resume windows
    and missing implementations, neither of which is a statement that a stage should
    not exist; conflating them would make the caller's hygiene gate fire on the
    inherited outputs a ``--from-stage`` resume is built to reuse.

    Args:
        config: The resolved configuration.
        config_dict: The same configuration as a plain dict.

    Returns:
        Sorted stage names, so the caller's message is stable across runs.
    """

    disabled: set[str] = set()

    # Read the top-level stage flags.
    for name, flag in config.stages.model_dump().items():
        if flag is False:
            disabled.add(name)

    # Read each stage's own enabled field.
    for name, stage_config in config_dict.items():
        if isinstance(stage_config, dict) and stage_config.get("enabled") is False:
            disabled.add(name)

    return sorted(disabled)


def execute_pipeline_run(
    config: CellQuorumConfig,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
    executor: PipelineExecutor | None = None,
    load_input: bool = True,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> PipelineRunResult:
    """
    Execute registered CellQuorum stages from a validated configuration.

    This is the first true execution entry point. It builds the pipeline context,
    optionally loads the configured AnnData input, builds the stage plan, executes
    registered stages through PipelineExecutor, and writes provenance containing
    both bootstrap and stage execution records.

    Args:
        config: Validated CellQuorum configuration.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for tests or custom execution.
        executor: Optional PipelineExecutor override.
        load_input: Whether to load config.input.h5ad into context.adata.

    Returns:
        PipelineRunResult containing config, plan, final context, provenance
        artifacts, and PipelineExecutionResult.

    Raises:
        TypeError: If config is not a CellQuorumConfig.
    """

    # Validate the config type early.
    if not isinstance(config, CellQuorumConfig):
        raise TypeError(
            "execute_pipeline_run expected a CellQuorumConfig object. "
            f"Received: {type(config).__name__}"
        )

    # anndata >= 0.11 refuses to write pandas nullable / Arrow-backed string
    # columns (common in externally annotated inputs) unless the caller opts
    # in. Enable it once for the whole run so every stage's h5ad write
    # succeeds. Older anndata lacks the setting and writes these dtypes
    # unconditionally, so the guard is a no-op there.
    if hasattr(ad.settings, "allow_write_nullable_strings"):
        ad.settings.allow_write_nullable_strings = True

    # Mark the start of execution-frame setup.
    bootstrap_started_at = datetime.now(UTC)

    # Build the initialized pipeline context, loading AnnData when requested.
    context = build_pipeline_context(
        config,
        output_dir=output_dir,
        backend_registry=backend_registry,
        load_input=load_input,
    )

    # Build the pipeline plan using the context backend registry.
    plan = build_pipeline_plan(
        config,
        backend_registry=context.backend_registry,
    )

    # Mark the end of execution-frame setup before stage execution.
    bootstrap_ended_at = datetime.now(UTC)

    # Resolve the executor. Honor the run-level continue-on-failure switch so an
    # unattended canary attempts every stage instead of halting on the first
    # failure (a caller-supplied executor keeps its own policy).
    resolved_executor = executor or PipelineExecutor(
        stop_on_failure=not config.run.continue_on_stage_failure,
        until_stage=until_stage,
    )

    # Resume from a checkpoint when asked to start mid-pipeline. Done here, after
    # the plan exists, because the plan is what gets restricted — and it must fail
    # loudly rather than silently starting from raw input, which would look like a
    # resume while producing different numbers.
    if from_stage:
        from cellquorum.core.checkpoint import (
            load_checkpoint,
            resolve_start_checkpoint,
        )
        from cellquorum.core.fingerprint import compute_upstream_fingerprint
        from cellquorum.core.stages import stage_order_map

        stage_order = stage_order_map()
        start_record = resolve_start_checkpoint(
            context.paths, from_stage=from_stage, stage_order=stage_order
        )
        # Refuse a checkpoint written under settings this run is not using. Without
        # this the run would resume happily and report success while its numbers came
        # from a config nobody chose, and resolved_config.json would disagree with the
        # object on disk.
        context = context.with_adata(
            load_checkpoint(
                start_record,
                expected_upstream_fingerprint=compute_upstream_fingerprint(
                    config=config.model_dump(),
                    stage_order=stage_order,
                    through_stage=start_record.stage,
                ),
            )
        )
        plan = _restrict_plan_from_stage(plan, from_stage)

    # Build the run reporter from config verbosity settings.
    reporter = RunReporter(verbose=config.run.verbose, level=config.run.log_level)

    # Print startup banner with version and project metadata.
    run_id = context.paths.root.name
    reporter.banner(__version__, config.project.name, run_id)

    # Compute which stages will actually run (planned + registered + per-stage
    # enabled). A stage runs only if: in plan, registered, AND its sub-config
    # .enabled is True (when present).
    planned_stage_names = []
    config_dict = config.model_dump()
    for stage in plan.stages:
        # Check plan gate and registration.
        if not stage.enabled or resolved_executor.registry.get(stage.name) is None:
            continue
        # Check per-stage sub-config .enabled field.
        stage_config = config_dict.get(stage.name, {})
        if isinstance(stage_config, dict):
            per_stage_enabled = stage_config.get("enabled", True)
        else:
            per_stage_enabled = True
        if per_stage_enabled:
            planned_stage_names.append(stage.name)

    # Echo the resolved configuration showing only runnable stages.
    reporter.config_echo(config, planned_stage_names=planned_stage_names)

    # Refuse to write into a directory that already holds outputs of a stage this
    # config DISABLES. Those files cannot have come from this run, and nothing on
    # disk says so, which is how a manuscript ends up citing QC thresholds that were
    # never applied. Read the disabled set from the CONFIG rather than from the plan:
    # a --from-stage resume legitimately narrows the plan, and treating those earlier
    # stages as disabled would break resume on its own inherited outputs.
    assert_output_dir_matches_config(
        context.paths.root,
        disabled_stages=_config_disabled_stage_names(config, config_dict),
    )

    # Measure wall-clock time around stage execution.
    execution_start = time.perf_counter()

    # Execute registered stages from the plan with progress reporting.
    execution_result = resolved_executor.run(
        context=context,
        plan=plan,
        reporter=reporter,
    )

    # Measure elapsed wall-clock time.
    execution_elapsed = time.perf_counter() - execution_start

    # Print the final run summary.
    reporter.run_summary(
        execution_result.stage_execution_records,
        str(context.paths.root),
        execution_elapsed,
    )

    # Inventory files in the output tree that this run did not write. The gate above
    # already refused outputs of a DISABLED stage; this catches the narrower case it
    # cannot see -- a stage that ran fine but no longer produces some group, cluster
    # or pathway, leaving the previous run's artifact for it behind looking current.
    # It reports and never fails: a --from-stage resume inherits earlier outputs
    # legitimately, and timestamps alone cannot tell that apart from a leftover.
    inherited = find_inherited_artifacts(context.paths.root, run_started_at=bootstrap_started_at)
    inherited_warning = format_inherited_artifacts(inherited)
    if inherited_warning:
        logger.warning(inherited_warning)
        _write_inherited_artifacts_report(context=context, artifacts=inherited)

    # Build the bootstrap execution record.
    bootstrap_record = StageExecutionRecord(
        stage_name="bootstrap",
        status="success",
        started_at_utc=bootstrap_started_at,
        ended_at_utc=bootstrap_ended_at,
        duration_seconds=(bootstrap_ended_at - bootstrap_started_at).total_seconds(),
        backend_used="python",
        notes=["Initialized CellQuorum execution frame."],
        warnings=[*plan.warnings, *([inherited_warning] if inherited_warning else [])],
        metrics={
            "n_planned_stages": len(plan.stages),
            "n_enabled_stages": len(plan.enabled_stage_names()),
            "n_backend_status_rows": len(plan.backend_status_table),
            "input_loaded": execution_result.context.adata is not None,
            "n_successful_stages": len(execution_result.succeeded_stage_names()),
            "n_skipped_stages": len(execution_result.skipped_stage_names()),
            "n_failed_stages": len(execution_result.failed_stage_names()),
            "n_inherited_artifacts": len(inherited),
            "inherited_artifact_bytes": sum(item.size_bytes for item in inherited),
        },
    )

    # Write provenance with bootstrap and real stage execution records.
    all_records = [
        bootstrap_record,
        *execution_result.stage_execution_records,
    ]
    artifacts = write_pipeline_provenance(
        config=config,
        plan=plan,
        context=execution_result.context,
        stage_execution_records=all_records,
    )

    # Persist the final in-memory AnnData so a from-scratch run leaves a real
    # annotated deliverable on disk (not just per-stage snapshots/provenance).
    _write_final_object(config=config, context=execution_result.context)

    # Render the human-readable run report AFTER provenance is written, so it
    # sees the complete record set. Report failures never fail the run unless
    # the user opts in via report.fail_on_report_error.
    _write_run_report_after_provenance(
        config=config,
        context=execution_result.context,
        records=all_records,
        artifact_manager=artifacts,
    )

    # Return the executed pipeline run result.
    return PipelineRunResult(
        config=config,
        plan=plan,
        context=execution_result.context,
        artifacts=artifacts,
        execution_result=execution_result,
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


def execute_pipeline_run_from_config_file(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    backend_registry: BackendRegistry | None = None,
    executor: PipelineExecutor | None = None,
    load_input: bool = True,
    from_stage: str | None = None,
    until_stage: str | None = None,
) -> PipelineRunResult:
    """
    Load a config file and execute registered CellQuorum stages.

    Args:
        config_path: Path to a CellQuorum YAML configuration file.
        output_dir: Optional explicit output directory override.
        backend_registry: Optional backend registry for tests or custom execution.
        executor: Optional PipelineExecutor override.
        load_input: Whether to load config.input.h5ad into context.adata.
        from_stage: Optional stage to start at, resuming from a checkpoint.
        until_stage: Optional stage to stop after.

    Returns:
        PipelineRunResult containing config, plan, final context, provenance
        artifacts, and PipelineExecutionResult.
    """

    # Load and validate the configuration file.
    config = load_config(config_path)

    # Execute the pipeline run from the validated config. from_stage/until_stage
    # must be forwarded: silently dropping them would run every stage while the
    # caller believed the run had been restricted to one.
    return execute_pipeline_run(
        config,
        output_dir=output_dir,
        backend_registry=backend_registry,
        executor=executor,
        load_input=load_input,
        from_stage=from_stage,
        until_stage=until_stage,
    )
