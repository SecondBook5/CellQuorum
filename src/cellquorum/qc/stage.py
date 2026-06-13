"""Pipeline stage wrapper for the CellQuorum QC module."""

from __future__ import annotations

# Import Mapping for dictionary-based QC config extraction.
from collections.abc import Mapping

# Import dataclass helpers for structured workflow outputs.
from dataclasses import dataclass, field

# Import Path for output-directory handling.
from pathlib import Path

# Import AnnData for runtime object checks and filtering.
import anndata as ad

# Import pandas for decision-table alignment checks.
import pandas as pd

# Import top-level runtime context.
from cellquorum.core.context import PipelineContext

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import pipeline stage result and artifact records.
from cellquorum.core.stage import StageArtifact, StageResult

# Import QC artifact writing utilities.
from cellquorum.qc.artifacts import QCArtifactManifest, write_qc_artifacts

# Import QC configuration.
from cellquorum.qc.config import QCConfig, validate_qc_config_dict

# Import QC decision construction.
from cellquorum.qc.decisions import QCDecisionResult, build_qc_decisions

# Import QC metric calculation.
from cellquorum.qc.metrics import QCMetricsResult, calculate_qc_metrics

# Import QC threshold construction.
from cellquorum.qc.thresholds import QCThresholdResult, build_qc_thresholds

# Import QC input validation.
from cellquorum.qc.validation import QCInputValidationSummary, validate_qc_input_adata


class QCStageError(CellQuorumDataError):
    """
    Report QC stage orchestration failures.

    The stage layer is responsible for connecting validation, metric calculation,
    thresholding, decision construction, filtering, and artifact writing. Errors
    here usually mean that individually valid pieces could not be composed safely.
    """


@dataclass(frozen=True)
class QCWorkflowResult:
    """
    Store all intermediate and final QC workflow outputs.

    Args:
        adata: AnnData object after optional QC filtering.
        validation_summary: Input validation summary.
        metrics_result: Calculated QC metrics.
        threshold_result: Constructed QC thresholds.
        decision_result: Applied QC decisions.
        artifact_manifest: Optional written artifact manifest.
        notes: Non-critical workflow notes.
        warnings: Important workflow warnings.
        metrics: JSON-friendly stage metrics.
    """

    # Store the AnnData object after optional QC filtering.
    adata: ad.AnnData

    # Store the input validation summary.
    validation_summary: QCInputValidationSummary

    # Store calculated QC metrics.
    metrics_result: QCMetricsResult

    # Store constructed QC thresholds.
    threshold_result: QCThresholdResult

    # Store applied QC decisions.
    decision_result: QCDecisionResult

    # Store optional artifact manifest.
    artifact_manifest: QCArtifactManifest | None = None

    # Store non-critical notes.
    notes: list[str] = field(default_factory=list)

    # Store important warnings.
    warnings: list[str] = field(default_factory=list)

    # Store JSON-friendly stage metrics.
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class QCStage:
    """
    Execute CellQuorum quality control as a pipeline stage.

    Args:
        config: Optional explicit QC configuration. When omitted, the stage tries
            to resolve a QC config from PipelineContext.config and falls back to
            QCConfig().
        output_subdir: Subdirectory under context.paths.results used for QC
            artifacts.
        write_artifacts: Whether to write QC artifacts during stage execution.
    """

    # Store the stable pipeline stage name.
    name: str = "qc"

    # Store an optional explicit QC config.
    config: QCConfig | None = None

    # Store the artifact output subdirectory name.
    output_subdir: str = "qc"

    # Store whether this stage should write artifacts.
    write_artifacts: bool = True

    def run(self, context: object) -> StageResult:
        """
        Execute the QC stage.

        Args:
            context: PipelineContext containing AnnData, config, and paths.

        Returns:
            StageResult containing the post-QC AnnData object, artifacts, notes,
            warnings, and summary metrics.

        Raises:
            QCStageError: If the context or QC workflow is invalid.
        """

        # Validate the pipeline context.
        if not isinstance(context, PipelineContext):
            raise QCStageError(
                "QCStage.run expected a PipelineContext. " f"Received: {type(context).__name__}."
            )

        # Resolve the QC configuration.
        qc_config = resolve_qc_config(context=context, explicit_config=self.config)

        # Require an AnnData object from the context.
        input_adata = context.require_adata()

        # Return a no-op result when QC is disabled.
        if not qc_config.enabled:
            return StageResult(
                adata=input_adata,
                artifacts=[],
                notes=["QC stage was skipped because qc.enabled is false."],
                warnings=[],
                metrics={
                    "enabled": False,
                    "stage": self.name,
                    "input_shape": summarize_adata_shape_for_stage(input_adata),
                    "output_shape": summarize_adata_shape_for_stage(input_adata),
                },
            )

        # Resolve the QC artifact output directory.
        output_dir = context.paths.results / self.output_subdir

        # Execute the full QC workflow.
        workflow_result = run_qc_workflow(
            adata=input_adata,
            config=qc_config,
            output_dir=output_dir,
            write_artifacts=self.write_artifacts,
            summary_extra={
                "stage": self.name,
                "run_id": context.run_id,
                "random_seed": context.random_seed,
            },
        )

        # Convert QC artifact manifest records into generic stage artifacts.
        stage_artifacts = (
            []
            if workflow_result.artifact_manifest is None
            else stage_artifacts_from_qc_manifest(workflow_result.artifact_manifest)
        )

        # Return the generic stage result.
        return StageResult(
            adata=workflow_result.adata,
            artifacts=stage_artifacts,
            notes=workflow_result.notes,
            warnings=workflow_result.warnings,
            metrics=workflow_result.metrics,
        )


def run_qc_workflow(
    *,
    adata: ad.AnnData,
    config: QCConfig | None = None,
    output_dir: str | Path | None = None,
    write_artifacts: bool = True,
    summary_extra: dict[str, object] | None = None,
) -> QCWorkflowResult:
    """
    Run the full QC workflow outside the generic pipeline executor.

    This helper is useful for tests, notebooks, and future CLI integration. It
    performs the same orchestration used by QCStage.run.

    Args:
        adata: Input AnnData object.
        config: Optional QC configuration. Defaults to QCConfig().
        output_dir: Optional artifact output directory.
        write_artifacts: Whether to write QC artifacts.
        summary_extra: Optional extra JSON-friendly values for qc_summary.json.

    Returns:
        QCWorkflowResult containing all intermediate outputs and stage metrics.

    Raises:
        QCStageError: If filtering or artifact settings are inconsistent.
    """

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate QC configuration type.
    if not isinstance(qc_config, QCConfig):
        raise QCStageError(
            "run_qc_workflow expected config to be a QCConfig object. "
            f"Received: {type(qc_config).__name__}."
        )

    # Validate AnnData input type.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "run_qc_workflow expected an AnnData object. " f"Received: {type(adata).__name__}."
        )

    # Prepare AnnData for QC, including configured duplicate-name handling.
    qc_adata, preparation_notes = prepare_adata_for_qc(adata, qc_config)

    # Validate the QC input.
    validation_summary = validate_qc_input_adata(qc_adata, qc_config)

    # Calculate QC metrics.
    metrics_result = calculate_qc_metrics(qc_adata, qc_config)

    # Build configured QC thresholds.
    threshold_result = build_qc_thresholds(
        cell_metrics=metrics_result.cell_metrics,
        gene_metrics=metrics_result.gene_metrics,
        config=qc_config,
    )

    # Build explicit QC decisions.
    decision_result = build_qc_decisions(
        cell_metrics=metrics_result.cell_metrics,
        gene_metrics=metrics_result.gene_metrics,
        thresholds=threshold_result,
        config=qc_config,
    )

    # Apply filtering when requested.
    output_adata = apply_qc_filter_to_adata(
        adata=qc_adata,
        decision_result=decision_result,
        config=qc_config,
    )

    # Initialize artifact manifest.
    artifact_manifest: QCArtifactManifest | None = None

    # Write artifacts when requested.
    if write_artifacts:
        # Validate that an output directory was supplied.
        if output_dir is None:
            raise QCStageError("QC artifact writing requires an output_dir.")

        # Write QC artifacts.
        artifact_manifest = write_qc_artifacts(
            output_dir=output_dir,
            metrics_result=metrics_result,
            threshold_result=threshold_result,
            decision_result=decision_result,
            config=qc_config,
            adata=output_adata,
            summary_extra=summary_extra,
        )

    # Build stage notes.
    notes = build_qc_stage_notes(
        config=qc_config,
        input_adata=qc_adata,
        output_adata=output_adata,
        preparation_notes=preparation_notes,
    )

    # Build stage warnings.
    warnings = collect_qc_stage_warnings(
        validation_summary=validation_summary,
        metrics_result=metrics_result,
        threshold_result=threshold_result,
        decision_result=decision_result,
        artifact_manifest=artifact_manifest,
    )

    # Build JSON-friendly stage metrics.
    stage_metrics = build_qc_stage_metrics(
        config=qc_config,
        input_adata=qc_adata,
        output_adata=output_adata,
        validation_summary=validation_summary,
        metrics_result=metrics_result,
        threshold_result=threshold_result,
        decision_result=decision_result,
        artifact_manifest=artifact_manifest,
    )

    # Return the structured workflow result.
    return QCWorkflowResult(
        adata=output_adata,
        validation_summary=validation_summary,
        metrics_result=metrics_result,
        threshold_result=threshold_result,
        decision_result=decision_result,
        artifact_manifest=artifact_manifest,
        notes=notes,
        warnings=warnings,
        metrics=stage_metrics,
    )


def resolve_qc_config(
    *,
    context: PipelineContext,
    explicit_config: QCConfig | None = None,
) -> QCConfig:
    """
    Resolve the QC configuration for stage execution.

    Args:
        context: Pipeline context.
        explicit_config: Optional explicit QC configuration supplied to QCStage.

    Returns:
        Resolved QCConfig.

    Raises:
        QCStageError: If a discovered QC configuration has an unsupported type.
    """

    # Prefer an explicit stage-level QC configuration.
    if explicit_config is not None:
        # Validate explicit config type.
        if not isinstance(explicit_config, QCConfig):
            raise QCStageError(
                "explicit_config must be a QCConfig object. "
                f"Received: {type(explicit_config).__name__}."
            )

        # Return the explicit config.
        return explicit_config

    # Use context.config directly when it is already a QCConfig.
    if isinstance(context.config, QCConfig):
        return context.config

    # Use context.config.qc when present.
    if hasattr(context.config, "qc"):
        # Read the qc attribute.
        qc_config = context.config.qc

        # Return when already validated.
        if isinstance(qc_config, QCConfig):
            return qc_config

        # Validate mappings into QCConfig.
        if isinstance(qc_config, Mapping):
            return validate_qc_config_dict(qc_config)

        # Reject unsupported qc attributes.
        raise QCStageError(
            "context.config.qc must be a QCConfig or mapping. "
            f"Received: {type(qc_config).__name__}."
        )

    # Use mapping-based config["qc"] when present.
    if isinstance(context.config, Mapping) and "qc" in context.config:
        # Extract the QC config mapping or object.
        qc_config = context.config["qc"]

        # Return when already validated.
        if isinstance(qc_config, QCConfig):
            return qc_config

        # Validate mappings into QCConfig.
        if isinstance(qc_config, Mapping):
            return validate_qc_config_dict(qc_config)

        # Reject unsupported mapping values.
        raise QCStageError(
            "context.config['qc'] must be a QCConfig or mapping. "
            f"Received: {type(qc_config).__name__}."
        )

    # Fall back to default QC configuration while top-level integration is still bootstrapping.
    return QCConfig()


def prepare_adata_for_qc(adata: ad.AnnData, config: QCConfig) -> tuple[ad.AnnData, list[str]]:
    """
    Prepare AnnData before QC metric calculation.

    This currently implements configured duplicate-name repair for observation
    and variable names. A copy is made only when a name repair is needed.

    Args:
        adata: Input AnnData object.
        config: QC configuration.

    Returns:
        Tuple containing prepared AnnData and preparation notes.
    """

    # Initialize preparation notes.
    notes: list[str] = []

    # Track whether a copy is required.
    needs_copy = (
        not adata.obs_names.is_unique and config.duplicate_names.obs_names == "make_unique"
    ) or (not adata.var_names.is_unique and config.duplicate_names.var_names == "make_unique")

    # Copy only when duplicate-name repair is needed.
    prepared = adata.copy() if needs_copy else adata

    # Make duplicate observation names unique when configured.
    if not prepared.obs_names.is_unique and config.duplicate_names.obs_names == "make_unique":
        # Repair duplicate observation names.
        prepared.obs_names_make_unique()

        # Store a note.
        notes.append("Made duplicate AnnData.obs_names unique before QC metric calculation.")

    # Make duplicate variable names unique when configured.
    if not prepared.var_names.is_unique and config.duplicate_names.var_names == "make_unique":
        # Repair duplicate variable names.
        prepared.var_names_make_unique()

        # Store a note.
        notes.append("Made duplicate AnnData.var_names unique before QC metric calculation.")

    # Return prepared AnnData and notes.
    return prepared, notes


def apply_qc_filter_to_adata(
    *,
    adata: ad.AnnData,
    decision_result: QCDecisionResult,
    config: QCConfig,
) -> ad.AnnData:
    """
    Apply QC filtering decisions to AnnData when configured.

    Args:
        adata: AnnData object used for QC.
        decision_result: QC decision result.
        config: QC configuration.

    Returns:
        Original or filtered AnnData object.

    Raises:
        QCStageError: If decision tables do not align with AnnData.
    """

    # Return the original object when filtering is disabled.
    if not config.should_filter():
        return adata

    # Validate cell decision alignment.
    validate_decision_index_alignment(
        expected_index=adata.obs_names,
        observed_index=decision_result.cell_decisions.index,
        axis_name="obs",
    )

    # Validate gene decision alignment.
    validate_decision_index_alignment(
        expected_index=adata.var_names,
        observed_index=decision_result.gene_decisions.index,
        axis_name="var",
    )

    # Extract cell keep mask by position.
    cell_keep = decision_result.cell_decisions["keep"].to_numpy(dtype=bool)

    # Extract gene keep mask by position.
    gene_keep = decision_result.gene_decisions["keep"].to_numpy(dtype=bool)

    # Return a filtered AnnData copy.
    return adata[cell_keep, gene_keep].copy()


def validate_decision_index_alignment(
    *,
    expected_index: pd.Index,
    observed_index: pd.Index,
    axis_name: str,
) -> None:
    """
    Validate that a decision table index matches an AnnData axis by position.

    Args:
        expected_index: AnnData axis index.
        observed_index: Decision table index.
        axis_name: Human-readable axis label.

    Raises:
        QCStageError: If the index lengths or values do not match.
    """

    # Compare index lengths first.
    if len(expected_index) != len(observed_index):
        raise QCStageError(
            f"QC {axis_name} decision index has length {len(observed_index)}, "
            f"but AnnData.{axis_name} has length {len(expected_index)}."
        )

    # Compare stringified index values by position.
    if [str(value) for value in observed_index] != [str(value) for value in expected_index]:
        raise QCStageError(
            f"QC {axis_name} decision index does not match AnnData.{axis_name}_names."
        )


def build_qc_stage_notes(
    *,
    config: QCConfig,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
    preparation_notes: list[str],
) -> list[str]:
    """
    Build human-readable QC stage notes.

    Args:
        config: QC configuration.
        input_adata: AnnData object used for QC.
        output_adata: AnnData object after optional filtering.
        preparation_notes: Notes emitted during input preparation.

    Returns:
        Ordered stage notes.
    """

    # Start with preparation notes.
    notes = list(preparation_notes)

    # Add completion note.
    notes.append(f"QC completed in {config.mode} mode.")

    # Add filtering note when filtering ran.
    if config.should_filter():
        notes.append(
            "QC filtering retained "
            f"{output_adata.n_obs}/{input_adata.n_obs} cells and "
            f"{output_adata.n_vars}/{input_adata.n_vars} genes."
        )

    # Add report-only note when filtering did not run.
    else:
        notes.append("QC decisions were reported but AnnData was not filtered.")

    # Return notes.
    return notes


def collect_qc_stage_warnings(
    *,
    validation_summary: QCInputValidationSummary,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    artifact_manifest: QCArtifactManifest | None,
) -> list[str]:
    """
    Collect warnings emitted across the QC workflow.

    Args:
        validation_summary: Input validation summary.
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
        artifact_manifest: Optional artifact manifest.

    Returns:
        De-duplicated warning list in first-seen order.
    """

    # Initialize warning messages.
    warnings: list[str] = []

    # Extend warnings from validation.
    warnings.extend(validation_summary.warnings)

    # Extend warnings from metrics.
    warnings.extend(metrics_result.warnings)

    # Extend warnings from thresholds.
    warnings.extend(threshold_result.warnings)

    # Extend warnings from decisions.
    warnings.extend(decision_result.warnings)

    # Extend warnings from artifacts when present.
    if artifact_manifest is not None:
        warnings.extend(artifact_manifest.warnings)

    # Return de-duplicated warnings.
    return deduplicate_strings(warnings)


def build_qc_stage_metrics(
    *,
    config: QCConfig,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
    validation_summary: QCInputValidationSummary,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    artifact_manifest: QCArtifactManifest | None,
) -> dict[str, object]:
    """
    Build JSON-friendly metrics for the generic StageResult.

    Args:
        config: QC configuration.
        input_adata: AnnData object used for QC.
        output_adata: AnnData object after optional filtering.
        validation_summary: Input validation summary.
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
        artifact_manifest: Optional artifact manifest.

    Returns:
        JSON-friendly stage metrics dictionary.
    """

    # Return the structured stage metrics.
    return {
        "enabled": config.enabled,
        "mode": config.mode,
        "threshold_strategy": config.threshold_strategy,
        "should_filter": config.should_filter(),
        "input_shape": summarize_adata_shape_for_stage(input_adata),
        "output_shape": summarize_adata_shape_for_stage(output_adata),
        "validation": validation_summary.to_dict(),
        "metrics": metrics_result.to_summary_dict(),
        "thresholds": threshold_result.to_summary_dict(),
        "decisions": decision_result.to_summary_dict(),
        "artifacts": None if artifact_manifest is None else artifact_manifest.to_dict(),
    }


def stage_artifacts_from_qc_manifest(manifest: QCArtifactManifest) -> list[StageArtifact]:
    """
    Convert QC artifact manifest records into generic StageArtifact records.

    Args:
        manifest: QC artifact manifest.

    Returns:
        StageArtifact records for written QC artifacts.
    """

    # Convert each written QC artifact into a generic stage artifact.
    return [
        StageArtifact(
            name=artifact_name,
            path=artifact_path,
            kind=infer_artifact_kind(artifact_path),
            description=describe_qc_artifact(artifact_name),
        )
        for artifact_name, artifact_path in manifest.artifacts.items()
    ]


def infer_artifact_kind(path: Path) -> str:
    """
    Infer a generic artifact kind from a file extension.

    Args:
        path: Artifact path.

    Returns:
        Artifact kind string.
    """

    # Resolve lower-case suffix.
    suffix = path.suffix.lower()

    # Map CSV files to csv kind.
    if suffix == ".csv":
        return "csv"

    # Map JSON files to json kind.
    if suffix == ".json":
        return "json"

    # Map h5ad files to h5ad kind.
    if suffix == ".h5ad":
        return "h5ad"

    # Fall back to file kind.
    return "file"


def describe_qc_artifact(artifact_name: str) -> str:
    """
    Return a human-readable description for a QC artifact.

    Args:
        artifact_name: Stable QC artifact label.

    Returns:
        Artifact description.
    """

    # Store known artifact descriptions.
    descriptions = {
        "cell_metrics": "Cell-level QC metric table.",
        "gene_metrics": "Gene-level QC metric table.",
        "feature_masks": "Feature-family masks used for QC metrics.",
        "thresholds": "QC threshold table.",
        "cell_decisions": "Cell-level QC keep/fail decision table.",
        "gene_decisions": "Gene-level QC keep/fail decision table.",
        "qc_h5ad": "AnnData object after optional QC filtering.",
        "summary": "QC summary JSON.",
    }

    # Return known description or a generic fallback.
    return descriptions.get(artifact_name, f"QC artifact: {artifact_name}.")


def summarize_adata_shape_for_stage(adata: ad.AnnData) -> dict[str, int]:
    """
    Summarize AnnData shape for stage metrics.

    Args:
        adata: AnnData object.

    Returns:
        Shape summary dictionary.
    """

    # Return observation and variable counts.
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }


def deduplicate_strings(values: list[str]) -> list[str]:
    """
    De-duplicate strings while preserving first-seen order.

    Args:
        values: Input string values.

    Returns:
        De-duplicated string values.
    """

    # Initialize seen values.
    seen: set[str] = set()

    # Initialize output values.
    deduplicated: list[str] = []

    # Iterate over values in input order.
    for value in values:
        # Skip duplicate values.
        if value in seen:
            continue

        # Mark value as seen.
        seen.add(value)

        # Store value.
        deduplicated.append(value)

    # Return de-duplicated values.
    return deduplicated


__all__ = [
    "QCStage",
    "QCStageError",
    "QCWorkflowResult",
    "apply_qc_filter_to_adata",
    "build_qc_stage_metrics",
    "build_qc_stage_notes",
    "collect_qc_stage_warnings",
    "deduplicate_strings",
    "describe_qc_artifact",
    "infer_artifact_kind",
    "prepare_adata_for_qc",
    "resolve_qc_config",
    "run_qc_workflow",
    "stage_artifacts_from_qc_manifest",
    "summarize_adata_shape_for_stage",
    "validate_decision_index_alignment",
]
