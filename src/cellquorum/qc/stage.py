"""QC pipeline stage for CellQuorum."""

from __future__ import annotations

# Import logging for loud, auditable QC decisions (no-silent-decisions rule).
import logging

# Import Mapping for dictionary-like config resolution.
from collections.abc import Mapping

# Import dataclass for the concrete stage object.
from dataclasses import dataclass

# Import Path for stage output directory handling.
from pathlib import Path

# Import AnnData for stage input and output typing.
import anndata as ad

# Import pandas for AnnData obs/var decision annotation typing.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import pipeline stage artifact and result contracts.
from cellquorum.core.stage import StageArtifact, StageResult

# Import QC artifact writer utilities.
from cellquorum.qc.artifacts import QCArtifactManifest, write_qc_artifacts

# Import QC configuration.
from cellquorum.qc.config import QCConfig, validate_qc_config_dict

# Import QC decision construction.
from cellquorum.qc.decisions import QCDecisionResult, build_qc_decisions

# Import QC metric calculation.
from cellquorum.qc.metrics import QCMetricsResult, calculate_qc_metrics

# Import QC threshold construction.
from cellquorum.qc.thresholds import QCThresholdResult, build_qc_thresholds

logger = logging.getLogger(__name__)


class QCStageError(CellQuorumDataError):
    """
    Report QC stage execution failures.

    The QC stage is the first full analysis stage. Errors here should explain
    whether the failure came from missing context state, invalid QC configuration,
    metric calculation, thresholding, decision construction, filtering, or
    artifact writing.
    """


@dataclass(frozen=True)
class QCStage:
    """
    Execute the complete CellQuorum QC module.

    The stage wires together the QC submodule layers:

    1. validate and calculate QC metrics
    2. build fixed and/or MAD thresholds
    3. apply thresholds into explicit decision tables
    4. optionally filter AnnData
    5. write machine-readable artifacts
    6. return a StageResult for provenance and downstream stages

    Args:
        config: Optional QCConfig override. If omitted, the stage resolves QC
            configuration from context.config.qc when available, otherwise it
            uses QCConfig().
        output_subdir: Subdirectory under context.paths.results where QC
            artifacts should be written.
    """

    # Store the stable stage name expected by the pipeline contract.
    name: str = "qc"

    # Store an optional explicit QC configuration override.
    config: QCConfig | None = None

    # Store the results subdirectory used for QC artifacts.
    output_subdir: str = "qc"

    def run(self, context: object) -> StageResult:
        """
        Execute the QC stage.

        Args:
            context: PipelineContext-like object containing config, paths, and
                AnnData.

        Returns:
            StageResult containing the QC-updated AnnData object, written
            artifacts, notes, warnings, and structured QC metrics.

        Raises:
            QCStageError: If required context state is missing or QC execution
                fails.
        """

        # Retrieve the active AnnData object.
        adata = get_context_adata(context)

        # Resolve the effective QC configuration.
        qc_config = resolve_qc_config(context, override=self.config)

        # Return an explicit no-op result when QC is disabled.
        if not is_qc_stage_enabled(context, qc_config):
            return build_disabled_qc_stage_result(
                adata=adata,
                stage_name=self.name,
                qc_config=qc_config,
            )

        # Resolve the QC artifact output directory.
        output_dir = get_qc_output_dir(context, self.output_subdir)

        # Calculate cell-level, gene-level, and feature-family QC metrics.
        metrics_result = calculate_qc_metrics(adata, qc_config)

        # Build fixed and adaptive threshold records.
        threshold_result = build_qc_thresholds(
            cell_metrics=metrics_result.cell_metrics,
            gene_metrics=metrics_result.gene_metrics,
            config=qc_config,
        )

        # Apply threshold records to produce explicit decision tables.
        decision_result = build_qc_decisions(
            cell_metrics=metrics_result.cell_metrics,
            gene_metrics=metrics_result.gene_metrics,
            thresholds=threshold_result,
            config=qc_config,
        )

        # Build the output AnnData object, optionally filtered.
        output_adata = build_qc_output_adata(
            adata=adata,
            decision_result=decision_result,
            config=qc_config,
        )

        # Carry calculated QC metrics onto the QC AnnData before figure/h5ad
        # artifact writing. The durable metric tables remain canonical, but
        # visualization reads from obs/var columns by convention.
        metric_annotation_warnings = annotate_adata_with_qc_metrics(
            adata=output_adata,
            metrics_result=metrics_result,
        )

        # Initialize addon metrics dictionary.
        addon_metrics: dict[str, dict] = {}

        # Cell-cycle scoring (opt-in): fill default Tirosh lists when empty.
        if qc_config.cell_cycle.enabled:
            from cellquorum.qc.cell_cycle import (
                TIROSH_G2M_GENES,
                TIROSH_S_GENES,
                score_cell_cycle,
            )

            cc_config = qc_config.cell_cycle
            if not cc_config.s_genes:
                cc_config = cc_config.model_copy(update={"s_genes": TIROSH_S_GENES})
            if not cc_config.g2m_genes:
                cc_config = cc_config.model_copy(update={"g2m_genes": TIROSH_G2M_GENES})
            cc_metrics = score_cell_cycle(output_adata, cc_config)
            addon_metrics["cell_cycle"] = cc_metrics

        # Doublet detection (flag-only unless config.remove): consensus over methods.
        if qc_config.doublets.enabled:
            from cellquorum.qc.doublets import detect_doublets

            backend = None
            registry = getattr(context, "backend_registry", None)
            if registry is not None:
                try:
                    backend = registry.get("rscript")
                except Exception:
                    backend = None

            # Resolve the sample/library key for per-sample doublet detection:
            # prefer the cohort sample_key, then a plain sample_id column. Doublet
            # detectors should model each capture separately, not the pooled set.
            qc_context_config = getattr(context, "config", None)
            qc_cohort = getattr(qc_context_config, "cohort", None)
            doublet_sample_key = None
            for candidate in (
                getattr(qc_cohort, "sample_key", None),
                "sample_id",
            ):
                if candidate and candidate in output_adata.obs.columns:
                    doublet_sample_key = candidate
                    break

            doublet_metrics = detect_doublets(
                output_adata,
                qc_config.doublets,
                backend,
                sample_key=doublet_sample_key,
            )
            addon_metrics["doublets"] = doublet_metrics

            # Honor doublets.remove (config-gated): drop consensus-flagged
            # doublets from the output object. This is the ONLY QC path that
            # removes cells beyond threshold filtering, and it defaults off.
            if qc_config.doublets.remove and "predicted_doublet" in output_adata.obs.columns:
                doublet_mask = output_adata.obs["predicted_doublet"].to_numpy(dtype=bool)
                n_removed = int(doublet_mask.sum())
                if n_removed > 0:
                    output_adata = output_adata[~doublet_mask].copy()
                # Record the removal in the doublet metrics for provenance.
                doublet_metrics = {**doublet_metrics, "n_removed": n_removed}
                addon_metrics["doublets"] = doublet_metrics

        # Resolve group_key for QC figure grouping. Prefer the central cohort
        # schema (condition, then donor, then sample), then fall back to the
        # design block, then a plain sample_id column.
        group_key = None
        context_config = getattr(context, "config", None)
        cohort = getattr(context_config, "cohort", None)
        design = getattr(context_config, "design", None)
        candidates = [
            getattr(cohort, "condition_key", None),
            getattr(cohort, "donor_key", None),
            getattr(cohort, "sample_key", None),
            getattr(design, "condition_col", None),
            getattr(design, "donor_col", None),
            "sample_id",
        ]
        for candidate in candidates:
            if candidate and candidate in output_adata.obs.columns:
                group_key = candidate
                break

        # Write all configured QC artifacts.
        artifact_manifest = write_qc_artifacts(
            output_dir=output_dir,
            metrics_result=metrics_result,
            threshold_result=threshold_result,
            decision_result=decision_result,
            config=qc_config,
            adata=output_adata,
            summary_extra=build_qc_stage_summary_extra(
                context=context,
                qc_config=qc_config,
                stage_name=self.name,
            ),
            group_key=group_key,
        )

        # Convert artifact manifest paths into StageArtifact records.
        stage_artifacts = build_stage_artifacts_from_manifest(artifact_manifest)

        # Combine warnings from all QC layers.
        warnings = collect_qc_stage_warnings(
            metrics_result=metrics_result,
            threshold_result=threshold_result,
            decision_result=decision_result,
            artifact_manifest=artifact_manifest,
        )

        # Surface any preserved-not-overwritten metric-column conflicts.
        warnings.extend(metric_annotation_warnings)

        # No-silent-decisions guard: in a no-drop mode (flag_no_drop)
        # flagged cells REMAIN in the object and flow into every downstream stage.
        # Say so loudly — a resolved config's benign-looking mode must not hide the
        # fact that N% of cells failed QC yet nothing was removed.
        if not qc_config.should_filter():
            n_failed = int(decision_result.summary.get("n_cells_failed", 0))
            n_total = int(decision_result.summary.get("n_cells", 0))
            if n_failed > 0:
                pct = (100.0 * n_failed / n_total) if n_total else 0.0
                no_drop_msg = (
                    f"QC mode '{qc_config.mode}' flagged {n_failed} of {n_total} cells "
                    f"({pct:.1f}%) as failing QC but did NOT remove them — they remain "
                    "in the object and enter downstream analysis. Set qc.mode='filter' "
                    "or 'both' to drop them."
                )
                logger.warning(no_drop_msg)
                warnings.append(no_drop_msg)

        # Build human-readable stage notes.
        notes = build_qc_stage_notes(
            qc_config=qc_config,
            decision_result=decision_result,
            input_adata=adata,
            output_adata=output_adata,
        )

        # Build structured stage metrics for provenance.
        stage_metrics = build_qc_stage_metrics(
            stage_name=self.name,
            qc_config=qc_config,
            metrics_result=metrics_result,
            threshold_result=threshold_result,
            decision_result=decision_result,
            artifact_manifest=artifact_manifest,
            input_adata=adata,
            output_adata=output_adata,
        )

        # Merge addon metrics (cell-cycle, doublets) into stage metrics.
        if addon_metrics:
            stage_metrics.update(addon_metrics)

        # Return the stage result.
        return StageResult(
            adata=output_adata,
            artifacts=stage_artifacts,
            notes=notes,
            warnings=warnings,
            metrics=stage_metrics,
        )


def resolve_qc_config(
    context: object,
    *,
    override: QCConfig | None = None,
) -> QCConfig:
    """
    Resolve the effective QC configuration for a stage run.

    Resolution order:

    1. explicit QCStage(config=...) override
    2. context.config when it is already a QCConfig
    3. context.config.qc when present
    4. context.config["qc"] when present
    5. QCConfig() defaults

    Args:
        context: PipelineContext-like object.
        override: Optional explicit QCConfig override.

    Returns:
        Resolved QCConfig.

    Raises:
        QCStageError: If the resolved QC config is invalid.
    """

    # Prefer the explicit stage-level override.
    if override is not None:
        # Validate override type.
        if not isinstance(override, QCConfig):
            raise QCStageError(
                "QCStage config override must be a QCConfig object. "
                f"Received: {type(override).__name__}."
            )

        # Return the override.
        return override

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Accept context.config as a QCConfig directly.
    if isinstance(context_config, QCConfig):
        return context_config

    # Resolve context.config["qc"] for dictionary-like configs.
    if isinstance(context_config, Mapping) and "qc" in context_config:
        return coerce_qc_config(context_config["qc"])

    # Resolve context.config.qc for object-like configs.
    if hasattr(context_config, "qc"):
        return coerce_qc_config(context_config.qc)

    # Fall back to default QC configuration.
    return QCConfig()


def coerce_qc_config(value: object) -> QCConfig:
    """
    Coerce a candidate QC config value into QCConfig.

    Args:
        value: Candidate QC configuration value.

    Returns:
        Validated QCConfig.

    Raises:
        QCStageError: If the candidate cannot become QCConfig.
    """

    # Preserve QCConfig objects.
    if isinstance(value, QCConfig):
        return value

    # Validate dictionary-like QC configuration.
    if isinstance(value, Mapping):
        return validate_qc_config_dict(value)

    # Reject unsupported values.
    raise QCStageError(
        "QC configuration must be a QCConfig object or mapping. "
        f"Received: {type(value).__name__}."
    )


def is_qc_stage_enabled(context: object, qc_config: QCConfig) -> bool:
    """
    Return whether the QC stage should execute.

    The stage is enabled only when QCConfig.enabled is true and any top-level
    context.config.stages.qc flag is also true.

    Args:
        context: PipelineContext-like object.
        qc_config: Resolved QC configuration.

    Returns:
        True when QC should run, otherwise False.
    """

    # Respect the QC module-level enabled flag first.
    if not qc_config.enabled:
        return False

    # Read the context-level config if present.
    context_config = getattr(context, "config", None)

    # Handle dictionary-style stage selection.
    if isinstance(context_config, Mapping):
        # Extract the stages mapping.
        stages = context_config.get("stages")

        # Respect a dictionary-style stages.qc flag when present.
        if isinstance(stages, Mapping) and "qc" in stages:
            return bool(stages["qc"])

    # Handle object-style stage selection.
    stages = getattr(context_config, "stages", None)

    # Respect object-style stages.qc when present.
    if stages is not None and hasattr(stages, "qc"):
        return bool(stages.qc)

    # Default to enabled when no top-level stage selection is present.
    return True


def get_context_adata(context: object) -> ad.AnnData:
    """
    Retrieve AnnData from a PipelineContext-like object.

    Args:
        context: PipelineContext-like object.

    Returns:
        Active AnnData object.

    Raises:
        QCStageError: If AnnData is missing or invalid.
    """

    # Prefer the formal PipelineContext helper when present.
    require_adata = getattr(context, "require_adata", None)

    # Use require_adata when callable.
    if callable(require_adata):
        try:
            # Retrieve AnnData through the context helper.
            adata = require_adata()

        # Convert context errors into QC stage errors.
        except Exception as error:
            raise QCStageError("QC stage requires an AnnData object in context.") from error

    # Fall back to a direct context.adata attribute.
    else:
        # Retrieve direct AnnData attribute.
        adata = getattr(context, "adata", None)

    # Validate AnnData type.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "QC stage requires context.adata to be an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    # Return AnnData.
    return adata


def get_qc_output_dir(context: object, output_subdir: str) -> Path:
    """
    Resolve the QC stage output directory.

    Args:
        context: PipelineContext-like object with paths.results.
        output_subdir: QC subdirectory under results.

    Returns:
        QC artifact output directory.

    Raises:
        QCStageError: If context paths are missing or invalid.
    """

    # Reject empty output subdirectories.
    if not isinstance(output_subdir, str) or not output_subdir.strip():
        raise QCStageError("QCStage output_subdir must be a non-empty string.")

    # Retrieve the context paths object.
    paths = getattr(context, "paths", None)

    # Require context paths.
    if paths is None:
        raise QCStageError("QC stage requires context.paths with a results directory.")

    # Require a results directory on the paths object.
    if not hasattr(paths, "results"):
        raise QCStageError("QC stage requires context.paths.results.")

    # Resolve the results directory.
    results_dir = Path(paths.results)

    # Return the QC output directory.
    return results_dir / output_subdir


def build_qc_output_adata(
    *,
    adata: ad.AnnData,
    decision_result: QCDecisionResult,
    config: QCConfig,
) -> ad.AnnData:
    """
    Build the QC-updated AnnData object.

    The output AnnData is always annotated with QC decision columns. It is
    filtered only when config.mode is filter or both.

    Args:
        adata: Input AnnData object.
        decision_result: QC decision result.
        config: QC configuration.

    Returns:
        Annotated and optionally filtered AnnData object.
    """

    # Validate input AnnData.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "build_qc_output_adata expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    # Validate decision result type.
    if not isinstance(decision_result, QCDecisionResult):
        raise QCStageError(
            "decision_result must be a QCDecisionResult. "
            f"Received: {type(decision_result).__name__}."
        )

    # Validate QC config type.
    if not isinstance(config, QCConfig):
        raise QCStageError("config must be a QCConfig. " f"Received: {type(config).__name__}.")

    # Copy and annotate the AnnData object with QC decisions.
    output_adata = annotate_adata_with_qc_decisions(adata, decision_result)

    # Return annotated, unfiltered AnnData in report-only mode.
    if not config.should_filter():
        return output_adata

    # Return annotated and filtered AnnData in filter or both mode.
    return filter_adata_by_qc_decisions(output_adata, decision_result)


def annotate_adata_with_qc_decisions(
    adata: ad.AnnData,
    decision_result: QCDecisionResult,
) -> ad.AnnData:
    """
    Add QC decision columns to AnnData.obs and AnnData.var.

    Args:
        adata: Input AnnData object.
        decision_result: QC decision result.

    Returns:
        Copy of AnnData with QC decision annotations.

    Raises:
        QCStageError: If decision table indices do not match AnnData names.
    """

    # Validate cell decision alignment.
    validate_decision_index_alignment(
        expected=list(adata.obs_names),
        observed=list(decision_result.cell_decisions.index),
        label="cell_decisions",
    )

    # Validate gene decision alignment.
    validate_decision_index_alignment(
        expected=list(adata.var_names),
        observed=list(decision_result.gene_decisions.index),
        label="gene_decisions",
    )

    # Copy the AnnData object before mutation.
    annotated = adata.copy()

    # Add cell-level decision columns to obs.
    add_decision_columns_to_axis(
        axis_frame=annotated.obs,
        decisions=decision_result.cell_decisions,
        prefix="cellquorum_qc_",
    )

    # Add gene-level decision columns to var.
    add_decision_columns_to_axis(
        axis_frame=annotated.var,
        decisions=decision_result.gene_decisions,
        prefix="cellquorum_qc_",
    )

    # Return annotated AnnData.
    return annotated


def annotate_adata_with_qc_metrics(
    *,
    adata: ad.AnnData,
    metrics_result: QCMetricsResult,
) -> list[str]:
    """
    Add calculated QC metric columns to an AnnData object in place.

    QC metrics are calculated as explicit tables first. Plotting and downstream
    inspection, however, expect common cell-level metrics such as
    ``pct_counts_mito`` to be available on ``adata.obs``. This helper aligns the
    metric tables to the possibly filtered QC AnnData and stores non-conflicting
    metric columns on ``obs`` and ``var``. Pre-existing columns are preserved,
    never overwritten.

    Args:
        adata: QC AnnData to annotate.
        metrics_result: Calculated QC metrics.

    Returns:
        Human-readable warnings for any metric columns skipped because they
        already existed on ``obs``/``var``.

    Raises:
        QCStageError: If the QC AnnData axes cannot be aligned to the metric
            tables.
    """

    # Validate input types.
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "annotate_adata_with_qc_metrics expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )
    if not isinstance(metrics_result, QCMetricsResult):
        raise QCStageError(
            "metrics_result must be a QCMetricsResult. "
            f"Received: {type(metrics_result).__name__}."
        )

    # Align and add cell-level metrics.
    cell_metrics = align_metric_table_to_axis(
        axis_names=adata.obs_names,
        metric_table=metrics_result.cell_metrics,
        axis_label="obs",
    )
    obs_conflicts = add_metric_columns_to_axis(axis_frame=adata.obs, metrics=cell_metrics)

    # Align and add gene-level metrics.
    gene_metrics = align_metric_table_to_axis(
        axis_names=adata.var_names,
        metric_table=metrics_result.gene_metrics,
        axis_label="var",
    )
    var_conflicts = add_metric_columns_to_axis(axis_frame=adata.var, metrics=gene_metrics)

    # Report preserved-not-overwritten columns so the QC stage can warn.
    warnings: list[str] = []
    if obs_conflicts:
        warnings.append(
            "QC metric columns already present in obs were preserved, not "
            f"overwritten: {', '.join(obs_conflicts)}."
        )
    if var_conflicts:
        warnings.append(
            "QC metric columns already present in var were preserved, not "
            f"overwritten: {', '.join(var_conflicts)}."
        )
    return warnings


def align_metric_table_to_axis(
    *,
    axis_names: pd.Index,
    metric_table: pd.DataFrame,
    axis_label: str,
) -> pd.DataFrame:
    """
    Align a QC metric table to AnnData obs/var names.

    Args:
        axis_names: AnnData axis names.
        metric_table: QC metric table indexed by the original axis names.
        axis_label: Human-readable axis label for errors.

    Returns:
        Metric table aligned to ``axis_names``.

    Raises:
        QCStageError: If axis names are not present in the metric table.
    """

    # Fast path: no filtering/reordering occurred.
    if list(axis_names) == list(metric_table.index):
        return metric_table

    # Reindex supports filtered outputs while preserving the QC AnnData order.
    missing = pd.Index(axis_names).difference(metric_table.index)
    if len(missing) > 0:
        preview = ", ".join(map(str, missing[:5]))
        raise QCStageError(
            f"Cannot annotate QC {axis_label} metrics: {len(missing)} axis name(s) "
            f"are missing from the metric table. First missing: {preview}."
        )

    return metric_table.reindex(axis_names)


def add_metric_columns_to_axis(
    *,
    axis_frame: pd.DataFrame,
    metrics: pd.DataFrame,
) -> list[str]:
    """
    Add metric-table columns to an AnnData axis frame by row order.

    Pre-existing columns on ``axis_frame`` are never overwritten: an upstream
    tool may have populated ``total_counts`` or ``pct_counts_mito`` with values
    we must not silently clobber. Conflicting metric columns are skipped and
    returned so the caller can surface them as warnings.

    Args:
        axis_frame: AnnData obs or var DataFrame.
        metrics: Aligned metric table.

    Returns:
        Names of metric columns skipped because they already existed on
        ``axis_frame``.
    """

    # Store unprefixed metric names so plotting code and users see standard
    # Scanpy-style QC columns: total_counts, pct_counts_mito, etc. Skip any
    # column already present rather than overwriting it (flag-not-clobber).
    conflicts: list[str] = []
    for column in metrics.columns:
        if column in axis_frame.columns:
            conflicts.append(str(column))
            continue
        axis_frame[column] = metrics[column].to_numpy()

    return conflicts


def add_decision_columns_to_axis(
    *,
    axis_frame: pd.DataFrame,
    decisions: pd.DataFrame,
    prefix: str,
) -> None:
    """
    Add decision-table columns to AnnData obs or var.

    Args:
        axis_frame: AnnData obs or var DataFrame.
        decisions: Decision table aligned to axis_frame by row order.
        prefix: Prefix for stored QC decision columns.
    """

    # Iterate over decision columns.
    for column in decisions.columns:
        # Build the output column name.
        output_column = f"{prefix}{column}"

        # Store values by position to preserve duplicate-name compatibility.
        axis_frame[output_column] = decisions[column].to_numpy()


def filter_adata_by_qc_decisions(
    adata: ad.AnnData,
    decision_result: QCDecisionResult,
) -> ad.AnnData:
    """
    Filter AnnData using QC decision tables.

    Args:
        adata: AnnData object already aligned to decision tables.
        decision_result: QC decision result.

    Returns:
        Filtered AnnData object.

    Raises:
        QCStageError: If decision table indices are not aligned.
    """

    # Validate cell decision alignment.
    validate_decision_index_alignment(
        expected=list(adata.obs_names),
        observed=list(decision_result.cell_decisions.index),
        label="cell_decisions",
    )

    # Validate gene decision alignment.
    validate_decision_index_alignment(
        expected=list(adata.var_names),
        observed=list(decision_result.gene_decisions.index),
        label="gene_decisions",
    )

    # Build the cell keep mask.
    keep_cells = decision_result.cell_decisions["keep"].to_numpy(dtype=bool)

    # Build the gene keep mask.
    keep_genes = decision_result.gene_decisions["keep"].to_numpy(dtype=bool)

    # Return the filtered AnnData object.
    return adata[keep_cells, keep_genes].copy()


def validate_decision_index_alignment(
    *,
    expected: list[str],
    observed: list[str],
    label: str,
) -> None:
    """
    Validate that a decision table index matches AnnData axis names exactly.

    Args:
        expected: AnnData axis names in order.
        observed: Decision table index values in order.
        label: Human-readable decision table label.

    Raises:
        QCStageError: If indices differ.
    """

    # Return silently when indices match exactly.
    if expected == observed:
        return

    # Raise a clear alignment error.
    raise QCStageError(
        f"{label} index does not match the corresponding AnnData axis names. "
        "QC decisions must be aligned before annotation or filtering."
    )


def build_disabled_qc_stage_result(
    *,
    adata: ad.AnnData,
    stage_name: str,
    qc_config: QCConfig,
) -> StageResult:
    """
    Build a no-op StageResult for disabled QC.

    Args:
        adata: Active AnnData object.
        stage_name: Stable stage name.
        qc_config: Resolved QC configuration.

    Returns:
        StageResult representing a disabled QC no-op.
    """

    # Return a no-op stage result.
    return StageResult(
        adata=adata,
        artifacts=[],
        notes=["QC stage skipped because QC is disabled."],
        warnings=[],
        metrics={
            "stage_name": stage_name,
            "enabled": False,
            "mode": qc_config.mode,
            "reason": "qc_disabled",
        },
    )


def build_qc_stage_summary_extra(
    *,
    context: object,
    qc_config: QCConfig,
    stage_name: str,
) -> dict[str, object]:
    """
    Build extra summary values for qc_summary.json.

    Args:
        context: PipelineContext-like object.
        qc_config: QC configuration.
        stage_name: Stable stage name.

    Returns:
        Extra JSON-friendly QC summary fields.
    """

    # Return stage-level context metadata.
    return {
        "stage_name": stage_name,
        "run_id": str(getattr(context, "run_id", "cellquorum-run")),
        "random_seed": int(getattr(context, "random_seed", 1337)),
        "mode": qc_config.mode,
        "threshold_strategy": qc_config.threshold_strategy,
        "enabled_metric_families": qc_config.enabled_metric_families(),
    }


def build_stage_artifacts_from_manifest(
    manifest: QCArtifactManifest,
) -> list[StageArtifact]:
    """
    Convert a QCArtifactManifest into stage artifact records.

    Args:
        manifest: QC artifact manifest.

    Returns:
        StageArtifact records.
    """

    # Validate manifest type.
    if not isinstance(manifest, QCArtifactManifest):
        raise QCStageError(
            "manifest must be a QCArtifactManifest. " f"Received: {type(manifest).__name__}."
        )

    # Initialize stage artifacts.
    artifacts: list[StageArtifact] = []

    # Convert each written artifact into a stage artifact.
    for artifact_name, artifact_value in manifest.artifacts.items():
        # Figures are stored as a list of paths; create multiple artifacts.
        if artifact_name == "figures" and isinstance(artifact_value, list):
            for idx, figure_path_str in enumerate(artifact_value):
                figure_path = Path(figure_path_str)
                artifacts.append(
                    StageArtifact(
                        name=f"qc_figure_{idx}",
                        path=figure_path,
                        kind=infer_artifact_kind(figure_path),
                        description=f"QC diagnostic figure {figure_path.name}",
                    )
                )
        else:
            # Other artifacts are single Path objects.
            artifacts.append(
                StageArtifact(
                    name=f"qc_{artifact_name}",
                    path=artifact_value,
                    kind=infer_artifact_kind(artifact_value),
                    description=describe_qc_artifact(artifact_name),
                )
            )

    # Return stage artifacts.
    return artifacts


def infer_artifact_kind(path: Path) -> str:
    """
    Infer a StageArtifact kind from a path suffix.

    Args:
        path: Artifact path.

    Returns:
        Artifact kind label.
    """

    # Normalize the file suffix.
    suffix = path.suffix.lower()

    # Map CSV files.
    if suffix == ".csv":
        return "csv"

    # Map JSON files.
    if suffix == ".json":
        return "json"

    # Map AnnData h5ad files.
    if suffix == ".h5ad":
        return "h5ad"

    # Return a generic file kind otherwise.
    return "file"


def describe_qc_artifact(artifact_name: str) -> str:
    """
    Return a human-readable description for a QC artifact.

    Args:
        artifact_name: Stable QC artifact label.

    Returns:
        Description string.
    """

    # Define artifact descriptions.
    descriptions = {
        "cell_metrics": "Cell-level QC metric table.",
        "gene_metrics": "Gene-level QC metric table.",
        "feature_masks": "Feature-family QC mask table.",
        "thresholds": "QC threshold table.",
        "cell_decisions": "Cell-level QC keep/fail decision table.",
        "gene_decisions": "Gene-level QC keep/fail decision table.",
        "qc_h5ad": "QC-annotated AnnData object.",
        "summary": "Structured QC summary JSON.",
    }

    # Return a known description or a fallback.
    return descriptions.get(artifact_name, f"QC artifact: {artifact_name}.")


def collect_qc_stage_warnings(
    *,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    artifact_manifest: QCArtifactManifest,
) -> list[str]:
    """
    Collect warnings from all QC stage layers.

    Args:
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
        artifact_manifest: QC artifact manifest.

    Returns:
        Combined warning list.
    """

    # Return warnings in execution order.
    return [
        *metrics_result.warnings,
        *threshold_result.warnings,
        *decision_result.warnings,
        *artifact_manifest.warnings,
    ]


def build_qc_stage_notes(
    *,
    qc_config: QCConfig,
    decision_result: QCDecisionResult,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> list[str]:
    """
    Build human-readable QC stage notes.

    Args:
        qc_config: QC configuration.
        decision_result: QC decision result.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        Stage note strings.
    """

    # Retrieve decision summary.
    summary = decision_result.summary

    # Initialize notes.
    notes = [
        f"QC completed in {qc_config.mode} mode.",
        (
            "Cells kept: "
            f"{summary['n_cells_kept']}/{summary['n_cells']}; "
            "genes kept: "
            f"{summary['n_genes_kept']}/{summary['n_genes']}."
        ),
    ]

    # Add an explicit filtering note when filtering occurred.
    if qc_config.should_filter():
        notes.append(
            "QC filtering changed AnnData shape from "
            f"{input_adata.n_obs} cells x {input_adata.n_vars} genes to "
            f"{output_adata.n_obs} cells x {output_adata.n_vars} genes."
        )

    # Return stage notes.
    return notes


def build_qc_stage_metrics(
    *,
    stage_name: str,
    qc_config: QCConfig,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    artifact_manifest: QCArtifactManifest,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> dict[str, object]:
    """
    Build structured QC stage metrics for provenance.

    Args:
        stage_name: Stable stage name.
        qc_config: QC configuration.
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
        artifact_manifest: QC artifact manifest.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        JSON-friendly stage metrics.
    """

    # Return structured metrics.
    return {
        "stage_name": stage_name,
        "enabled": True,
        "mode": qc_config.mode,
        "threshold_strategy": qc_config.threshold_strategy,
        "input_shape": {
            "n_obs": int(input_adata.n_obs),
            "n_vars": int(input_adata.n_vars),
        },
        "output_shape": {
            "n_obs": int(output_adata.n_obs),
            "n_vars": int(output_adata.n_vars),
        },
        "metric_summary": metrics_result.to_summary_dict(),
        "threshold_summary": threshold_result.to_summary_dict(),
        "decision_summary": decision_result.to_summary_dict(),
        "artifact_manifest": artifact_manifest.to_dict(),
    }


__all__ = [
    "QCStage",
    "QCStageError",
    "add_decision_columns_to_axis",
    "add_metric_columns_to_axis",
    "annotate_adata_with_qc_decisions",
    "annotate_adata_with_qc_metrics",
    "align_metric_table_to_axis",
    "build_disabled_qc_stage_result",
    "build_qc_output_adata",
    "build_qc_stage_metrics",
    "build_qc_stage_notes",
    "build_qc_stage_summary_extra",
    "build_stage_artifacts_from_manifest",
    "coerce_qc_config",
    "collect_qc_stage_warnings",
    "describe_qc_artifact",
    "filter_adata_by_qc_decisions",
    "get_context_adata",
    "get_qc_output_dir",
    "infer_artifact_kind",
    "is_qc_stage_enabled",
    "resolve_qc_config",
    "validate_decision_index_alignment",
]
