"""QC result annotation, report metadata, and stage summaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc.artifacts import QCArtifactManifest
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.floors import FLOOR_REASON_COLUMN, FloorResult
from cellquorum.stages.qc.metrics import QCMetricsResult
from cellquorum.stats.donor_comparison import TwoGroupTest, two_group_test_on_donor_medians

KEEP_COLUMN = "cellquorum_qc_keep"


def summarize_ambient_correction(adata: ad.AnnData) -> dict[str, object]:
    """Report upstream correction provenance without inferring it from QC settings."""
    provenance = adata.uns.get("cellquorum", {})
    record = provenance.get("ambient_correction") if isinstance(provenance, Mapping) else None
    if not isinstance(record, Mapping) or not record.get("method"):
        return {"status": "not_recorded", "method": None, "n_libraries": 0}
    fractions = record.get("contamination_fractions", {})
    return {
        "status": "upstream_recorded",
        "method": str(record["method"]),
        "n_libraries": len(fractions) if isinstance(fractions, Mapping) else 0,
    }


def compute_qc_donor_comparisons(
    adata: ad.AnnData, keys: dict[str, str], *, paired: bool
) -> tuple[dict[str, TwoGroupTest], list[str]]:
    """Prepare donor-level QC results before artifact writing and rendering."""
    from cellquorum.visualization.qc.panels import assemble_qc_frame, summarize_by_sample

    sample, donor, condition = (keys.get(k) for k in ("sample_key", "patient_key", "condition_key"))
    control, case = keys.get("normal_label"), keys.get("disease_label")
    if not paired or not all((sample, donor, condition, control, case)):
        return {}, []
    metadata = adata.obs[[sample, donor, condition]]
    if (
        metadata.isna().any().any()
        or metadata.astype("string").apply(lambda values: values.str.strip().eq("")).any().any()
    ):
        return {}, ["QC donor comparisons omitted: sample, donor, or condition labels are missing."]
    if (metadata.groupby(sample, observed=True)[[donor, condition]].nunique() > 1).any().any():
        return {}, ["QC donor comparisons omitted: a sample maps to multiple donors or conditions."]
    frame = assemble_qc_frame(
        obs=adata.obs,
        sample_key=sample,
        donor_key=donor,
        condition_key=condition,
    )
    table = summarize_by_sample(frame)
    comparisons = {}
    for metric in table.select_dtypes(include="number"):
        if metric in {"cells_in", "cells_kept", "cells_removed"}:
            continue
        test = two_group_test_on_donor_medians(
            table,
            value_col=metric,
            group_col="condition",
            donor_col="donor",
            group1=control,
            group2=case,
            paired=True,
            incomplete_pairs="drop",
        )
        if test is not None:
            comparisons[metric] = test
    return comparisons, []


def build_qc_output_adata(*, adata: ad.AnnData, floors: FloorResult) -> ad.AnnData:
    """Annotate with floor outcomes and drop what is below the floor.

    Args:
        adata: Input AnnData object.
        floors: Masks and reasons from :func:`cellquorum.stages.qc.floors.apply_floors`.

    Returns:
        The object with floor columns written, restricted to what cleared the floor.
    """
    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            f"build_qc_output_adata expected an AnnData object. Received: {type(adata).__name__}."
        )

    cells = floors.cell_keep.reindex(adata.obs_names).fillna(True).to_numpy(dtype=bool)
    genes = floors.gene_keep.reindex(adata.var_names).fillna(True).to_numpy(dtype=bool)
    output_adata = adata.copy() if cells.all() and genes.all() else adata[cells, genes].copy()
    output_adata.obs[FLOOR_REASON_COLUMN] = (
        floors.reason.reindex(output_adata.obs_names).fillna("").to_numpy()
    )
    return output_adata


def build_qc_figure_adata(
    *,
    adata: ad.AnnData,
    output_adata: ad.AnnData,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
) -> ad.AnnData:
    """The object figures render from: every input cell, carrying floor outcomes.

    Args:
        adata: The unfiltered input.
        output_adata: The filtered output, used for its graded columns.
        metrics_result: Computed metrics, indexed by every input cell.
        floors: Floor masks and reasons.

    Returns:
        An unfiltered object with metrics, floor outcomes and graded columns on obs.
    """

    figure_adata = ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy())
    figure_adata.obs[FLOOR_REASON_COLUMN] = (
        floors.reason.reindex(figure_adata.obs_names).fillna("").to_numpy()
    )
    figure_adata.obs[KEEP_COLUMN] = (
        floors.cell_keep.reindex(figure_adata.obs_names).fillna(True).to_numpy()
    )
    annotate_adata_with_qc_metrics(adata=figure_adata, metrics_result=metrics_result)

    for column in output_adata.obs.columns:
        if column not in figure_adata.obs.columns:
            figure_adata.obs[column] = output_adata.obs[column].reindex(figure_adata.obs_names)
    return figure_adata


def annotate_adata_with_qc_metrics(
    *,
    adata: ad.AnnData,
    metrics_result: QCMetricsResult,
) -> list[str]:
    """Add calculated QC metric columns to an AnnData object in place.

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

    if not isinstance(adata, ad.AnnData):
        raise QCStageError(
            "annotate_adata_with_qc_metrics expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )
    if not isinstance(metrics_result, QCMetricsResult):
        raise QCStageError(
            f"metrics_result must be a QCMetricsResult. Received: {type(metrics_result).__name__}."
        )

    cell_metrics = align_metric_table_to_axis(
        axis_names=adata.obs_names,
        metric_table=metrics_result.cell_metrics,
        axis_label="obs",
    )
    obs_conflicts = add_metric_columns_to_axis(axis_frame=adata.obs, metrics=cell_metrics)

    gene_metrics = align_metric_table_to_axis(
        axis_names=adata.var_names,
        metric_table=metrics_result.gene_metrics,
        axis_label="var",
    )
    var_conflicts = add_metric_columns_to_axis(axis_frame=adata.var, metrics=gene_metrics)

    warnings: list[str] = []
    if obs_conflicts:
        warnings.append(
            "QC recomputed obs metric columns whose inherited values described a "
            f"different object; replaced: {', '.join(obs_conflicts)}."
        )
    if var_conflicts:
        warnings.append(
            "QC recomputed var metric columns whose inherited values described a "
            "different object (gene-level metrics are aggregates over cells and do "
            f"not survive subsetting); replaced: {', '.join(var_conflicts)}."
        )
    return warnings


def align_metric_table_to_axis(
    *,
    axis_names: pd.Index,
    metric_table: pd.DataFrame,
    axis_label: str,
) -> pd.DataFrame:
    """Align a QC metric table to AnnData obs/var names.

    Args:
        axis_names: AnnData axis names.
        metric_table: QC metric table indexed by the original axis names.
        axis_label: Human-readable axis label for errors.

    Returns:
        Metric table aligned to ``axis_names``.

    Raises:
        QCStageError: If axis names are not present in the metric table.
    """

    if list(axis_names) == list(metric_table.index):
        return metric_table

    missing = pd.Index(axis_names).difference(metric_table.index)
    if len(missing) > 0:
        preview = ", ".join(map(str, missing[:5]))
        raise QCStageError(
            f"Cannot annotate QC {axis_label} metrics: {len(missing)} axis name(s) "
            f"are missing from the metric table. First missing: {preview}."
        )

    return metric_table.reindex(axis_names)


def _values_agree(existing: np.ndarray, fresh: np.ndarray) -> bool:
    """Do a pre-existing metric column and the freshly computed one say the same thing?"""
    if existing.shape != fresh.shape:
        return False
    try:
        return bool(
            np.allclose(
                existing.astype("float64"), fresh.astype("float64"), rtol=1e-9, equal_nan=True
            )
        )
    except (TypeError, ValueError):
        return bool(np.array_equal(existing, fresh))


def _describe_range(values: np.ndarray) -> str:
    """A one-number summary that makes a wrong-scale column obvious. Max, so that an
    inherited whole-atlas gene count reads as 200072 next to the arm's own 2125.
    """
    try:
        finite = values.astype("float64")
        finite = finite[np.isfinite(finite)]
        return f"max {finite.max():.6g}" if finite.size else "all non-finite"
    except (TypeError, ValueError):
        return f"{len(values)} non-numeric value(s)"


def add_metric_columns_to_axis(
    *,
    axis_frame: pd.DataFrame,
    metrics: pd.DataFrame,
) -> list[str]:
    """Add metric-table columns to an AnnData axis frame by row order.

    Args:
        axis_frame: AnnData obs or var DataFrame.
        metrics: Aligned metric table.

    Returns:
        Descriptions of the columns whose pre-existing values DISAGREED and were
        replaced, each naming the old and new magnitude, for the caller to surface.
    """

    replaced: list[str] = []
    for column in metrics.columns:
        fresh = metrics[column].to_numpy()
        if column in axis_frame.columns:
            existing = axis_frame[column].to_numpy()
            if _values_agree(existing, fresh):
                continue
            replaced.append(f"{column} ({_describe_range(existing)} -> {_describe_range(fresh)})")
        axis_frame[column] = fresh

    return replaced


def resolve_publication_qc_keys(
    *,
    adata: ad.AnnData,
    cohort: object | None,
    design: object | None,
) -> dict[str, str]:
    """Resolve the obs columns and condition labels the publication panels need.

    Args:
        adata: Object whose obs columns are being matched against.
        cohort: Cohort schema block (donor_key, sample_key, condition_key).
        design: Design block (donor_col, condition_col, case, control).

    Returns:
        Keyword arguments for ``write_publication_qc_figures``.
    """

    def first_present(*candidates: object) -> str | None:
        """Return the first candidate that names an existing obs column."""
        for candidate in candidates:
            if candidate and str(candidate) in adata.obs.columns:
                return str(candidate)
        return None

    keys: dict[str, str] = {}

    patient_key = first_present(
        getattr(cohort, "donor_key", None),
        getattr(design, "donor_col", None),
        "donor_id",
        "patient_id",
    )
    if patient_key:
        keys["patient_key"] = patient_key

    sample_key = first_present(getattr(cohort, "sample_key", None), "sample_id")
    if sample_key:
        keys["sample_key"] = sample_key

    condition_key = first_present(
        getattr(cohort, "condition_key", None),
        getattr(design, "condition_col", None),
        "condition",
    )
    if condition_key:
        keys["condition_key"] = condition_key

    control = getattr(design, "control", None)
    case = getattr(design, "case", None)
    levels = getattr(cohort, "condition_levels", None)
    if not control and levels:
        control = levels[0]
    if not case and levels and len(levels) > 1:
        case = levels[-1]
    if control:
        keys["normal_label"] = str(control)
    if case:
        keys["disease_label"] = str(case)

    return keys


def build_disabled_qc_stage_result(
    *,
    adata: ad.AnnData,
    stage_name: str,
    qc_config: QCConfig,
) -> StageResult:
    """Build a no-op StageResult for disabled QC.

    Args:
        adata: Active AnnData object.
        stage_name: Stable stage name.
        qc_config: Resolved QC configuration.

    Returns:
        StageResult representing a disabled QC no-op.
    """

    return StageResult(
        adata=adata,
        artifacts=[],
        notes=["QC stage skipped because QC is disabled."],
        warnings=[],
        metrics={
            "stage_name": stage_name,
            "enabled": False,
            "reason": "qc_disabled",
        },
    )


def build_qc_stage_summary_extra(
    *,
    context: object,
    qc_config: QCConfig,
    stage_name: str,
) -> dict[str, object]:
    """Build extra summary values for qc_summary.json.

    Args:
        context: PipelineContext-like object.
        qc_config: QC configuration.
        stage_name: Stable stage name.

    Returns:
        Extra JSON-friendly QC summary fields.
    """

    return {
        "stage_name": stage_name,
        "run_id": str(getattr(context, "run_id", "cellquorum-run")),
        "random_seed": int(getattr(context, "random_seed", 1337)),
        "floors": qc_config.floors.model_dump(),
        "enabled_metric_families": qc_config.enabled_metric_families(),
    }


def build_stage_artifacts_from_manifest(
    manifest: QCArtifactManifest,
) -> list[StageArtifact]:
    """Convert a QCArtifactManifest into stage artifact records.

    Args:
        manifest: QC artifact manifest.

    Returns:
        StageArtifact records.
    """

    if not isinstance(manifest, QCArtifactManifest):
        raise QCStageError(
            f"manifest must be a QCArtifactManifest. Received: {type(manifest).__name__}."
        )

    artifacts: list[StageArtifact] = []

    for artifact_name, artifact_value in manifest.artifacts.items():
        if isinstance(artifact_value, list):
            is_figures = artifact_name == "figures"
            prefix = "qc_figure" if is_figures else f"qc_{artifact_name}"
            for idx, path_string in enumerate(artifact_value):
                path = Path(path_string)
                artifacts.append(
                    StageArtifact(
                        name=f"{prefix}_{idx}",
                        path=path,
                        kind=infer_artifact_kind(path),
                        description=(
                            f"QC diagnostic figure {path.name}"
                            if is_figures
                            else f"{describe_qc_artifact(artifact_name)} ({path.name})"
                        ),
                    )
                )
        else:
            artifacts.append(
                StageArtifact(
                    name=f"qc_{artifact_name}",
                    path=artifact_value,
                    kind=infer_artifact_kind(artifact_value),
                    description=describe_qc_artifact(artifact_name),
                )
            )

    return artifacts


def infer_artifact_kind(path: Path) -> str:
    """Infer a StageArtifact kind from a path suffix.

    Args:
        path: Artifact path.

    Returns:
        Artifact kind label.
    """

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return "csv"

    if suffix == ".json":
        return "json"

    if suffix == ".h5ad":
        return "h5ad"

    return "file"


def describe_qc_artifact(artifact_name: str) -> str:
    """Return a human-readable description for a QC artifact.

    Args:
        artifact_name: Stable QC artifact label.

    Returns:
        Description string.
    """

    descriptions = {
        "cell_metrics": "Cell-level QC metric table.",
        "gene_metrics": "Gene-level QC metric table.",
        "feature_masks": "Feature-family QC mask table.",
        "thresholds": "QC threshold table.",
        "cell_decisions": "Cell-level QC keep/fail decision table.",
        "gene_decisions": "Gene-level QC keep/fail decision table.",
        "report": "Per-group QC report table (cells before/removed/%/after + TOTAL).",
        "html_report": "Browsable QC report (self-contained HTML, sortable tables).",
        "publication_tables": "Typeset QC tables (HTML page, booktabs LaTeX, raster).",
        "qc_h5ad": "QC-annotated AnnData object.",
        "summary": "Structured QC summary JSON.",
    }

    return descriptions.get(artifact_name, f"QC artifact: {artifact_name}.")


def collect_qc_stage_warnings(
    *,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    artifact_manifest: QCArtifactManifest,
) -> list[str]:
    """Collect warnings from all QC stage layers.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks, reasons and counts.
        artifact_manifest: QC artifact manifest.

    Returns:
        Combined warning list.
    """

    return [
        *metrics_result.warnings,
        *floors.warnings,
        *artifact_manifest.warnings,
    ]


def build_qc_stage_notes(
    *,
    qc_config: QCConfig,
    floors: FloorResult,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> list[str]:
    """Build human-readable QC stage notes.

    Args:
        qc_config: QC configuration.
        floors: Floor masks, reasons and counts.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        Stage note strings.
    """

    summary = floors.summary

    notes = [
        (
            "QC floors: "
            f"{summary['n_cells'] - summary['n_cells_below_floor']}/{summary['n_cells']} "
            "barcodes and "
            f"{summary['n_genes'] - summary['n_genes_below_floor']}/{summary['n_genes']} "
            "genes cleared the detection floor."
        ),
    ]

    if input_adata.shape != output_adata.shape:
        notes.append(
            "QC floors changed AnnData shape from "
            f"{input_adata.n_obs} cells x {input_adata.n_vars} genes to "
            f"{output_adata.n_obs} cells x {output_adata.n_vars} genes."
        )

    return notes


def build_qc_stage_metrics(
    *,
    stage_name: str,
    qc_config: QCConfig,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    artifact_manifest: QCArtifactManifest,
    input_adata: ad.AnnData,
    output_adata: ad.AnnData,
) -> dict[str, object]:
    """Build structured QC stage metrics for provenance.

    Args:
        stage_name: Stable stage name.
        qc_config: QC configuration.
        metrics_result: QC metrics result.
        floors: Floor masks, reasons and counts.
        artifact_manifest: QC artifact manifest.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        JSON-friendly stage metrics.
    """

    return {
        "stage_name": stage_name,
        "enabled": True,
        "floors": qc_config.floors.model_dump(),
        "input_shape": {
            "n_obs": int(input_adata.n_obs),
            "n_vars": int(input_adata.n_vars),
        },
        "output_shape": {
            "n_obs": int(output_adata.n_obs),
            "n_vars": int(output_adata.n_vars),
        },
        "metric_summary": metrics_result.to_summary_dict(),
        "floor_summary": floors.to_summary_dict(),
        "artifact_manifest": artifact_manifest.to_dict(),
    }


def _audit_to_columns(table: pd.DataFrame, *, index_name: str) -> dict[str, list]:
    """Flatten an audit table to a dict of lists, which anndata can store natively."""
    frame = table.reset_index(names=index_name)
    return {str(column): frame[column].tolist() for column in frame.columns}


__all__ = []
