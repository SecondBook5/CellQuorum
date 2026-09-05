# Pipeline step (order=20): qc — report helpers for the QC stage.
"""Turning a QC run into its reported result: artifacts, warnings, notes, metrics.

The no-silent-decisions rule lives mostly here. A QC stage that flags cells without
removing them, or that could not fit a mixture model, or that removed cells at different
rates in different study arms, has to say so in a place a reader will actually look —
which means the stage result, not a log line at ``--verbose``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellquorum.core.stage import StageArtifact, StageResult
from cellquorum.stages.qc._errors import QCStageError
from cellquorum.stages.qc.artifacts import QCArtifactManifest
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.floors import FloorResult
from cellquorum.stages.qc.metrics import QCMetricsResult


def resolve_publication_qc_keys(
    *,
    adata: ad.AnnData,
    cohort: object | None,
    design: object | None,
) -> dict[str, str]:
    """
    Resolve the obs columns and condition labels the publication panels need.

    ``write_publication_qc_figures`` defaults to ``patient_id``/``condition``,
    which no CellQuorum cohort schema uses — cohorts declare ``donor_key``. With
    the defaults left in place the writer raised ``QCPublicationFigureError`` and
    the caller swallowed it into a warning, so every run silently shipped the
    fallback diagnostic plots instead of the publication suite. Resolving the
    names from the cohort schema is what makes those panels actually render.

    Only keys that resolve to a real obs column are returned, so the writer keeps
    its own defaults for anything this cohort does not declare.

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

    # Donor/patient axis: the panels group and order samples by it.
    patient_key = first_present(
        getattr(cohort, "donor_key", None),
        getattr(design, "donor_col", None),
        "donor_id",
        "patient_id",
    )
    if patient_key:
        keys["patient_key"] = patient_key

    # Library/sample axis.
    sample_key = first_present(getattr(cohort, "sample_key", None), "sample_id")
    if sample_key:
        keys["sample_key"] = sample_key

    # Condition axis.
    condition_key = first_present(
        getattr(cohort, "condition_key", None),
        getattr(design, "condition_col", None),
        "condition",
    )
    if condition_key:
        keys["condition_key"] = condition_key

    # Condition labels drive the N/LE display mapping and panel colours. Prefer
    # the design's explicit control/case, then the cohort's ordered levels.
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
        "floors": qc_config.floors.model_dump(),
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
            f"manifest must be a QCArtifactManifest. Received: {type(manifest).__name__}."
        )

    # Initialize stage artifacts.
    artifacts: list[StageArtifact] = []

    # Convert each written artifact into a stage artifact.
    for artifact_name, artifact_value in manifest.artifacts.items():
        # Some entries are a set of files rather than one — the diagnostic figures,
        # the typeset tables in their three formats. Any list expands to one record
        # per path, keyed on the manifest name.
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
        "report": "Per-group QC report table (cells before/removed/%/after + TOTAL).",
        "html_report": "Browsable QC report (self-contained HTML, sortable tables).",
        "publication_tables": "Typeset QC tables (HTML page, booktabs LaTeX, raster).",
        "qc_h5ad": "QC-annotated AnnData object.",
        "summary": "Structured QC summary JSON.",
    }

    # Return a known description or a fallback.
    return descriptions.get(artifact_name, f"QC artifact: {artifact_name}.")


def collect_qc_stage_warnings(
    *,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    artifact_manifest: QCArtifactManifest,
) -> list[str]:
    """
    Collect warnings from all QC stage layers.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks, reasons and counts.
        artifact_manifest: QC artifact manifest.

    Returns:
        Combined warning list.
    """

    # Return warnings in execution order.
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
    """
    Build human-readable QC stage notes.

    Args:
        qc_config: QC configuration.
        floors: Floor masks, reasons and counts.
        input_adata: Input AnnData object.
        output_adata: Output AnnData object.

    Returns:
        Stage note strings.
    """

    summary = floors.summary

    # There is no mode to report: floors always remove what they judge, and grading never
    # removes anything. The note therefore states what the floors did and lets the graded
    # metrics speak for the verdict.
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

    # Return stage notes.
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
    """
    Build structured QC stage metrics for provenance.

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

    # Return structured metrics.
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
    """Flatten an audit table to a dict of lists, which anndata can store natively.

    A list of per-row dicts round-trips through h5ad as a single stringified blob, so an audit
    written that way survives the run and cannot be read back by a figure or a report.
    """
    frame = table.reset_index(names=index_name)
    return {str(column): frame[column].tolist() for column in frame.columns}
