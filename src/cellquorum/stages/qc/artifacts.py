"""QC artifact writing utilities for CellQuorum."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from uuid import uuid4

import anndata as ad
import pandas as pd

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc.attrition import AttritionAudit
from cellquorum.stages.qc.config import QCConfig
from cellquorum.stages.qc.floors import FloorResult, build_qc_report_table
from cellquorum.stages.qc.metrics import QCMetricsResult
from cellquorum.stages.qc.mixture import MitoMixtureResult
from cellquorum.stats.donor_comparison import TwoGroupTest


class QCArtifactError(CellQuorumDataError):
    """Report QC artifact writing failures."""


@dataclass(frozen=True)
class QCArtifactManifest:
    """Store a manifest of QC artifacts written to disk.

    Args:
        output_dir: Directory where QC artifacts were written.
        artifacts: Mapping from artifact label to filesystem path.
        skipped: Artifact labels skipped because config disabled them or required
            inputs were missing.
        warnings: Non-fatal artifact writing warnings.
    """

    output_dir: Path
    artifacts: dict[str, Path | list[str]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert the artifact manifest into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of written and skipped artifacts.
        """

        return {
            "output_dir": str(self.output_dir),
            "artifacts": {
                artifact_name: (
                    [str(p) for p in artifact_path]
                    if isinstance(artifact_path, list)
                    else str(artifact_path)
                )
                for artifact_name, artifact_path in self.artifacts.items()
            },
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }

    def get_path(self, artifact_name: str) -> Path:
        """Return the path for a single-file artifact.

        Args:
            artifact_name: Stable artifact label.

        Returns:
            Path to the requested artifact.

        Raises:
            QCArtifactError: If the artifact was not written, or is a group of files rather
                than one — asking for "the" path of a figure set has no answer, and returning
                the first would silently drop the rest.
        """

        if artifact_name not in self.artifacts:
            raise QCArtifactError(
                f"QC artifact '{artifact_name}' was not written. "
                f"Available artifacts: {', '.join(self.artifacts) or '<none>'}."
            )

        written = self.artifacts[artifact_name]
        if isinstance(written, list):
            raise QCArtifactError(
                f"QC artifact '{artifact_name}' is a group of {len(written)} files, not one "
                f"path. Read it from `.artifacts[{artifact_name!r}]`."
            )

        return written


def _report_figure_order(written: object) -> list[Path]:
    """Pick and order the figures worth inlining in the HTML report."""
    if not isinstance(written, list):
        return []
    paths = [Path(str(item)) for item in written]
    by_name = {path.name: path for path in paths if path.suffix == ".png"}

    preferred = [
        "qc_overview.png",
        "graded_lineage_family.png",
        "graded_family_cooccurrence.png",
        "qc_joint_density.png",
        "qc_mito_mixture.png",
        "qc_metric_total_counts.png",
        "qc_metric_n_genes_by_counts.png",
        "qc_metric_pct_counts_mito.png",
        "qc_metric_pct_counts_ribo.png",
        "qc_metric_pct_counts_in_top_20_genes.png",
        "qc_metric_qc_ev_malat1_fraction_value.png",
        "qc_metric_qc_ev_dissociation_stress_value.png",
        "qc_metric_doublet_score.png",
        "qc_attrition.png",
    ]
    return [by_name[name] for name in preferred if name in by_name]


def write_qc_artifacts(
    *,
    output_dir: str | PathLike[str] | Path,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    mixture: MitoMixtureResult | None = None,
    config: QCConfig | None = None,
    adata: ad.AnnData | None = None,
    summary_extra: dict[str, object] | None = None,
    group_key: str | None = None,
    report_groups: pd.Series | None = None,
    report_group_name: str = "cell_type",
    figure_adata: ad.AnnData | None = None,
    publication_keys: dict[str, str] | None = None,
    attrition_audit: AttritionAudit | None = None,
    donor_comparisons: dict[str, TwoGroupTest] | None = None,
) -> QCArtifactManifest:
    """Write QC module artifacts to disk.

    Args:
        output_dir: Directory where QC artifacts should be written.
        metrics_result: Calculated QC metrics.
        floors: Floor masks, reasons and counts.
        mixture: Fitted mitochondrial mixture, or None.
        config: Optional QC configuration. Defaults to QCConfig().
        adata: Optional AnnData object to write as qc.h5ad when enabled.
        summary_extra: Optional extra JSON-friendly values to include in the
            summary artifact.
        group_key: Optional obs column name for grouping QC figures by
            condition, donor, or sample.
        report_groups: Optional per-cell group labels (typically cell type)
            aligned to every input cell, used to resolve
            the per-group QC report table. When None the report collapses to a
            single TOTAL row.
        report_group_name: Name of the leading group column in the QC report
            table (defaults to ``cell_type``).
        figure_adata: Optional PRE-filter AnnData used for figures only. Under
            ``mode="filter"`` the ``adata`` written as qc.h5ad has already had
            failing cells removed, so keep/fail figures drawn from it report a
            100% pass rate no matter how many cells were dropped. Pass the
            annotated pre-filter object here so the figures show the population
            QC actually acted on. Defaults to ``adata``.
        publication_keys: Optional resolved obs column names and condition
            labels for the publication panels, e.g.
            ``{"patient_key": "donor_id", "condition_key": "condition"}``. The
            publication writer defaults to ``patient_id``, which no CellQuorum
            cohort schema uses, so without this the whole suite raises and is
            swallowed into a warning.
        attrition_audit: Optional differential-attrition audit to write as
            ``qc_attrition.csv``. None writes nothing and records the skip, so a
            run predating the audit is distinguishable from one where it found
            nothing to test.

    Returns:
        QCArtifactManifest describing written, skipped, and warned artifacts.

    Raises:
        QCArtifactError: If inputs are invalid or writing fails.
    """

    qc_config = QCConfig() if config is None else config

    validate_qc_artifact_inputs(
        metrics_result=metrics_result,
        floors=floors,
        config=qc_config,
        adata=adata,
    )

    output_path = prepare_qc_output_dir(output_dir)

    artifacts: dict[str, Path | list[str]] = {}

    skipped: list[str] = []

    warnings: list[str] = []

    if donor_comparisons:
        records = []
        for metric, comparison in donor_comparisons.items():
            record = {"metric": metric, **comparison.to_dict()}
            for key in ("donors_group1", "donors_group2"):
                record[key] = json.dumps(record[key])
            records.append(record)
        artifacts["donor_comparisons"] = write_dataframe_artifact(
            pd.DataFrame(records),
            output_path / "qc_donor_comparisons.csv",
            index=False,
        )

    if qc_config.outputs.write_metrics_table:
        artifacts["cell_metrics"] = write_dataframe_artifact(
            _metrics_with_posterior(metrics_result.cell_metrics, mixture),
            output_path / "cell_metrics.csv",
            index=True,
        )

        artifacts["gene_metrics"] = write_dataframe_artifact(
            metrics_result.gene_metrics,
            output_path / "gene_metrics.csv",
            index=True,
        )

        artifacts["feature_masks"] = write_dataframe_artifact(
            metrics_result.feature_masks,
            output_path / "feature_masks.csv",
            index=True,
        )

    else:
        skipped.extend(["cell_metrics", "gene_metrics", "feature_masks"])

    if qc_config.outputs.write_mixture_table:
        if mixture is not None:
            artifacts["mito_mixture"] = write_dataframe_artifact(
                mixture.to_dataframe(),
                output_path / "qc_mito_mixture.csv",
                index=False,
            )

        else:
            skipped.append("mito_mixture")

    else:
        skipped.append("mito_mixture")

    if qc_config.outputs.write_filter_table:
        artifacts["cell_floors"] = write_dataframe_artifact(
            floors.cell_table(),
            output_path / "cell_floors.csv",
            index=True,
        )

        artifacts["gene_floors"] = write_dataframe_artifact(
            floors.gene_table(),
            output_path / "gene_floors.csv",
            index=True,
        )

    else:
        skipped.extend(["cell_floors", "gene_floors"])

    if qc_config.outputs.attrition_audit and attrition_audit is not None:
        artifacts["attrition"] = write_dataframe_artifact(
            attrition_audit.to_dataframe(),
            output_path / "qc_attrition.csv",
            index=False,
        )

    else:
        skipped.append("attrition")

    figure_source = figure_adata if figure_adata is not None else adata

    if qc_config.outputs.cell_labels:
        if figure_source is not None:
            from cellquorum.visualization.qc.panels import resolve_cell_type_keys

            keys = publication_keys or {}

            coarse_key, granular_key = resolve_cell_type_keys(figure_source.obs)
            label_columns = {
                "sample": keys.get("sample_key") or keys.get("patient_key"),
                "donor": keys.get("patient_key"),
                "condition": keys.get("condition_key"),
                "cell_type": coarse_key,
                "cell_type_granular": granular_key,
            }
            labels = pd.DataFrame(index=figure_source.obs_names.copy())
            for name, column in label_columns.items():
                if column and column in figure_source.obs.columns:
                    labels[name] = figure_source.obs[column].astype(str).to_numpy()

            if len(labels.columns):
                artifacts["cell_labels"] = write_dataframe_artifact(
                    labels,
                    output_path / "cell_labels.csv",
                    index=True,
                )
            else:
                skipped.append("cell_labels")
        else:
            skipped.append("cell_labels")

    else:
        skipped.append("cell_labels")

    if qc_config.outputs.write_report_table:
        report_table = build_qc_report_table(
            floors.cell_table(),
            groups=report_groups,
            group_name=report_group_name,
        )

        artifacts["report"] = write_dataframe_artifact(
            report_table,
            output_path / "qc_report.csv",
            index=False,
        )

    else:
        skipped.append("report")

    if qc_config.outputs.write_h5ad:
        if adata is not None:
            artifacts["qc_h5ad"] = write_h5ad_artifact(adata, output_path / "qc.h5ad")

        else:
            skipped.append("qc_h5ad")
            warnings.append(
                "QCOutputConfig.write_h5ad is true, but no AnnData object was provided."
            )

    else:
        skipped.append("qc_h5ad")

    if qc_config.outputs.write_figures:
        if figure_source is not None:
            figure_paths: list[str] = []

            try:
                from cellquorum.visualization.qc.graded import write_graded_qc_figures

                graded_paths, graded_warnings = write_graded_qc_figures(
                    figure_source.obs,
                    output_path,
                    concern_severity=qc_config.graded.concern_severity,
                    sample_column=group_key or "sample_id",
                    pair_column=(publication_keys or {}).get("patient_key") or "donor_id",
                    condition_column=(publication_keys or {}).get("condition_key") or "condition",
                    dpi=qc_config.outputs.figure_dpi,
                )
                figure_paths.extend(str(path) for path in graded_paths)
                warnings.extend(graded_warnings)
            except Exception as exc:  # pragma: no cover - defensive figure fallback
                warnings.append(
                    f"Graded QC figures could not be written: {type(exc).__name__}: {exc}"
                )

            if qc_config.outputs.overview_figures:
                try:
                    from cellquorum.visualization.qc.panels import (
                        assemble_qc_frame,
                        write_qc_panels,
                    )

                    keys = publication_keys or {}
                    panel_frame = assemble_qc_frame(
                        obs=figure_source.obs,
                        cell_metrics=_metrics_with_posterior(metrics_result.cell_metrics, mixture),
                        cell_decisions=floors.cell_table(),
                        sample_key=keys.get("sample_key") or keys.get("patient_key"),
                        donor_key=keys.get("patient_key"),
                        condition_key=keys.get("condition_key"),
                    )
                    mixture_models, mixture_ceiling = resolve_mixture_panel_inputs(mixture)
                    figure_paths.extend(
                        str(p)
                        for p in write_qc_panels(
                            panel_frame,
                            output_path,
                            case_label=keys.get("disease_label"),
                            comparisons=donor_comparisons,
                            formats=(qc_config.outputs.figure_format,),
                            dpi=qc_config.outputs.figure_dpi,
                            mixture_models=mixture_models,
                            mixture_ceiling=mixture_ceiling,
                            mixture_posterior_cutoff=qc_config.mito_mixture.posterior_cutoff,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive figure fallback
                    warnings.append(
                        "QC overview panels could not be written: " f"{type(exc).__name__}: {exc}"
                    )

            artifacts["figures"] = figure_paths
        else:
            skipped.append("figures")
            warnings.append("QCOutputConfig.write_figures is true, but no AnnData was provided.")

    else:
        skipped.append("figures")

    if qc_config.outputs.publication_tables:
        if figure_source is not None:
            try:
                from cellquorum.visualization.qc.publication_table import (
                    write_qc_publication_tables,
                )

                keys = publication_keys or {}
                artifacts["publication_tables"] = [
                    str(path)
                    for path in write_qc_publication_tables(
                        output_path,
                        cell_metrics=metrics_result.cell_metrics,
                        cell_decisions=floors.cell_table(),
                        obs=figure_source.obs,
                        sample_key=(
                            keys.get("sample_key") or keys.get("patient_key") or "sample_id"
                        ),
                        donor_key=keys.get("patient_key"),
                        condition_key=keys.get("condition_key"),
                        gene_summary={
                            "n_genes": int(len(floors.gene_keep)),
                            "n_genes_kept": int(floors.gene_keep.sum()),
                        },
                        case_label=keys.get("disease_label"),
                        project=output_path.parent.parent.name or "CellQuorum",
                        formats=("html", "tex", qc_config.outputs.figure_format),
                        dpi=qc_config.outputs.figure_dpi,
                    )
                ]
            except Exception as exc:  # pragma: no cover - defensive table fallback
                skipped.append("publication_tables")
                warnings.append(
                    f"Publication QC tables could not be written: {type(exc).__name__}: {exc}"
                )
        else:
            skipped.append("publication_tables")
            warnings.append(
                "QCOutputConfig.publication_tables is true, but no AnnData was provided, so "
                "the per-sample table has no sample labels to group by."
            )
    else:
        skipped.append("publication_tables")

    if qc_config.outputs.html_report:
        if figure_source is not None:
            try:
                from cellquorum.visualization.qc.html_report import write_qc_html_report

                keys = publication_keys or {}

                artifacts["html_report"] = write_qc_html_report(
                    output_path / "qc_report.html",
                    cell_metrics=metrics_result.cell_metrics,
                    cell_decisions=floors.cell_table(),
                    obs=figure_source.obs,
                    sample_key=keys.get("sample_key") or keys.get("patient_key") or "sample_id",
                    donor_key=keys.get("patient_key"),
                    condition_key=keys.get("condition_key"),
                    gene_summary={
                        "n_genes": int(len(floors.gene_keep)),
                        "n_genes_kept": int(floors.gene_keep.sum()),
                    },
                    project=output_path.parent.parent.name or "CellQuorum",
                    floors=qc_config.floors.model_dump(),
                    case_label=keys.get("disease_label"),
                    figures=_report_figure_order(artifacts.get("figures")),
                    notes=[warning for warning in warnings if "no distribution plotted" in warning],
                )
            except Exception as exc:  # pragma: no cover - defensive report fallback
                skipped.append("html_report")
                warnings.append(f"HTML QC report could not be written: {type(exc).__name__}: {exc}")
        else:
            skipped.append("html_report")
            warnings.append(
                "QCOutputConfig.html_report is true, but no AnnData was provided, so the "
                "per-sample attrition table has no sample labels to group by."
            )

    else:
        skipped.append("html_report")

    if qc_config.outputs.write_summary_json:
        summary_payload = build_qc_summary_payload(
            metrics_result=metrics_result,
            floors=floors,
            artifact_names=artifacts,
            skipped=skipped,
            warnings=warnings,
            summary_extra=summary_extra,
        )

        artifacts["summary"] = write_json_artifact(
            summary_payload,
            output_path / "qc_summary.json",
        )

    else:
        skipped.append("summary")

    return QCArtifactManifest(
        output_dir=output_path,
        artifacts=artifacts,
        skipped=skipped,
        warnings=warnings,
    )


def _metrics_with_posterior(
    cell_metrics: pd.DataFrame,
    mixture: MitoMixtureResult | None,
) -> pd.DataFrame:
    """Metric table with the mixture posterior attached, for the mixture panel."""
    if mixture is None or mixture.posterior.empty:
        return cell_metrics
    from cellquorum.stages.qc.mixture import MIQC_POSTERIOR_COLUMN

    merged = cell_metrics.copy()
    merged[MIQC_POSTERIOR_COLUMN] = mixture.posterior.reindex(merged.index)
    return merged


def resolve_mixture_panel_inputs(
    mixture: MitoMixtureResult | None,
) -> tuple[pd.DataFrame | None, float | None]:
    """Reduce a fitted mixture to what the mixture figure needs, or to nothing.

    Args:
        mixture: The fitted mixture, or None when the policy did not run.

    Returns:
        The fitted model table and the single mitochondrial ceiling to draw, or
        ``(None, None)`` when no model was fit. The ceiling is returned only when
        ONE applies to the whole object: a grouped fit produces one ceiling per
        group, and drawing any single one of them as a horizontal line across a
        pooled scatter would assert a bound that most of the cells were never
        judged against.
    """

    if mixture is None:
        return None, None

    models = mixture.to_dataframe()
    if not len(models):
        return None, None

    ceilings = [
        value
        for record in mixture.ceilings
        if (value := getattr(record, "ceiling", None)) is not None
    ]
    ceiling = float(ceilings[0]) if len(ceilings) == 1 else None
    return models, ceiling


def validate_qc_artifact_inputs(
    *,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    config: QCConfig,
    adata: ad.AnnData | None,
) -> None:
    """Validate inputs before writing QC artifacts.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks and counts.
        config: QC configuration.
        adata: Optional AnnData object.

    Raises:
        QCArtifactError: If inputs are invalid.
    """

    if not isinstance(metrics_result, QCMetricsResult):
        raise QCArtifactError(
            f"metrics_result must be a QCMetricsResult. Received: {type(metrics_result).__name__}."
        )

    if not isinstance(floors, FloorResult):
        raise QCArtifactError(f"floors must be a FloorResult. Received: {type(floors).__name__}.")

    if not isinstance(config, QCConfig):
        raise QCArtifactError(f"config must be a QCConfig. Received: {type(config).__name__}.")

    if adata is not None and not isinstance(adata, ad.AnnData):
        raise QCArtifactError(
            f"adata must be an AnnData object when provided. Received: {type(adata).__name__}."
        )

    validate_artifact_dataframe(metrics_result.cell_metrics, table_name="cell_metrics")
    validate_artifact_dataframe(metrics_result.gene_metrics, table_name="gene_metrics")
    validate_artifact_dataframe(metrics_result.feature_masks, table_name="feature_masks")

    validate_artifact_dataframe(floors.cell_table(), table_name="cell_floors")
    validate_artifact_dataframe(floors.gene_table(), table_name="gene_floors")


def prepare_qc_output_dir(output_dir: str | PathLike[str] | Path) -> Path:
    """Prepare a QC artifact output directory.

    Args:
        output_dir: Candidate output directory.

    Returns:
        Resolved output directory path.

    Raises:
        QCArtifactError: If output_dir is empty, points to a file, or cannot be created.
    """

    output_path = Path(output_dir)

    if str(output_path).strip() == "":
        raise QCArtifactError("QC output_dir cannot be empty.")

    if output_path.exists() and not output_path.is_dir():
        raise QCArtifactError(
            f"QC output_dir must be a directory, but path exists as a file: {output_path}."
        )

    try:
        output_path.mkdir(parents=True, exist_ok=True)

    except OSError as error:
        raise QCArtifactError(f"Failed to create QC output directory '{output_path}'.") from error

    return output_path


def validate_artifact_dataframe(table: pd.DataFrame, *, table_name: str) -> None:
    """Validate a DataFrame before artifact writing.

    Args:
        table: Candidate DataFrame.
        table_name: Human-readable table label.

    Raises:
        QCArtifactError: If the table is invalid.
    """

    if not isinstance(table, pd.DataFrame):
        raise QCArtifactError(
            f"{table_name} must be a pandas DataFrame. Received: {type(table).__name__}."
        )


def write_dataframe_artifact(
    table: pd.DataFrame,
    path: Path,
    *,
    index: bool,
) -> Path:
    """Write a DataFrame artifact as CSV.

    Args:
        table: DataFrame to write.
        path: Destination CSV path.
        index: Whether to include the DataFrame index.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    validate_artifact_dataframe(table, table_name=path.stem)

    ensure_parent_dir(path)

    temp_path = build_temp_path(path)

    try:
        table.to_csv(temp_path, index=index)

        temp_path.replace(path)

    except Exception as error:
        cleanup_temp_path(temp_path)

        raise QCArtifactError(f"Failed to write QC table artifact '{path}'.") from error

    return path


def write_json_artifact(payload: dict[str, object], path: Path) -> Path:
    """Write a JSON artifact.

    Args:
        payload: JSON-friendly payload.
        path: Destination JSON path.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    if not isinstance(payload, dict):
        raise QCArtifactError(
            f"JSON artifact payload must be a dictionary. Received: {type(payload).__name__}."
        )

    ensure_parent_dir(path)

    temp_path = build_temp_path(path)

    try:
        temp_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temp_path.replace(path)

    except Exception as error:
        cleanup_temp_path(temp_path)

        raise QCArtifactError(f"Failed to write QC JSON artifact '{path}'.") from error

    return path


def write_h5ad_artifact(adata: ad.AnnData, path: Path) -> Path:
    """Write an AnnData artifact as h5ad.

    Args:
        adata: AnnData object to write.
        path: Destination h5ad path.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    if not isinstance(adata, ad.AnnData):
        raise QCArtifactError(
            f"write_h5ad_artifact expected an AnnData object. Received: {type(adata).__name__}."
        )

    ensure_parent_dir(path)

    from cellquorum.core.h5ad_io import H5adWriteError, write_h5ad

    try:
        write_h5ad(adata, path)
    except H5adWriteError as error:
        raise QCArtifactError(f"Failed to write QC AnnData artifact '{path}'.") from error

    return path


def build_qc_summary_payload(
    *,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    artifact_names: Mapping[str, Path | list[str]],
    skipped: list[str],
    warnings: list[str],
    summary_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the QC summary JSON payload.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks, reasons and counts.
        artifact_names: Written artifact paths by label. A value may be a list, because two
            entries are groups of files written in one step rather than single paths.
        skipped: Skipped artifact labels.
        warnings: Artifact warnings.
        summary_extra: Optional extra summary values.

    Returns:
        JSON-friendly QC summary payload.
    """

    payload: dict[str, object] = {
        "metrics": metrics_result.to_summary_dict(),
        "floors": floors.to_summary_dict(),
        "artifacts": {
            artifact_name: (
                [str(p) for p in artifact_path]
                if isinstance(artifact_path, list)
                else str(artifact_path)
            )
            for artifact_name, artifact_path in artifact_names.items()
        },
        "skipped": list(skipped),
        "warnings": list(warnings),
    }

    if summary_extra is not None:
        if not isinstance(summary_extra, dict):
            raise QCArtifactError(
                "summary_extra must be a dictionary when provided. "
                f"Received: {type(summary_extra).__name__}."
            )

        payload["extra"] = summary_extra

    return payload


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory for an artifact path exists.

    Args:
        path: Artifact destination path.

    Raises:
        QCArtifactError: If parent directory creation fails.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

    except OSError as error:
        raise QCArtifactError(
            f"Failed to create parent directory for QC artifact '{path}'."
        ) from error


def build_temp_path(path: Path) -> Path:
    """Build a temporary path next to a destination artifact.

    Args:
        path: Destination path.

    Returns:
        Temporary path used for atomic writing.
    """

    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def cleanup_temp_path(path: Path) -> None:
    """Remove a temporary artifact path if present.

    Args:
        path: Temporary path to remove.
    """

    if path.exists():
        path.unlink()


def to_jsonable(value: object) -> object:
    """Convert common scientific Python values into JSON-friendly objects.

    Args:
        value: Candidate value.

    Returns:
        JSON-compatible representation.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())

    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))

    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return value.item()

        except (AttributeError, ValueError, TypeError):
            pass

    return value


__all__ = [
    "QCArtifactError",
    "QCArtifactManifest",
    "build_qc_summary_payload",
    "build_temp_path",
    "cleanup_temp_path",
    "ensure_parent_dir",
    "prepare_qc_output_dir",
    "resolve_mixture_panel_inputs",
    "to_jsonable",
    "validate_artifact_dataframe",
    "validate_qc_artifact_inputs",
    "write_dataframe_artifact",
    "write_h5ad_artifact",
    "write_json_artifact",
    "write_qc_artifacts",
]
