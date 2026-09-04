"""QC artifact writing utilities for CellQuorum."""

from __future__ import annotations

# Import JSON helpers for summary artifact writing.
import json

# Import dataclass helpers for structured artifact manifests.
from dataclasses import dataclass, field

# Import PathLike for flexible filesystem path input typing.
from os import PathLike

# Import Path for filesystem-safe artifact writing.
from pathlib import Path

# Import AnnData for optional QC object writing.
import anndata as ad

# Import pandas for table artifact validation and writing.
import pandas as pd

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import the differential-attrition audit container.
from cellquorum.stages.qc.attrition import AttritionAudit

# Import QC configuration.
from cellquorum.stages.qc.config import QCConfig

# Import the floor result and the report-table builder.
from cellquorum.stages.qc.floors import FloorResult, build_qc_report_table
from cellquorum.stages.qc.metrics import QCMetricsResult
from cellquorum.stages.qc.mixture import MitoMixtureResult

# Import QC threshold result container.


class QCArtifactError(CellQuorumDataError):
    """
    Report QC artifact writing failures.

    QC artifacts are the durable outputs of the QC module. Errors here should be
    explicit because partial, missing, or malformed outputs make downstream
    provenance and reproducibility difficult.
    """


@dataclass(frozen=True)
class QCArtifactManifest:
    """
    Store a manifest of QC artifacts written to disk.

    Args:
        output_dir: Directory where QC artifacts were written.
        artifacts: Mapping from artifact label to filesystem path.
        skipped: Artifact labels skipped because config disabled them or required
            inputs were missing.
        warnings: Non-fatal artifact writing warnings.
    """

    # Store the QC artifact output directory.
    output_dir: Path

    # Store written artifact paths by stable artifact label.
    artifacts: dict[str, Path | list[str]] = field(default_factory=dict)

    # Store skipped artifact labels.
    skipped: list[str] = field(default_factory=list)

    # Store non-fatal artifact writing warnings.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the artifact manifest into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of written and skipped artifacts.
        """

        # Return a JSON-friendly manifest payload.
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
        """
        Return the path for a written artifact.

        Args:
            artifact_name: Stable artifact label.

        Returns:
            Path to the requested artifact.

        Raises:
            QCArtifactError: If the artifact was not written.
        """

        # Raise clearly when the requested artifact was not written.
        if artifact_name not in self.artifacts:
            raise QCArtifactError(
                f"QC artifact '{artifact_name}' was not written. "
                f"Available artifacts: {', '.join(self.artifacts) or '<none>'}."
            )

        # Return the artifact path.
        return self.artifacts[artifact_name]


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
) -> QCArtifactManifest:
    """
    Write QC module artifacts to disk.

    This function writes machine-readable tables and summaries produced by the
    QC module. It respects QCOutputConfig flags, creates the output directory,
    writes CSV/JSON artifacts atomically, optionally writes a QC AnnData object,
    and returns a structured artifact manifest.

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

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate artifact writer inputs.
    validate_qc_artifact_inputs(
        metrics_result=metrics_result,
        floors=floors,
        config=qc_config,
        adata=adata,
    )

    # Prepare the output directory.
    output_path = prepare_qc_output_dir(output_dir)

    # Initialize written artifact mapping.
    artifacts: dict[str, Path] = {}

    # Initialize skipped artifact labels.
    skipped: list[str] = []

    # Initialize artifact warnings.
    warnings: list[str] = []

    # Write metric tables when enabled.
    if qc_config.outputs.write_metrics_table:
        # Write cell metrics, including any column a model-based rule computed
        # while building its threshold.
        #
        # Those columns are what the rule actually thresholded on, so a table
        # without them cannot be used to reproduce the rule's own decision. The
        # decision step attaches them to an internal copy; attaching them here
        # too is what puts them on disk.
        artifacts["cell_metrics"] = write_dataframe_artifact(
            # With the mixture posterior attached: it is a per-cell measurement, so it belongs
            # in the metric table rather than only inside the graded evidence columns.
            _metrics_with_posterior(metrics_result.cell_metrics, mixture),
            output_path / "cell_metrics.csv",
            index=True,
        )

        # Write gene metrics.
        artifacts["gene_metrics"] = write_dataframe_artifact(
            metrics_result.gene_metrics,
            output_path / "gene_metrics.csv",
            index=True,
        )

        # Write feature masks.
        artifacts["feature_masks"] = write_dataframe_artifact(
            metrics_result.feature_masks,
            output_path / "feature_masks.csv",
            index=True,
        )

    # Record skipped metric tables.
    else:
        # Store metric table skips.
        skipped.extend(["cell_metrics", "gene_metrics", "feature_masks"])

    # thresholds.csv is gone with the threshold path. What replaced it is not another table of
    # bounds: the graded policy is two severity bars plus a concordance requirement, recorded in
    # provenance and stated in the figure notes, and the per-cell evidence behind it is on obs.
    if qc_config.outputs.write_mixture_table:
        # The fitted mitochondrial mixture, when that policy ran: both component fits, their
        # variances, and how many cells each post-processing rule moved.
        #
        # It used to ride the threshold flag on the grounds that it was the derivation of the
        # mitochondrial threshold. There is no such threshold now — the posterior feeds the graded
        # metabolic axis — so the model has a flag of its own, which is what it always was.
        if mixture is not None:
            artifacts["mito_mixture"] = write_dataframe_artifact(
                mixture.to_dataframe(),
                output_path / "qc_mito_mixture.csv",
                index=False,
            )

        # Record the skipped mixture table when the policy did not run.
        else:
            skipped.append("mito_mixture")

    else:
        skipped.append("mito_mixture")

    # Write decision tables when enabled.
    if qc_config.outputs.write_filter_table:
        # Write cell decisions.
        artifacts["cell_floors"] = write_dataframe_artifact(
            floors.cell_table(),
            output_path / "cell_floors.csv",
            index=True,
        )

        # Write gene decisions.
        artifacts["gene_floors"] = write_dataframe_artifact(
            floors.gene_table(),
            output_path / "gene_floors.csv",
            index=True,
        )

    # Record skipped decision tables.
    else:
        # Store decision table skips.
        skipped.extend(["cell_floors", "gene_floors"])

    # Write the differential-attrition audit when one was run. The table carries
    # its skipped tests too, so a reader can tell "checked and clean" from
    # "never checked" -- which is the difference between a methods sentence that
    # is true and one that is a guess.
    if qc_config.outputs.attrition_audit and attrition_audit is not None:
        artifacts["attrition"] = write_dataframe_artifact(
            attrition_audit.to_dataframe(),
            output_path / "qc_attrition.csv",
            index=False,
        )

    # Record the skipped attrition table.
    else:
        # Store the attrition table skip.
        skipped.append("attrition")

    # Resolve the object that anything describing the FILTER must read from.
    # Under mode="filter" ``adata`` has already lost the failing cells, so a
    # keep/fail figure or an attrition table drawn from it reports a 100% pass
    # rate however many cells were dropped.
    figure_source = figure_adata if figure_adata is not None else adata

    # Write the grouping labels of every cell that ENTERED QC. The h5ad written
    # above has already lost the removed cells under mode="filter", so this table
    # is the only place a later re-render can learn what a removed cell was — and
    # without it a by-cell-type attrition figure reports 0% removed for every
    # type, which is the most convincing wrong figure this stage can produce.
    if qc_config.outputs.cell_labels:
        if figure_source is not None:
            from cellquorum.visualization.qc.panels import resolve_cell_type_keys

            keys = publication_keys or {}
            # Cell-type columns are auto-detected rather than passed through
            # ``publication_keys``: that dict is splatted into the legacy figure
            # writer, which would reject an argument it does not declare.
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

            # Skipped, not warned: an object with no cohort or cell-type columns
            # has no labels to lose, and the missing object or missing cohort keys
            # are already reported where they are resolved. The skip list is the
            # manifest's own channel for "flag on, nothing to write".
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

    # Record skipped label table when disabled.
    else:
        skipped.append("cell_labels")

    # Write the per-group QC report table when enabled.
    if qc_config.outputs.write_report_table:
        # Build the report table from the (unfiltered) cell decisions.
        report_table = build_qc_report_table(
            floors.cell_table(),
            groups=report_groups,
            group_name=report_group_name,
        )

        # Write the QC report table.
        artifacts["report"] = write_dataframe_artifact(
            report_table,
            output_path / "qc_report.csv",
            index=False,
        )

    # Record skipped report table.
    else:
        # Store report table skip.
        skipped.append("report")

    # Write optional QC AnnData object when enabled.
    if qc_config.outputs.write_h5ad:
        # Write AnnData when supplied.
        if adata is not None:
            artifacts["qc_h5ad"] = write_h5ad_artifact(adata, output_path / "qc.h5ad")

        # Record a warning and skip when AnnData is unavailable.
        else:
            skipped.append("qc_h5ad")
            warnings.append(
                "QCOutputConfig.write_h5ad is true, but no AnnData object was provided."
            )

    # Record skipped QC AnnData object when disabled.
    else:
        # Store h5ad skip.
        skipped.append("qc_h5ad")

    # Write the single-file HTML QC report when enabled. It reads the same tables
    # already written above, so it never disagrees with them.
    if qc_config.outputs.html_report:
        if figure_source is not None:
            try:
                from cellquorum.visualization.qc.html_report import write_qc_html_report

                keys = publication_keys or {}
                # A Path, like every other table artifact, so callers can treat the
                # manifest uniformly.
                artifacts["html_report"] = write_qc_html_report(
                    output_path / "qc_report.html",
                    cell_metrics=metrics_result.cell_metrics,
                    cell_decisions=floors.cell_table(),
                    obs=figure_source.obs,
                    # Fall back through sample -> donor: a cohort without a
                    # sample column still gets a meaningful attrition table.
                    sample_key=keys.get("sample_key") or keys.get("patient_key") or "sample_id",
                    donor_key=keys.get("patient_key"),
                    condition_key=keys.get("condition_key"),
                    gene_summary={
                        "n_genes": int(len(floors.gene_keep)),
                        "n_genes_kept": int(floors.gene_keep.sum()),
                    },
                    project=output_path.parent.parent.name or "CellQuorum",
                    mode=str(qc_config.mode),
                    case_label=keys.get("disease_label"),
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

    # Record skipped HTML report when disabled.
    else:
        skipped.append("html_report")

    # Write the typeset publication tables when enabled. Same numbers as the CSVs,
    # set as a manuscript Table 1 rather than dumped as a grid.
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

    # Write QC figures when enabled and AnnData is available. Figures render from
    # the pre-filter object when one is supplied: the filtered object cannot show
    # what QC removed.
    if qc_config.outputs.write_figures:
        if figure_source is not None:
            # Collect paths across the three independent figure writers below.
            # None of them is a prerequisite for another: a run can emit the
            # overview panels without the per-metric audit plots, which is the
            # default, because sixteen distribution plots do not answer "what did
            # QC remove" and the six panels do.
            figure_paths: list[str] = []

            # Graded QC panels: always written when graded columns exist. Not behind a flag —
            # they are the only figures that can describe the graded model, and a run whose
            # verdict nobody can see is the failure this whole area exists to fix.
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

            # Write the figure-ready QC panel set. It answers "what did QC do to this
            # cohort", which is the question a reviewer asks and the one per-metric
            # histograms cannot address.
            #
            # The two v1 writers that used to sit here — `visualization.qc.diagnostics`
            # (per-metric audit plots) and `visualization.qc.publication` (the legacy
            # mast-cell/LE-KC panels) — were deleted with the threshold path. Both keyed
            # on `cellquorum_qc_keep`, a verdict that no longer exists, so neither could
            # render a graded run. `write_graded_qc_figures` above replaces them.
            if qc_config.outputs.overview_figures:
                try:
                    from cellquorum.visualization.qc.panels import (
                        assemble_qc_frame,
                        write_qc_panels,
                    )

                    keys = publication_keys or {}
                    panel_frame = assemble_qc_frame(
                        obs=figure_source.obs,
                        # The mixture panel colours cells by the posterior the model assigned
                        # them, which now comes from the mixture directly rather than from
                        # threshold-derived columns merged into the metric table.
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
            # Store figure skip when AnnData is absent.
            skipped.append("figures")
            warnings.append("QCOutputConfig.write_figures is true, but no AnnData was provided.")

    # Record figure skip when disabled.
    else:
        # Store figure skip.
        skipped.append("figures")

    # Write summary JSON after other artifacts so it can include manifest metadata.
    if qc_config.outputs.write_summary_json:
        # Build summary payload.
        summary_payload = build_qc_summary_payload(
            metrics_result=metrics_result,
            floors=floors,
            artifact_names=artifacts,
            skipped=skipped,
            warnings=warnings,
            summary_extra=summary_extra,
        )

        # Write summary JSON.
        artifacts["summary"] = write_json_artifact(
            summary_payload,
            output_path / "qc_summary.json",
        )

    # Record skipped summary JSON.
    else:
        # Store summary skip.
        skipped.append("summary")

    # Return the artifact manifest.
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
    """Metric table with the mixture posterior attached, for the mixture panel.

    Without the posterior the panel still renders, in two flat colours that say nothing about the
    model — so it is attached here rather than left to the caller to remember.
    """
    if mixture is None or mixture.posterior.empty:
        return cell_metrics
    from cellquorum.stages.qc.mixture import MIQC_POSTERIOR_COLUMN

    merged = cell_metrics.copy()
    merged[MIQC_POSTERIOR_COLUMN] = mixture.posterior.reindex(merged.index)
    return merged


def resolve_mixture_panel_inputs(
    mixture: MitoMixtureResult | None,
) -> tuple[pd.DataFrame | None, float | None]:
    """
    Reduce a fitted mixture to what the mixture figure needs, or to nothing.

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

    # Return nothing when the mixture policy did not run.
    if mixture is None:
        return None, None

    # Return nothing when the policy ran but fit no group, which is what happens
    # on an object too small or too uniform for the mixture to be identifiable.
    models = mixture.to_dataframe()
    if not len(models):
        return None, None

    # Take the ceiling only when it is unambiguous.
    # Bound in the comprehension rather than filtered via getattr: filtering on an attribute
    # lookup cannot narrow the element type, so the list stayed `float | None` and the float()
    # below was unchecked.
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
    """
    Validate inputs before writing QC artifacts.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks and counts.
        config: QC configuration.
        adata: Optional AnnData object.

    Raises:
        QCArtifactError: If inputs are invalid.
    """

    # Validate metrics result type.
    if not isinstance(metrics_result, QCMetricsResult):
        raise QCArtifactError(
            f"metrics_result must be a QCMetricsResult. Received: {type(metrics_result).__name__}."
        )

    # Validate threshold result type.

    # Validate decision result type.
    if not isinstance(floors, FloorResult):
        raise QCArtifactError(f"floors must be a FloorResult. Received: {type(floors).__name__}.")

    # Validate config type.
    if not isinstance(config, QCConfig):
        raise QCArtifactError(f"config must be a QCConfig. Received: {type(config).__name__}.")

    # Validate optional AnnData type.
    if adata is not None and not isinstance(adata, ad.AnnData):
        raise QCArtifactError(
            f"adata must be an AnnData object when provided. Received: {type(adata).__name__}."
        )

    # Validate metric result tables.
    validate_artifact_dataframe(metrics_result.cell_metrics, table_name="cell_metrics")
    validate_artifact_dataframe(metrics_result.gene_metrics, table_name="gene_metrics")
    validate_artifact_dataframe(metrics_result.feature_masks, table_name="feature_masks")

    # Validate decision result tables.
    validate_artifact_dataframe(floors.cell_table(), table_name="cell_floors")
    validate_artifact_dataframe(floors.gene_table(), table_name="gene_floors")


def prepare_qc_output_dir(output_dir: str | PathLike[str] | Path) -> Path:
    """
    Prepare a QC artifact output directory.

    Args:
        output_dir: Candidate output directory.

    Returns:
        Resolved output directory path.

    Raises:
        QCArtifactError: If output_dir is empty, points to a file, or cannot be created.
    """

    # Convert the output directory to a Path.
    output_path = Path(output_dir)

    # Reject empty path strings.
    if str(output_path).strip() == "":
        raise QCArtifactError("QC output_dir cannot be empty.")

    # Reject an existing regular file.
    if output_path.exists() and not output_path.is_dir():
        raise QCArtifactError(
            f"QC output_dir must be a directory, but path exists as a file: {output_path}."
        )

    # Create the output directory if needed.
    try:
        # Create parent directories as needed.
        output_path.mkdir(parents=True, exist_ok=True)

    # Convert filesystem errors into QC artifact errors.
    except OSError as error:
        raise QCArtifactError(f"Failed to create QC output directory '{output_path}'.") from error

    # Return the prepared output directory.
    return output_path


def validate_artifact_dataframe(table: pd.DataFrame, *, table_name: str) -> None:
    """
    Validate a DataFrame before artifact writing.

    Args:
        table: Candidate DataFrame.
        table_name: Human-readable table label.

    Raises:
        QCArtifactError: If the table is invalid.
    """

    # Validate DataFrame type.
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
    """
    Write a DataFrame artifact as CSV.

    Args:
        table: DataFrame to write.
        path: Destination CSV path.
        index: Whether to include the DataFrame index.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    # Validate the table.
    validate_artifact_dataframe(table, table_name=path.stem)

    # Ensure the destination parent directory exists.
    ensure_parent_dir(path)

    # Build a temporary path for atomic replacement.
    temp_path = build_temp_path(path)

    # Try writing the CSV artifact.
    try:
        # Write the DataFrame to a temporary CSV file.
        table.to_csv(temp_path, index=index)

        # Atomically replace the target path.
        temp_path.replace(path)

    # Convert filesystem or pandas errors into QC artifact errors.
    except Exception as error:
        # Remove the temporary file if it exists.
        cleanup_temp_path(temp_path)

        # Raise a contextual artifact error.
        raise QCArtifactError(f"Failed to write QC table artifact '{path}'.") from error

    # Return the written path.
    return path


def write_json_artifact(payload: dict[str, object], path: Path) -> Path:
    """
    Write a JSON artifact.

    Args:
        payload: JSON-friendly payload.
        path: Destination JSON path.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    # Validate the payload type.
    if not isinstance(payload, dict):
        raise QCArtifactError(
            f"JSON artifact payload must be a dictionary. Received: {type(payload).__name__}."
        )

    # Ensure the destination parent directory exists.
    ensure_parent_dir(path)

    # Build a temporary path for atomic replacement.
    temp_path = build_temp_path(path)

    # Try writing the JSON artifact.
    try:
        # Write formatted JSON to the temporary path.
        temp_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Atomically replace the target path.
        temp_path.replace(path)

    # Convert filesystem or JSON errors into QC artifact errors.
    except Exception as error:
        # Remove the temporary file if it exists.
        cleanup_temp_path(temp_path)

        # Raise a contextual artifact error.
        raise QCArtifactError(f"Failed to write QC JSON artifact '{path}'.") from error

    # Return the written path.
    return path


def write_h5ad_artifact(adata: ad.AnnData, path: Path) -> Path:
    """
    Write an AnnData artifact as h5ad.

    Args:
        adata: AnnData object to write.
        path: Destination h5ad path.

    Returns:
        Written artifact path.

    Raises:
        QCArtifactError: If writing fails.
    """

    # Validate AnnData input.
    if not isinstance(adata, ad.AnnData):
        raise QCArtifactError(
            f"write_h5ad_artifact expected an AnnData object. Received: {type(adata).__name__}."
        )

    # Ensure the destination parent directory exists.
    ensure_parent_dir(path)

    # Through the shared writer: it opts in to nullable strings, writes atomically,
    # and coerces the handful of things h5py refuses (see cellquorum.core.h5ad_io).
    from cellquorum.core.h5ad_io import H5adWriteError, write_h5ad

    try:
        write_h5ad(adata, path)
    except H5adWriteError as error:
        raise QCArtifactError(f"Failed to write QC AnnData artifact '{path}'.") from error

    # Return the written path.
    return path


def build_qc_summary_payload(
    *,
    metrics_result: QCMetricsResult,
    floors: FloorResult,
    artifact_names: dict[str, Path],
    skipped: list[str],
    warnings: list[str],
    summary_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the QC summary JSON payload.

    Args:
        metrics_result: QC metrics result.
        floors: Floor masks, reasons and counts.
        artifact_names: Written artifact paths by label.
        skipped: Skipped artifact labels.
        warnings: Artifact warnings.
        summary_extra: Optional extra summary values.

    Returns:
        JSON-friendly QC summary payload.
    """

    # Build the base summary payload.
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

    # Add optional extra summary fields.
    if summary_extra is not None:
        # Validate extra summary type.
        if not isinstance(summary_extra, dict):
            raise QCArtifactError(
                "summary_extra must be a dictionary when provided. "
                f"Received: {type(summary_extra).__name__}."
            )

        # Store extra values under a namespaced key.
        payload["extra"] = summary_extra

    # Return the summary payload.
    return payload


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory for an artifact path exists.

    Args:
        path: Artifact destination path.

    Raises:
        QCArtifactError: If parent directory creation fails.
    """

    # Try creating the parent directory.
    try:
        # Create all parent directories as needed.
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert filesystem errors into artifact errors.
    except OSError as error:
        raise QCArtifactError(
            f"Failed to create parent directory for QC artifact '{path}'."
        ) from error


def build_temp_path(path: Path) -> Path:
    """
    Build a temporary path next to a destination artifact.

    Args:
        path: Destination path.

    Returns:
        Temporary path used for atomic writing.
    """

    # Return a temporary sibling path.
    return path.with_name(f".{path.name}.tmp")


def cleanup_temp_path(path: Path) -> None:
    """
    Remove a temporary artifact path if present.

    Args:
        path: Temporary path to remove.
    """

    # Remove only when the temp path exists.
    if path.exists():
        # Remove the temp path.
        path.unlink()


def to_jsonable(value: object) -> object:
    """
    Convert common scientific Python values into JSON-friendly objects.

    Args:
        value: Candidate value.

    Returns:
        JSON-compatible representation.
    """

    # Convert pathlib paths to strings.
    if isinstance(value, Path):
        return str(value)

    # Convert dictionaries recursively.
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    # Convert lists recursively.
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    # Convert tuples recursively.
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    # Convert pandas Series recursively.
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())

    # Convert pandas DataFrames to row records.
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))

    # Convert NumPy scalar-like objects when available without importing NumPy directly here.
    if hasattr(value, "item") and not isinstance(value, str):
        # Try scalar conversion.
        try:
            # Return the converted scalar.
            return value.item()

        # Fall through when item() is not scalar-like.
        except (AttributeError, ValueError, TypeError):
            pass

    # Return the original value for normal JSON-compatible objects.
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
