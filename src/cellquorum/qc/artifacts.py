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

# Import QC configuration.
from cellquorum.qc.config import QCConfig

# Import QC decision result container.
from cellquorum.qc.decisions import QCDecisionResult

# Import QC metric result container.
from cellquorum.qc.metrics import QCMetricsResult

# Import QC threshold result container.
from cellquorum.qc.thresholds import QCThresholdResult


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
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    config: QCConfig | None = None,
    adata: ad.AnnData | None = None,
    summary_extra: dict[str, object] | None = None,
    group_key: str | None = None,
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
        threshold_result: Constructed QC thresholds.
        decision_result: Applied QC decisions.
        config: Optional QC configuration. Defaults to QCConfig().
        adata: Optional AnnData object to write as qc.h5ad when enabled.
        summary_extra: Optional extra JSON-friendly values to include in the
            summary artifact.
        group_key: Optional obs column name for grouping QC figures by
            condition, donor, or sample.

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
        threshold_result=threshold_result,
        decision_result=decision_result,
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
        # Write cell metrics.
        artifacts["cell_metrics"] = write_dataframe_artifact(
            metrics_result.cell_metrics,
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

    # Write threshold table when enabled.
    if qc_config.outputs.write_threshold_table:
        # Write threshold records.
        artifacts["thresholds"] = write_dataframe_artifact(
            threshold_result.to_dataframe(),
            output_path / "thresholds.csv",
            index=False,
        )

    # Record skipped threshold table.
    else:
        # Store threshold table skip.
        skipped.append("thresholds")

    # Write decision tables when enabled.
    if qc_config.outputs.write_filter_table:
        # Write cell decisions.
        artifacts["cell_decisions"] = write_dataframe_artifact(
            decision_result.cell_decisions,
            output_path / "cell_decisions.csv",
            index=True,
        )

        # Write gene decisions.
        artifacts["gene_decisions"] = write_dataframe_artifact(
            decision_result.gene_decisions,
            output_path / "gene_decisions.csv",
            index=True,
        )

    # Record skipped decision tables.
    else:
        # Store decision table skips.
        skipped.extend(["cell_decisions", "gene_decisions"])

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

    # Write QC figures when enabled and AnnData is available.
    if qc_config.outputs.write_figures:
        if adata is not None:
            # Import figure writer locally to avoid circular imports.
            from cellquorum.qc.visualization import write_qc_figures

            # Generate and write QC figures.
            fig_result = write_qc_figures(
                adata,
                output_path,
                dpi=qc_config.outputs.figure_dpi,
                figure_format=qc_config.outputs.figure_format,
                overwrite=True,
                thresholds=threshold_result,
                group_key=group_key,
            )

            # Record figure paths in the artifact manifest.
            artifacts["figures"] = [str(p) for p in fig_result.figure_paths]

            # Propagate figure-generation warnings.
            warnings.extend(fig_result.warnings)
        else:
            # Store figure skip when AnnData is absent.
            skipped.append("figures")
            warnings.append("QCOutputConfig.write_figures is true, but no AnnData was " "provided.")

    # Record figure skip when disabled.
    else:
        # Store figure skip.
        skipped.append("figures")

    # Write summary JSON after other artifacts so it can include manifest metadata.
    if qc_config.outputs.write_summary_json:
        # Build summary payload.
        summary_payload = build_qc_summary_payload(
            metrics_result=metrics_result,
            threshold_result=threshold_result,
            decision_result=decision_result,
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


def validate_qc_artifact_inputs(
    *,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    config: QCConfig,
    adata: ad.AnnData | None,
) -> None:
    """
    Validate inputs before writing QC artifacts.

    Args:
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
        config: QC configuration.
        adata: Optional AnnData object.

    Raises:
        QCArtifactError: If inputs are invalid.
    """

    # Validate metrics result type.
    if not isinstance(metrics_result, QCMetricsResult):
        raise QCArtifactError(
            "metrics_result must be a QCMetricsResult. "
            f"Received: {type(metrics_result).__name__}."
        )

    # Validate threshold result type.
    if not isinstance(threshold_result, QCThresholdResult):
        raise QCArtifactError(
            "threshold_result must be a QCThresholdResult. "
            f"Received: {type(threshold_result).__name__}."
        )

    # Validate decision result type.
    if not isinstance(decision_result, QCDecisionResult):
        raise QCArtifactError(
            "decision_result must be a QCDecisionResult. "
            f"Received: {type(decision_result).__name__}."
        )

    # Validate config type.
    if not isinstance(config, QCConfig):
        raise QCArtifactError("config must be a QCConfig. " f"Received: {type(config).__name__}.")

    # Validate optional AnnData type.
    if adata is not None and not isinstance(adata, ad.AnnData):
        raise QCArtifactError(
            "adata must be an AnnData object when provided. " f"Received: {type(adata).__name__}."
        )

    # Validate metric result tables.
    validate_artifact_dataframe(metrics_result.cell_metrics, table_name="cell_metrics")
    validate_artifact_dataframe(metrics_result.gene_metrics, table_name="gene_metrics")
    validate_artifact_dataframe(metrics_result.feature_masks, table_name="feature_masks")

    # Validate decision result tables.
    validate_artifact_dataframe(decision_result.cell_decisions, table_name="cell_decisions")
    validate_artifact_dataframe(decision_result.gene_decisions, table_name="gene_decisions")


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
            f"{table_name} must be a pandas DataFrame. " f"Received: {type(table).__name__}."
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
            "JSON artifact payload must be a dictionary. " f"Received: {type(payload).__name__}."
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
            "write_h5ad_artifact expected an AnnData object. " f"Received: {type(adata).__name__}."
        )

    # Ensure the destination parent directory exists.
    ensure_parent_dir(path)

    # Build a temporary path for atomic replacement.
    temp_path = build_temp_path(path)

    # Try writing the h5ad artifact.
    try:
        # Write AnnData to a temporary h5ad file.
        adata.write_h5ad(temp_path)

        # Atomically replace the target path.
        temp_path.replace(path)

    # Convert AnnData or filesystem errors into QC artifact errors.
    except Exception as error:
        # Remove the temporary file if it exists.
        cleanup_temp_path(temp_path)

        # Raise a contextual artifact error.
        raise QCArtifactError(f"Failed to write QC AnnData artifact '{path}'.") from error

    # Return the written path.
    return path


def build_qc_summary_payload(
    *,
    metrics_result: QCMetricsResult,
    threshold_result: QCThresholdResult,
    decision_result: QCDecisionResult,
    artifact_names: dict[str, Path],
    skipped: list[str],
    warnings: list[str],
    summary_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the QC summary JSON payload.

    Args:
        metrics_result: QC metrics result.
        threshold_result: QC threshold result.
        decision_result: QC decision result.
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
        "thresholds": threshold_result.to_summary_dict(),
        "decisions": decision_result.to_summary_dict(),
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
    "to_jsonable",
    "validate_artifact_dataframe",
    "validate_qc_artifact_inputs",
    "write_dataframe_artifact",
    "write_h5ad_artifact",
    "write_json_artifact",
    "write_qc_artifacts",
]
