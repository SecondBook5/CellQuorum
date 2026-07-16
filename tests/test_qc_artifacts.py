"""Tests for CellQuorum QC artifact writing utilities."""

from __future__ import annotations

# Import JSON helpers for reading summary artifacts.
import json

# Import Path for filesystem assertions.
from pathlib import Path

# Import AnnData for h5ad artifact tests.
import anndata as ad

# Import NumPy for deterministic test matrices and JSON conversion tests.
import numpy as np

# Import pandas for artifact DataFrame construction and CSV assertions.
import pandas as pd

# Import pandas testing helpers.
import pandas.testing as pdt

# Import pytest for exception assertions.
import pytest

# Import QC artifact utilities under test.
from cellquorum.qc.artifacts import (
    QCArtifactError,
    QCArtifactManifest,
    build_qc_summary_payload,
    build_temp_path,
    cleanup_temp_path,
    ensure_parent_dir,
    prepare_qc_output_dir,
    to_jsonable,
    validate_artifact_dataframe,
    validate_qc_artifact_inputs,
    write_dataframe_artifact,
    write_h5ad_artifact,
    write_json_artifact,
    write_qc_artifacts,
)

# Import QC configuration.
from cellquorum.qc.config import QCConfig

# Import QC decision result container.
from cellquorum.qc.decisions import QCDecisionResult

# Import QC metrics result container.
from cellquorum.qc.metrics import QCMetricsResult

# Import QC threshold records and result container.
from cellquorum.qc.thresholds import QCThreshold, QCThresholdResult


def make_metrics_result() -> QCMetricsResult:
    """
    Build a small QC metrics result for artifact tests.

    Returns:
        QCMetricsResult with cell metrics, gene metrics, feature masks, and summary.
    """

    # Build cell-level QC metrics.
    cell_metrics = pd.DataFrame(
        {
            "total_counts": [10.0, 20.0],
            "n_genes_by_counts": [2, 3],
            "pct_counts_mito": [5.0, 10.0],
        },
        index=["cell_1", "cell_2"],
    )

    # Build gene-level QC metrics.
    gene_metrics = pd.DataFrame(
        {
            "n_cells_by_counts": [1, 2],
            "total_counts": [5.0, 25.0],
        },
        index=["gene_1", "gene_2"],
    )

    # Build feature-family masks.
    feature_masks = pd.DataFrame(
        {
            "cellquorum_is_mito": [True, False],
            "cellquorum_is_ribo": [False, True],
            "cellquorum_is_hemoglobin": [False, False],
            "cellquorum_is_custom_exclude": [False, False],
        },
        index=["gene_1", "gene_2"],
    )

    # Return the structured metrics result.
    return QCMetricsResult(
        cell_metrics=cell_metrics,
        gene_metrics=gene_metrics,
        feature_masks=feature_masks,
        summary={
            "matrix_source": "X",
            "n_cells": 2,
            "n_genes": 2,
            "total_counts_sum": 30.0,
        },
        warnings=["metrics warning"],
    )


def make_threshold_result() -> QCThresholdResult:
    """
    Build a small QC threshold result for artifact tests.

    Returns:
        QCThresholdResult containing one cell threshold and one warning.
    """

    # Build a representative threshold.
    threshold = QCThreshold(
        axis="cell",
        metric="pct_counts_mito",
        rule_name="fixed_max_mito_percent",
        lower=None,
        upper=8.0,
        source="fixed",
    )

    # Return the structured threshold result.
    return QCThresholdResult(
        thresholds=[threshold],
        warnings=["threshold warning"],
    )


def make_decision_result() -> QCDecisionResult:
    """
    Build a small QC decision result for artifact tests.

    Returns:
        QCDecisionResult with cell and gene decision tables.
    """

    # Build cell-level decisions.
    cell_decisions = pd.DataFrame(
        {
            "keep": [True, False],
            "fail_any_qc": [False, True],
            "failed_rules": ["", "fixed_max_mito_percent"],
            "fixed_max_mito_percent": [False, True],
        },
        index=["cell_1", "cell_2"],
    )

    # Build gene-level decisions.
    gene_decisions = pd.DataFrame(
        {
            "keep": [False, True],
            "fail_any_qc": [True, False],
            "failed_rules": ["fixed_min_cells_per_gene", ""],
            "fixed_min_cells_per_gene": [True, False],
        },
        index=["gene_1", "gene_2"],
    )

    # Return the structured decision result.
    return QCDecisionResult(
        cell_decisions=cell_decisions,
        gene_decisions=gene_decisions,
        summary={
            "n_cells": 2,
            "n_cells_kept": 1,
            "n_cells_failed": 1,
            "n_genes": 2,
            "n_genes_kept": 1,
            "n_genes_failed": 1,
        },
        warnings=["decision warning"],
    )


def make_test_adata(obs_extra: dict[str, list] | None = None) -> ad.AnnData:
    """
    Build a tiny AnnData object for h5ad artifact tests.

    Args:
        obs_extra: Optional extra obs columns as {col_name: [val1, val2]}.

    Returns:
        Small AnnData object.
    """

    # Build a deterministic count matrix.
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])

    # Build observation metadata.
    obs_data = obs_extra.copy() if obs_extra else {}
    obs = pd.DataFrame(obs_data, index=["cell_1", "cell_2"])

    # Build variable metadata.
    var = pd.DataFrame(index=["gene_1", "gene_2"])

    # Return the AnnData object.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_qc_artifact_manifest_serializes_and_retrieves_paths(tmp_path: Path) -> None:
    """
    Verify QCArtifactManifest serializes paths and retrieves written artifacts.

    The manifest is the artifact writer's durable contract for downstream stages
    and provenance.
    """

    # Build a manifest with one artifact.
    manifest = QCArtifactManifest(
        output_dir=tmp_path,
        artifacts={"summary": tmp_path / "qc_summary.json"},
        skipped=["figures"],
        warnings=["example warning"],
    )

    # Confirm manifest serialization is JSON-friendly.
    assert manifest.to_dict() == {
        "output_dir": str(tmp_path),
        "artifacts": {
            "summary": str(tmp_path / "qc_summary.json"),
        },
        "skipped": ["figures"],
        "warnings": ["example warning"],
    }

    # Confirm artifact path lookup works.
    assert manifest.get_path("summary") == tmp_path / "qc_summary.json"


def test_qc_artifact_manifest_get_path_rejects_missing_artifact(tmp_path: Path) -> None:
    """
    Verify manifest path lookup fails clearly for missing artifacts.

    Callers should not silently receive invalid paths.
    """

    # Build an empty manifest.
    manifest = QCArtifactManifest(output_dir=tmp_path)

    # Confirm missing artifact lookup fails clearly.
    with pytest.raises(QCArtifactError, match="was not written"):
        manifest.get_path("summary")


def test_prepare_qc_output_dir_creates_nested_directory(tmp_path: Path) -> None:
    """
    Verify QC output directory preparation creates nested directories.

    The artifact writer should be able to create a fresh QC output directory.
    """

    # Build a nested output directory path.
    output_dir = tmp_path / "nested" / "qc"

    # Prepare the output directory.
    prepared = prepare_qc_output_dir(output_dir)

    # Confirm the directory was created.
    assert prepared == output_dir
    assert prepared.exists()
    assert prepared.is_dir()


def test_prepare_qc_output_dir_rejects_existing_file(tmp_path: Path) -> None:
    """
    Verify QC output directory preparation rejects file paths.

    Artifact output roots must be directories, not existing files.
    """

    # Build an existing file path.
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    # Confirm existing files are rejected.
    with pytest.raises(QCArtifactError, match="must be a directory"):
        prepare_qc_output_dir(file_path)


def test_validate_artifact_dataframe_accepts_dataframe() -> None:
    """
    Verify artifact DataFrame validation accepts DataFrames.

    Empty DataFrames are allowed because some artifact tables may be schema-only.
    """

    # Confirm DataFrame validation passes.
    validate_artifact_dataframe(pd.DataFrame(), table_name="empty_table")


def test_validate_artifact_dataframe_rejects_non_dataframe() -> None:
    """
    Verify artifact DataFrame validation rejects non-DataFrame inputs.

    Table artifact writing requires pandas DataFrames.
    """

    # Confirm non-DataFrame input fails clearly.
    with pytest.raises(QCArtifactError, match="must be a pandas DataFrame"):
        validate_artifact_dataframe({"not": "dataframe"}, table_name="bad_table")  # type: ignore[arg-type]


def test_write_dataframe_artifact_writes_csv_atomically(tmp_path: Path) -> None:
    """
    Verify DataFrame artifact writing creates a readable CSV.

    The writer should include the index when requested and clean up temporary
    paths after successful replacement.
    """

    # Build a small table.
    table = pd.DataFrame({"value": [1, 2]}, index=["a", "b"])

    # Define the destination path.
    path = tmp_path / "tables" / "example.csv"

    # Write the DataFrame artifact.
    written_path = write_dataframe_artifact(table, path, index=True)

    # Confirm the expected path was returned.
    assert written_path == path

    # Confirm the file exists.
    assert path.exists()

    # Confirm the temporary path was removed.
    assert not build_temp_path(path).exists()

    # Read the CSV back.
    observed = pd.read_csv(path, index_col=0)

    # Confirm the table round-tripped.
    pdt.assert_frame_equal(observed, table)


def test_write_dataframe_artifact_rejects_non_dataframe(tmp_path: Path) -> None:
    """
    Verify DataFrame artifact writing rejects invalid table objects.

    Invalid table inputs should fail before filesystem writes.
    """

    # Confirm invalid table input fails clearly.
    with pytest.raises(QCArtifactError, match="must be a pandas DataFrame"):
        write_dataframe_artifact(  # type: ignore[arg-type]
            {"not": "dataframe"},
            tmp_path / "bad.csv",
            index=True,
        )


def test_write_json_artifact_writes_sorted_json(tmp_path: Path) -> None:
    """
    Verify JSON artifact writing creates readable JSON.

    JSON output should be formatted and JSON-compatible.
    """

    # Build a payload containing normal values and a pathlib path.
    payload = {
        "z": 1,
        "a": {
            "path": tmp_path / "example.txt",
        },
    }

    # Define the destination path.
    path = tmp_path / "summary" / "qc_summary.json"

    # Write the JSON artifact.
    written_path = write_json_artifact(payload, path)

    # Confirm the expected path was returned.
    assert written_path == path

    # Confirm the temporary path was removed.
    assert not build_temp_path(path).exists()

    # Read the JSON payload back.
    observed = json.loads(path.read_text(encoding="utf-8"))

    # Confirm pathlib values were converted to strings.
    assert observed == {
        "a": {
            "path": str(tmp_path / "example.txt"),
        },
        "z": 1,
    }


def test_write_json_artifact_rejects_non_dict_payload(tmp_path: Path) -> None:
    """
    Verify JSON artifact writing requires dictionary payloads.

    The summary artifact contract should always be object-shaped JSON.
    """

    # Confirm non-dictionary payloads fail clearly.
    with pytest.raises(QCArtifactError, match="payload must be a dictionary"):
        write_json_artifact(["not", "dict"], tmp_path / "bad.json")  # type: ignore[arg-type]


def test_write_h5ad_artifact_writes_readable_anndata(tmp_path: Path) -> None:
    """
    Verify AnnData artifact writing creates a readable h5ad file.

    This tests the optional qc.h5ad artifact path.
    """

    # Build a test AnnData object.
    adata = make_test_adata()

    # Define the destination path.
    path = tmp_path / "qc.h5ad"

    # Write the AnnData artifact.
    written_path = write_h5ad_artifact(adata, path)

    # Confirm the expected path was returned.
    assert written_path == path

    # Confirm the file exists.
    assert path.exists()

    # Read the AnnData object back.
    observed = ad.read_h5ad(path)

    # Confirm dimensions round-tripped.
    assert observed.n_obs == 2
    assert observed.n_vars == 2

    # Confirm names round-tripped.
    assert list(observed.obs_names) == ["cell_1", "cell_2"]
    assert list(observed.var_names) == ["gene_1", "gene_2"]


def test_write_h5ad_artifact_rejects_non_anndata(tmp_path: Path) -> None:
    """
    Verify AnnData artifact writing rejects invalid objects.

    The h5ad writer should only accept AnnData objects.
    """

    # Confirm non-AnnData input fails clearly.
    with pytest.raises(QCArtifactError, match="expected an AnnData object"):
        write_h5ad_artifact({"not": "anndata"}, tmp_path / "bad.h5ad")  # type: ignore[arg-type]


def test_build_qc_summary_payload_combines_module_summaries(tmp_path: Path) -> None:
    """
    Verify QC summary payload combines metric, threshold, and decision summaries.

    The summary JSON should be a single auditable payload spanning the QC module.
    """

    # Build the summary payload.
    payload = build_qc_summary_payload(
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        artifact_names={"cell_metrics": tmp_path / "cell_metrics.csv"},
        skipped=["figures"],
        warnings=["artifact warning"],
        summary_extra={"run_id": "test_run"},
    )

    # Confirm metrics summary is present.
    assert payload["metrics"]["n_cells"] == 2  # type: ignore[index]

    # Confirm threshold summary is present.
    assert payload["thresholds"]["n_thresholds"] == 1  # type: ignore[index]

    # Confirm decision summary is present.
    assert payload["decisions"]["n_cells_kept"] == 1  # type: ignore[index]

    # Confirm artifact paths are stringified.
    assert payload["artifacts"] == {
        "cell_metrics": str(tmp_path / "cell_metrics.csv"),
    }

    # Confirm skipped labels are preserved.
    assert payload["skipped"] == ["figures"]

    # Confirm warnings are preserved.
    assert payload["warnings"] == ["artifact warning"]

    # Confirm extra payload is namespaced.
    assert payload["extra"] == {"run_id": "test_run"}


def test_build_qc_summary_payload_rejects_invalid_summary_extra() -> None:
    """
    Verify summary payload construction rejects invalid extra payloads.

    Extra summary values should be object-shaped to remain predictable.
    """

    # Confirm invalid summary_extra fails clearly.
    with pytest.raises(QCArtifactError, match="summary_extra must be a dictionary"):
        build_qc_summary_payload(
            metrics_result=make_metrics_result(),
            threshold_result=make_threshold_result(),
            decision_result=make_decision_result(),
            artifact_names={},
            skipped=[],
            warnings=[],
            summary_extra=["not", "dict"],  # type: ignore[arg-type]
        )


def test_validate_qc_artifact_inputs_accepts_valid_inputs() -> None:
    """
    Verify artifact input validation accepts the expected result objects.

    The writer should accept valid metrics, thresholds, decisions, config, and
    optional AnnData.
    """

    # Confirm valid inputs pass validation.
    validate_qc_artifact_inputs(
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=QCConfig(),
        adata=make_test_adata(),
    )


def test_validate_qc_artifact_inputs_rejects_invalid_result_objects() -> None:
    """
    Verify artifact input validation rejects invalid result objects.

    This keeps writer errors clear before filesystem operations begin.
    """

    # Confirm invalid metrics_result fails clearly.
    with pytest.raises(QCArtifactError, match="metrics_result must be"):
        validate_qc_artifact_inputs(
            metrics_result={"bad": "metrics"},  # type: ignore[arg-type]
            threshold_result=make_threshold_result(),
            decision_result=make_decision_result(),
            config=QCConfig(),
            adata=None,
        )

    # Confirm invalid threshold_result fails clearly.
    with pytest.raises(QCArtifactError, match="threshold_result must be"):
        validate_qc_artifact_inputs(
            metrics_result=make_metrics_result(),
            threshold_result={"bad": "thresholds"},  # type: ignore[arg-type]
            decision_result=make_decision_result(),
            config=QCConfig(),
            adata=None,
        )

    # Confirm invalid decision_result fails clearly.
    with pytest.raises(QCArtifactError, match="decision_result must be"):
        validate_qc_artifact_inputs(
            metrics_result=make_metrics_result(),
            threshold_result=make_threshold_result(),
            decision_result={"bad": "decisions"},  # type: ignore[arg-type]
            config=QCConfig(),
            adata=None,
        )


def test_validate_qc_artifact_inputs_rejects_invalid_config_and_adata() -> None:
    """
    Verify artifact input validation rejects invalid config and AnnData inputs.

    Config and optional AnnData should be strongly typed.
    """

    # Confirm invalid config fails clearly.
    with pytest.raises(QCArtifactError, match="config must be a QCConfig"):
        validate_qc_artifact_inputs(
            metrics_result=make_metrics_result(),
            threshold_result=make_threshold_result(),
            decision_result=make_decision_result(),
            config={"bad": "config"},  # type: ignore[arg-type]
            adata=None,
        )

    # Confirm invalid AnnData fails clearly.
    with pytest.raises(QCArtifactError, match="adata must be an AnnData object"):
        validate_qc_artifact_inputs(
            metrics_result=make_metrics_result(),
            threshold_result=make_threshold_result(),
            decision_result=make_decision_result(),
            config=QCConfig(),
            adata={"bad": "adata"},  # type: ignore[arg-type]
        )


def test_write_qc_artifacts_writes_default_tables_summary_and_h5ad(tmp_path: Path) -> None:
    """
    Verify the full artifact writer writes default QC artifacts.

    With default output settings and an AnnData object provided, the writer should
    write metric tables, threshold table, decision tables, summary JSON, h5ad, and
    figures (when QC metrics are present in adata.obs).
    """

    # Build an output directory.
    output_dir = tmp_path / "qc"

    # Build an adata with QC metrics for figure generation.
    adata_with_qc = make_test_adata(
        obs_extra={
            "total_counts": [10.0, 20.0],
            "n_genes_by_counts": [2, 3],
            "pct_counts_mito": [5.0, 10.0],
        }
    )

    # Write QC artifacts.
    manifest = write_qc_artifacts(
        output_dir=output_dir,
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=QCConfig(),
        adata=adata_with_qc,
        summary_extra={"run_id": "test_run"},
    )

    # Confirm the manifest output directory.
    assert manifest.output_dir == output_dir

    # Confirm expected artifact labels were written including figures.
    assert set(manifest.artifacts) == {
        "cell_metrics",
        "gene_metrics",
        "feature_masks",
        "thresholds",
        "cell_decisions",
        "gene_decisions",
        "qc_h5ad",
        "summary",
        "figures",
    }

    # Confirm no artifacts were skipped.
    assert manifest.skipped == []

    # Confirm all written artifact paths exist.
    for artifact_name, artifact_value in manifest.artifacts.items():
        if artifact_name == "figures":
            # Figures are stored as a list of string paths.
            assert isinstance(artifact_value, list)
            for figure_path in artifact_value:
                assert Path(figure_path).exists()
        else:
            # Other artifacts are stored as Path objects.
            assert artifact_value.exists()

    # Read back cell metrics.
    cell_metrics = pd.read_csv(manifest.get_path("cell_metrics"), index_col=0)

    # Confirm cell metrics round-tripped.
    pdt.assert_frame_equal(cell_metrics, make_metrics_result().cell_metrics)

    # Read back threshold table.
    thresholds = pd.read_csv(manifest.get_path("thresholds"))

    # Confirm threshold table contents.
    assert thresholds.loc[0, "rule_name"] == "fixed_max_mito_percent"

    # Read back summary JSON.
    summary = json.loads(manifest.get_path("summary").read_text(encoding="utf-8"))

    # Confirm summary includes module summaries.
    assert summary["metrics"]["n_cells"] == 2
    assert summary["thresholds"]["n_thresholds"] == 1
    assert summary["decisions"]["n_cells_kept"] == 1

    # Confirm summary includes extra values.
    assert summary["extra"] == {"run_id": "test_run"}

    # Confirm summary does not list itself in artifact_names because it is written last.
    assert "summary" not in summary["artifacts"]

    # Confirm h5ad can be read.
    observed_adata = ad.read_h5ad(manifest.get_path("qc_h5ad"))

    # Confirm h5ad dimensions.
    assert observed_adata.shape == (2, 2)


def test_write_qc_artifacts_skips_h5ad_when_no_anndata_provided(tmp_path: Path) -> None:
    """
    Verify the full artifact writer skips h5ad when AnnData is absent.

    The writer should warn rather than failing because table artifacts are still
    valid without an AnnData object.
    """

    # Write QC artifacts without AnnData.
    manifest = write_qc_artifacts(
        output_dir=tmp_path / "qc",
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=QCConfig(outputs={"write_figures": False}),
        adata=None,
    )

    # Confirm h5ad was skipped.
    assert "qc_h5ad" in manifest.skipped

    # Confirm the h5ad warning was emitted.
    assert manifest.warnings == [
        "QCOutputConfig.write_h5ad is true, but no AnnData object was provided."
    ]

    # Confirm summary was still written.
    assert manifest.get_path("summary").exists()


def test_write_qc_artifacts_respects_output_flags(tmp_path: Path) -> None:
    """
    Verify the full artifact writer respects QCOutputConfig flags.

    Disabled outputs should be listed in skipped artifacts and not written to the
    manifest.
    """

    # Build a config that disables most outputs.
    config = QCConfig(
        outputs={
            "write_metrics_table": False,
            "write_filter_table": False,
            "write_threshold_table": False,
            "write_summary_json": True,
            "write_h5ad": False,
            "write_figures": False,
        }
    )

    # Write QC artifacts.
    manifest = write_qc_artifacts(
        output_dir=tmp_path / "qc",
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=config,
        adata=make_test_adata(),
    )

    # Confirm only the summary was written.
    assert set(manifest.artifacts) == {"summary"}

    # Confirm disabled artifacts were skipped.
    assert manifest.skipped == [
        "cell_metrics",
        "gene_metrics",
        "feature_masks",
        "thresholds",
        "cell_decisions",
        "gene_decisions",
        "qc_h5ad",
        "figures",
    ]

    # Confirm no warnings were emitted when disabled outputs are skipped explicitly.
    assert manifest.warnings == []


def test_write_qc_artifacts_can_skip_summary_json(tmp_path: Path) -> None:
    """
    Verify the full artifact writer can skip summary JSON.

    Some internal workflows may write tables only.
    """

    # Build a config that disables summary and figures.
    config = QCConfig(
        outputs={
            "write_summary_json": False,
            "write_figures": False,
            "write_h5ad": False,
        }
    )

    # Write QC artifacts.
    manifest = write_qc_artifacts(
        output_dir=tmp_path / "qc",
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=config,
        adata=None,
    )

    # Confirm summary was skipped.
    assert "summary" in manifest.skipped

    # Confirm summary was not written.
    assert "summary" not in manifest.artifacts


def test_write_qc_artifacts_rejects_invalid_output_dir(tmp_path: Path) -> None:
    """
    Verify the full artifact writer rejects invalid output directories.

    Existing files cannot be used as artifact output directories.
    """

    # Create an existing file path.
    file_path = tmp_path / "existing_file"
    file_path.write_text("not a directory", encoding="utf-8")

    # Confirm invalid output_dir fails clearly.
    with pytest.raises(QCArtifactError, match="must be a directory"):
        write_qc_artifacts(
            output_dir=file_path,
            metrics_result=make_metrics_result(),
            threshold_result=make_threshold_result(),
            decision_result=make_decision_result(),
            config=QCConfig(),
            adata=None,
        )


def test_ensure_parent_dir_creates_parent_directories(tmp_path: Path) -> None:
    """
    Verify parent directory creation helper creates nested parents.

    Atomic artifact writers rely on this helper before writing temp files.
    """

    # Build a nested artifact path.
    path = tmp_path / "a" / "b" / "artifact.csv"

    # Ensure the parent directory exists.
    ensure_parent_dir(path)

    # Confirm parent directories were created.
    assert path.parent.exists()
    assert path.parent.is_dir()


def test_build_temp_path_creates_hidden_sibling_path(tmp_path: Path) -> None:
    """
    Verify temporary artifact paths are built next to destination paths.

    Temporary files should live beside their final artifact so replacement is
    atomic on the same filesystem.
    """

    # Build an artifact path.
    path = tmp_path / "artifact.csv"

    # Build the temp path.
    temp_path = build_temp_path(path)

    # Confirm the temp path is a hidden sibling.
    assert temp_path == tmp_path / ".artifact.csv.tmp"


def test_cleanup_temp_path_removes_existing_temp_file(tmp_path: Path) -> None:
    """
    Verify temporary cleanup removes existing temp files.

    Failed writes should not leave stale temp files.
    """

    # Build a temp path.
    temp_path = tmp_path / ".artifact.csv.tmp"

    # Create the temp file.
    temp_path.write_text("temporary", encoding="utf-8")

    # Clean up the temp path.
    cleanup_temp_path(temp_path)

    # Confirm the temp file was removed.
    assert not temp_path.exists()


def test_cleanup_temp_path_ignores_missing_temp_file(tmp_path: Path) -> None:
    """
    Verify temporary cleanup ignores missing files.

    Cleanup should be safe to call after partial failures.
    """

    # Build a missing temp path.
    temp_path = tmp_path / ".missing.tmp"

    # Confirm cleanup does not raise.
    cleanup_temp_path(temp_path)

    # Confirm the path is still absent.
    assert not temp_path.exists()


def test_to_jsonable_converts_common_scientific_python_values(tmp_path: Path) -> None:
    """
    Verify JSON conversion handles paths, containers, pandas, and NumPy scalars.

    Summary payloads often contain scientific Python scalar types that standard
    json.dumps cannot serialize directly.
    """

    # Build values requiring conversion.
    value = {
        "path": tmp_path / "artifact.csv",
        "tuple": ("a", 1),
        "series": pd.Series({"x": np.int64(1)}),
        "dataframe": pd.DataFrame({"a": [np.float64(1.5)]}),
        "scalar": np.float64(2.5),
    }

    # Convert to JSON-friendly values.
    converted = to_jsonable(value)

    # Confirm converted values are JSON-compatible.
    assert converted == {
        "path": str(tmp_path / "artifact.csv"),
        "tuple": ["a", 1],
        "series": {"x": 1},
        "dataframe": [{"a": 1.5}],
        "scalar": 2.5,
    }

    # Confirm json.dumps accepts the converted payload.
    json.dumps(converted)


def test_write_qc_artifacts_emits_figures_when_enabled_and_adata_present(
    tmp_path: Path,
) -> None:
    """
    Verify QC figures are written when write_figures=True and adata is provided.

    This is the load-bearing test: figures should no longer be skipped when the
    writer has all necessary inputs.
    """

    # Build a config with write_figures enabled.
    config = QCConfig(outputs={"write_figures": True})

    # Build an adata with QC metrics and a condition column for grouping.
    adata = make_test_adata(
        obs_extra={
            "patient_id": ["P1", "P1"],
            "sample_id": ["P1_Normal", "P1_LE"],
            "condition": ["Normal", "Lymphedema"],
            "total_counts": [10.0, 20.0],
            "log1p_total_counts": [2.4, 3.0],
            "n_genes_by_counts": [2, 3],
            "log1p_n_genes_by_counts": [1.1, 1.4],
            "pct_counts_mito": [5.0, 10.0],
            "pct_counts_ribo": [20.0, 22.0],
            "pct_counts_hemoglobin": [0.0, 0.1],
            "pct_counts_in_top_20_genes": [25.0, 30.0],
            "cellquorum_qc_keep": [True, False],
        }
    )

    # Write QC artifacts including figures.
    manifest = write_qc_artifacts(
        output_dir=tmp_path / "qc",
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=config,
        adata=adata,
        group_key="condition",
    )

    # Confirm figures are recorded in the manifest and not skipped.
    assert "figures" in manifest.artifacts
    assert "figures" not in manifest.skipped

    # Confirm at least one figure path was recorded.
    figure_paths = manifest.artifacts["figures"]
    assert isinstance(figure_paths, list)
    assert len(figure_paths) > 0
    assert any("publication/qc_panel_A_mitochondrial_content" in path for path in figure_paths)
    assert any("publication/qc_panel_J_umi_detected_genes_normal" in path for path in figure_paths)
    assert any("publication/qc_panel_K_umi_detected_genes_le" in path for path in figure_paths)
    assert any("publication/qc_panel_L_by_condition_publication" in path for path in figure_paths)
    assert any("publication/qc_panel_M_cells_per_sample" in path for path in figure_paths)

    # Confirm all recorded figure files exist on disk.
    for figure_path in figure_paths:
        assert Path(figure_path).exists()


def test_write_qc_artifacts_skips_figures_without_adata(tmp_path: Path) -> None:
    """
    Verify QC figures are skipped with a clear warning when adata is absent.

    The writer should not crash; figures are a visual enhancement, not a
    required artifact.
    """

    # Build a config with write_figures enabled.
    config = QCConfig(outputs={"write_figures": True})

    # Write QC artifacts without providing an AnnData object.
    manifest = write_qc_artifacts(
        output_dir=tmp_path / "qc",
        metrics_result=make_metrics_result(),
        threshold_result=make_threshold_result(),
        decision_result=make_decision_result(),
        config=config,
        adata=None,
    )

    # Confirm figures were skipped.
    assert "figures" in manifest.skipped
    assert "figures" not in manifest.artifacts

    # Confirm a clear warning was emitted.
    assert any("no AnnData was provided" in warning for warning in manifest.warnings)
