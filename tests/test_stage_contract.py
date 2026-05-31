"""Tests for CellQuorum stage contracts and lifecycle records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from cellquorum.core.stage import (
    PipelineStage,
    StageArtifact,
    StageExecutionError,
    StageExecutionRecord,
    StageResult,
    StageSkipReason,
)


def test_stage_artifact_to_dict_stringifies_path() -> None:
    """
    Verify that StageArtifact serializes into a JSON-friendly dictionary.

    Artifact records are used in provenance files, report context, and execution
    records. Since JSON cannot directly serialize Path objects, the path must be
    stringified during serialization.
    """

    # Create a representative artifact.
    artifact = StageArtifact(
        name="qc_summary",
        path=Path("results/qc/qc_summary.csv"),
        kind="csv",
        description="Cell-level QC summary table.",
    )

    # Convert the artifact to a dictionary.
    payload = artifact.to_dict()

    # Confirm the artifact name was serialized.
    assert payload["name"] == "qc_summary"

    # Confirm the artifact path was serialized as a string.
    assert payload["path"] == "results/qc/qc_summary.csv"

    # Confirm the artifact kind was serialized.
    assert payload["kind"] == "csv"

    # Confirm the artifact description was serialized.
    assert payload["description"] == "Cell-level QC summary table."


def test_stage_result_accepts_artifacts_notes_warnings_and_metrics() -> None:
    """
    Verify that StageResult stores all required stage execution outputs.

    This test protects the most important early CellQuorum design rule: every
    stage must return the updated data object together with explicit artifacts,
    notes, warnings, and structured metrics. That contract prevents future
    stages from silently writing files or hiding important execution details.
    """

    # Create a tiny AnnData object so the stage result has a real data payload.
    adata = ad.AnnData(X=np.ones((2, 3)))

    # Create a representative artifact entry.
    artifact = StageArtifact(
        name="qc_summary",
        path=Path("results/qc/qc_summary.csv"),
        kind="csv",
        description="Cell-level QC summary table.",
    )

    # Create a stage result with every supported metadata field populated.
    result = StageResult(
        adata=adata,
        artifacts=[artifact],
        notes=["QC completed."],
        warnings=["Example warning."],
        metrics={"n_cells": 2, "n_genes": 3},
    )

    # Confirm the AnnData object was retained.
    assert result.adata.n_obs == 2

    # Confirm the AnnData object retained the expected number of variables.
    assert result.adata.n_vars == 3

    # Confirm artifact metadata is accessible.
    assert result.artifacts[0].name == "qc_summary"

    # Confirm artifact paths are stored as Path objects.
    assert result.artifacts[0].path == Path("results/qc/qc_summary.csv")

    # Confirm notes are preserved.
    assert result.notes == ["QC completed."]

    # Confirm warnings are preserved.
    assert result.warnings == ["Example warning."]

    # Confirm structured metrics are preserved.
    assert result.metrics["n_cells"] == 2

    # Confirm structured metrics can store gene counts.
    assert result.metrics["n_genes"] == 3


def test_stage_result_to_summary_dict_excludes_adata_and_serializes_outputs() -> None:
    """
    Verify that StageResult can produce a lightweight provenance summary.

    AnnData objects are not appropriate for JSON provenance. The summary should
    serialize only artifacts, notes, warnings, and metrics.
    """

    # Create a tiny AnnData object.
    adata = ad.AnnData(X=np.ones((1, 2)))

    # Create a representative artifact entry.
    artifact = StageArtifact(
        name="qc_metrics",
        path=Path("results/qc/qc_metrics.json"),
        kind="json",
        description="Structured QC metrics.",
    )

    # Build a stage result.
    result = StageResult(
        adata=adata,
        artifacts=[artifact],
        notes=["Metrics computed."],
        warnings=["Low cell count."],
        metrics={"n_cells": 1},
    )

    # Convert the result to a summary dictionary.
    payload = result.to_summary_dict()

    # Confirm AnnData is not included in the summary.
    assert "adata" not in payload

    # Confirm artifacts were serialized.
    assert payload["artifacts"] == [artifact.to_dict()]

    # Confirm notes were serialized.
    assert payload["notes"] == ["Metrics computed."]

    # Confirm warnings were serialized.
    assert payload["warnings"] == ["Low cell count."]

    # Confirm metrics were serialized.
    assert payload["metrics"] == {"n_cells": 1}


def test_stage_skip_reason_to_dict() -> None:
    """
    Verify that StageSkipReason serializes skip details.

    Skipped stages should be auditable. A skip record must explain why the stage
    did not run and preserve any structured gating details.
    """

    # Create a structured skip reason.
    skip_reason = StageSkipReason(
        stage_name="integration",
        reason="Only one batch was present.",
        details={"available_batches": 1},
    )

    # Convert the skip reason to a dictionary.
    payload = skip_reason.to_dict()

    # Confirm the stage name was serialized.
    assert payload["stage_name"] == "integration"

    # Confirm the reason was serialized.
    assert payload["reason"] == "Only one batch was present."

    # Confirm the details were serialized.
    assert payload["details"] == {"available_batches": 1}


def test_stage_execution_error_from_exception_and_to_dict() -> None:
    """
    Verify that exceptions become structured execution errors.

    Failed stages should preserve the exception type and message in a stable
    dictionary representation for provenance and reports.
    """

    # Create a representative exception.
    error = RuntimeError("QC threshold calculation failed.")

    # Build a structured execution error.
    execution_error = StageExecutionError.from_exception(
        stage_name="qc",
        error=error,
        details={"step": "thresholding"},
    )

    # Confirm the stage name was retained.
    assert execution_error.stage_name == "qc"

    # Confirm the exception type was captured.
    assert execution_error.error_type == "RuntimeError"

    # Confirm the exception message was captured.
    assert execution_error.message == "QC threshold calculation failed."

    # Confirm details were preserved.
    assert execution_error.details == {"step": "thresholding"}

    # Convert the execution error to a dictionary.
    payload = execution_error.to_dict()

    # Confirm the dictionary contains the expected error type.
    assert payload["error_type"] == "RuntimeError"

    # Confirm the dictionary contains the expected message.
    assert payload["message"] == "QC threshold calculation failed."


def test_stage_execution_record_success_captures_result_and_timing() -> None:
    """
    Verify that a successful execution record captures stage outputs.

    A successful stage lifecycle record should include timing, backend, inputs,
    outputs, notes, warnings, and metrics.
    """

    # Define a stable stage start time.
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Define a stable stage end time.
    ended_at = started_at + timedelta(seconds=2.5)

    # Create an input artifact.
    input_artifact = StageArtifact(
        name="raw_input",
        path=Path("objects/raw.h5ad"),
        kind="h5ad",
        description="Raw input AnnData object.",
    )

    # Create an output artifact.
    output_artifact = StageArtifact(
        name="qc_output",
        path=Path("objects/qc.h5ad"),
        kind="h5ad",
        description="QC-filtered AnnData object.",
    )

    # Create a stage result.
    result = StageResult(
        adata=ad.AnnData(X=np.ones((2, 2))),
        artifacts=[output_artifact],
        notes=["QC completed."],
        warnings=["Low mitochondrial threshold adjusted."],
        metrics={"n_cells_after_qc": 2},
    )

    # Build the successful execution record.
    record = StageExecutionRecord.success(
        stage_name="qc",
        result=result,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        backend_used="python",
        input_artifacts=[input_artifact],
    )

    # Confirm the stage name was stored.
    assert record.stage_name == "qc"

    # Confirm the success status was stored.
    assert record.status == "success"

    # Confirm duration was calculated.
    assert record.duration_seconds == 2.5

    # Confirm the backend was stored.
    assert record.backend_used == "python"

    # Confirm input artifacts were stored.
    assert record.input_artifacts == [input_artifact]

    # Confirm output artifacts came from the StageResult.
    assert record.output_artifacts == [output_artifact]

    # Confirm notes came from the StageResult.
    assert record.notes == ["QC completed."]

    # Confirm warnings came from the StageResult.
    assert record.warnings == ["Low mitochondrial threshold adjusted."]

    # Confirm metrics came from the StageResult.
    assert record.metrics == {"n_cells_after_qc": 2}

    # Confirm success records do not have skip reasons.
    assert record.skip_reason is None

    # Confirm success records do not have errors.
    assert record.error is None


def test_stage_execution_record_success_normalizes_naive_times_to_utc() -> None:
    """
    Verify that successful records normalize naive datetimes to UTC.

    Stage lifecycle records should always serialize timezone-aware timestamps.
    """

    # Define a naive start time.
    started_at = datetime(2026, 1, 1, 12, 0, 0)

    # Define a naive end time.
    ended_at = datetime(2026, 1, 1, 12, 0, 1)

    # Create a stage result.
    result = StageResult(adata=ad.AnnData(X=np.ones((1, 1))))

    # Build a successful execution record.
    record = StageExecutionRecord.success(
        stage_name="qc",
        result=result,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
    )

    # Confirm the start time is timezone-aware UTC.
    assert record.started_at_utc.tzinfo == UTC

    # Confirm the end time is timezone-aware UTC.
    assert record.ended_at_utc.tzinfo == UTC

    # Confirm duration was calculated correctly.
    assert record.duration_seconds == 1.0


def test_stage_execution_record_skipped_captures_reason_and_details() -> None:
    """
    Verify that skipped execution records capture explicit skip information.

    Skips should not be silent. The record should explain why the stage was
    skipped and preserve structured gating details.
    """

    # Define a stable skip decision time.
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Build a skipped execution record.
    record = StageExecutionRecord.skipped(
        stage_name="integration",
        reason="Only one batch was available.",
        started_at_utc=started_at,
        ended_at_utc=started_at,
        backend_used=None,
        details={"n_batches": 1},
        notes=["Integration not required."],
        warnings=["Batch metadata was absent."],
    )

    # Confirm the stage name was stored.
    assert record.stage_name == "integration"

    # Confirm the skipped status was stored.
    assert record.status == "skipped"

    # Confirm duration is zero for an instantaneous skip decision.
    assert record.duration_seconds == 0.0

    # Confirm skipped records do not produce output artifacts.
    assert record.output_artifacts == []

    # Confirm notes were stored.
    assert record.notes == ["Integration not required."]

    # Confirm warnings were stored.
    assert record.warnings == ["Batch metadata was absent."]

    # Confirm skip reason exists.
    assert record.skip_reason is not None

    # Confirm skip reason text was stored.
    assert record.skip_reason.reason == "Only one batch was available."

    # Confirm skip details were stored.
    assert record.skip_reason.details == {"n_batches": 1}

    # Confirm skipped records do not have execution errors.
    assert record.error is None


def test_stage_execution_record_failed_captures_exception() -> None:
    """
    Verify that failed execution records capture structured exception details.

    Failed stage records should preserve partial notes, warnings, metrics, input
    artifacts, output artifacts, backend information, and a structured error.
    """

    # Define a stable stage start time.
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Define a stable stage end time.
    ended_at = started_at + timedelta(seconds=3)

    # Create an input artifact.
    input_artifact = StageArtifact(
        name="input",
        path=Path("objects/input.h5ad"),
        kind="h5ad",
        description="Input AnnData object.",
    )

    # Create an output artifact produced before failure.
    output_artifact = StageArtifact(
        name="partial_qc",
        path=Path("results/qc/partial.csv"),
        kind="csv",
        description="Partial QC output.",
    )

    # Build a failed execution record from an exception.
    record = StageExecutionRecord.failed(
        stage_name="qc",
        error=RuntimeError("QC failed."),
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        backend_used="python",
        input_artifacts=[input_artifact],
        output_artifacts=[output_artifact],
        notes=["Started QC."],
        warnings=["Partial output was written."],
        metrics={"n_cells_before_failure": 10},
    )

    # Confirm the failed status was stored.
    assert record.status == "failed"

    # Confirm duration was calculated.
    assert record.duration_seconds == 3.0

    # Confirm backend was stored.
    assert record.backend_used == "python"

    # Confirm input artifacts were stored.
    assert record.input_artifacts == [input_artifact]

    # Confirm output artifacts were stored.
    assert record.output_artifacts == [output_artifact]

    # Confirm notes were stored.
    assert record.notes == ["Started QC."]

    # Confirm warnings were stored.
    assert record.warnings == ["Partial output was written."]

    # Confirm metrics were stored.
    assert record.metrics == {"n_cells_before_failure": 10}

    # Confirm failed records do not have skip reasons.
    assert record.skip_reason is None

    # Confirm the structured error exists.
    assert record.error is not None

    # Confirm the structured error type was captured.
    assert record.error.error_type == "RuntimeError"

    # Confirm the structured error message was captured.
    assert record.error.message == "QC failed."


def test_stage_execution_record_failed_accepts_structured_error() -> None:
    """
    Verify that failed execution records can accept a prebuilt structured error.

    Some stages may create richer errors directly rather than passing an
    exception object.
    """

    # Define a stable stage start time.
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Define a stable stage end time.
    ended_at = started_at + timedelta(seconds=1)

    # Create a structured execution error.
    execution_error = StageExecutionError(
        stage_name="annotation",
        error_type="ReferenceError",
        message="Reference atlas was unavailable.",
        details={"reference": "missing_atlas"},
    )

    # Build a failed execution record from the structured error.
    record = StageExecutionRecord.failed(
        stage_name="annotation",
        error=execution_error,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
    )

    # Confirm the structured error was preserved.
    assert record.error == execution_error

    # Confirm the failed status was stored.
    assert record.status == "failed"


def test_stage_execution_record_to_dict_serializes_all_fields() -> None:
    """
    Verify that execution records serialize to JSON-friendly dictionaries.

    Stage execution records will be written to provenance files and report
    context, so serialization must include timing, status, backend, artifacts,
    notes, warnings, metrics, skip reason, and error fields.
    """

    # Define a stable stage start time.
    started_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Define a stable stage end time.
    ended_at = started_at + timedelta(seconds=1)

    # Create an output artifact.
    artifact = StageArtifact(
        name="qc_summary",
        path=Path("results/qc/qc_summary.csv"),
        kind="csv",
        description="QC summary.",
    )

    # Create a stage result.
    result = StageResult(
        adata=ad.AnnData(X=np.ones((1, 1))),
        artifacts=[artifact],
        notes=["Done."],
        warnings=["Warning."],
        metrics={"n_cells": 1},
    )

    # Build a successful execution record.
    record = StageExecutionRecord.success(
        stage_name="qc",
        result=result,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        backend_used="python",
    )

    # Convert the record to a dictionary.
    payload = record.to_dict()

    # Confirm the stage name was serialized.
    assert payload["stage_name"] == "qc"

    # Confirm the status was serialized.
    assert payload["status"] == "success"

    # Confirm timestamps were serialized as strings.
    assert payload["started_at_utc"] == started_at.isoformat()

    # Confirm end timestamps were serialized as strings.
    assert payload["ended_at_utc"] == ended_at.isoformat()

    # Confirm duration was serialized.
    assert payload["duration_seconds"] == 1.0

    # Confirm backend was serialized.
    assert payload["backend_used"] == "python"

    # Confirm output artifacts were serialized.
    assert payload["output_artifacts"] == [artifact.to_dict()]

    # Confirm notes were serialized.
    assert payload["notes"] == ["Done."]

    # Confirm warnings were serialized.
    assert payload["warnings"] == ["Warning."]

    # Confirm metrics were serialized.
    assert payload["metrics"] == {"n_cells": 1}

    # Confirm success records serialize no skip reason.
    assert payload["skip_reason"] is None

    # Confirm success records serialize no error.
    assert payload["error"] is None


def test_stage_execution_record_rejects_negative_duration() -> None:
    """
    Verify that execution records reject impossible negative durations.

    Negative durations indicate a clock or lifecycle bug and should fail during
    record construction.
    """

    # Define a start time.
    started_at = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)

    # Define an impossible end time before the start.
    ended_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Create a stage result.
    result = StageResult(adata=ad.AnnData(X=np.ones((1, 1))))

    # Confirm negative durations are rejected.
    with pytest.raises(ValueError, match="end time cannot be earlier"):
        StageExecutionRecord.success(
            stage_name="qc",
            result=result,
            started_at_utc=started_at,
            ended_at_utc=ended_at,
        )


def test_pipeline_stage_protocol_accepts_stage_like_class() -> None:
    """
    Verify that a stage-like class can satisfy the PipelineStage protocol shape.

    This is a lightweight structural test. The protocol is intentionally simple:
    a stage exposes a stable name and a run method that returns StageResult.
    """

    class ExampleStage:
        """
        Minimal concrete stage used only for protocol-shape testing.
        """

        # Store a stable stage name.
        name = "example"

        def run(self, context: object) -> StageResult:
            """
            Return a simple StageResult.

            Args:
                context: Pipeline context placeholder.

            Returns:
                StageResult with a tiny AnnData object.
            """

            # Return a minimal stage result.
            return StageResult(adata=ad.AnnData(X=np.ones((1, 1))))

    # Create an example stage.
    stage: PipelineStage = ExampleStage()

    # Run the example stage.
    result = stage.run(context=object())

    # Confirm the stage name is accessible.
    assert stage.name == "example"

    # Confirm the stage returns a StageResult.
    assert isinstance(result, StageResult)
