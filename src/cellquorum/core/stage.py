"""Stage contracts and lifecycle records for CellQuorum pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import anndata as ad

StageStatus = Literal["success", "skipped", "failed"]


@dataclass(frozen=True)
class StageArtifact:
    """
    Describe a file or directory produced by a pipeline stage.

    Every CellQuorum stage should report its outputs explicitly instead of
    silently writing files to arbitrary locations. This makes reporting,
    testing, provenance tracking, and reruns much easier to manage.

    Args:
        name: Stable artifact name used by reports and downstream stages.
        path: Filesystem path to the artifact.
        kind: Artifact type, such as csv, parquet, json, markdown, figure, h5ad,
            or directory.
        description: Human-readable explanation of what the artifact contains.
    """

    # Store the stable artifact name used by reports and downstream stages.
    name: str

    # Store the artifact path.
    path: Path

    # Store the artifact kind, such as csv, json, figure, h5ad, or directory.
    kind: str

    # Store a human-readable artifact description.
    description: str

    def to_dict(self) -> dict[str, str]:
        """
        Convert the artifact record to a dictionary.

        This representation is used in provenance files, execution records, and
        reports. Paths are stringified because JSON cannot directly serialize
        Path objects.

        Returns:
            Dictionary representation of the artifact.
        """

        # Return a JSON-friendly artifact dictionary.
        return {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "description": self.description,
        }


@dataclass
class StageResult:
    """
    Store the complete result of one successfully executed pipeline stage.

    A publication-grade stage should not only return an AnnData object. It should
    return the updated data object plus all artifacts, notes, warnings, and
    structured metrics needed to audit what happened.

    Args:
        adata: Updated AnnData object after stage execution.
        artifacts: Files or directories produced by the stage.
        notes: Non-critical observations that should appear in reports.
        warnings: Important caveats that should appear in reports and provenance.
        metrics: JSON-serializable structured metrics for summaries and reports.
    """

    # Store the updated AnnData object after the stage has run.
    adata: ad.AnnData

    # Store files or directories produced by the stage.
    artifacts: list[StageArtifact] = field(default_factory=list)

    # Store non-critical observations that should appear in reports.
    notes: list[str] = field(default_factory=list)

    # Store important caveats that should appear in reports and provenance.
    warnings: list[str] = field(default_factory=list)

    # Store JSON-serializable structured metrics for summaries and reports.
    metrics: dict[str, object] = field(default_factory=dict)

    # Store the stage-level lifecycle status returned by the stage itself.
    status: StageStatus = "success"

    # Store the explicit skip reason when status is skipped.
    skip_reason: str | None = None

    # Store the method implementation version, when known.
    method_version: str | None = None

    # Store the backend used by the stage, when known.
    backend: str | None = None

    # Store the execution device used by the stage, when known.
    device: str | None = None

    # Store a fingerprint of the inputs consumed by the stage, when known.
    input_fingerprint: str | None = None

    # Store a fingerprint of the outputs produced by the stage, when known.
    output_fingerprint: str | None = None

    # Store an optional checkpoint path for resumable execution.
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        """
        Backfill explicit lifecycle fields from legacy skip metrics.

        Older stages in this repository report skips as
        ``metrics["skipped"] = True``. Keep those stages working while making the
        canonical status available on ``StageResult`` itself.
        """

        # Keep old MethodDispatchStage-style skips compatible with the new contract.
        if self.status == "success" and self.metrics.get("skipped") is True:
            self.status = "skipped"
            reason = self.metrics.get("reason")
            self.skip_reason = str(reason) if reason is not None else "skipped"

    @classmethod
    def skipped(
        cls,
        *,
        adata: ad.AnnData,
        reason: str,
        artifacts: list[StageArtifact] | None = None,
        notes: list[str] | None = None,
        warnings: list[str] | None = None,
        metrics: dict[str, object] | None = None,
        method_version: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> StageResult:
        """
        Build an explicit skipped stage result.

        Args:
            adata: Unchanged AnnData object to carry forward if needed.
            reason: Human-readable skip reason.
            artifacts: Optional artifacts emitted before the skip.
            notes: Optional notes emitted during the skip decision.
            warnings: Optional warnings emitted during the skip decision.
            metrics: Optional structured skip metrics.
            method_version: Optional method implementation version.
            backend: Optional backend label.
            device: Optional device label.
            input_fingerprint: Optional consumed-input fingerprint.
            output_fingerprint: Optional produced-output fingerprint.
            checkpoint_path: Optional checkpoint path.

        Returns:
            StageResult with status ``skipped``.
        """

        # Preserve the legacy metrics signal for old tests and reports.
        resolved_metrics = {} if metrics is None else dict(metrics)
        resolved_metrics.setdefault("skipped", True)
        resolved_metrics.setdefault("reason", reason)

        return cls(
            adata=adata,
            artifacts=[] if artifacts is None else list(artifacts),
            notes=[] if notes is None else list(notes),
            warnings=[] if warnings is None else list(warnings),
            metrics=resolved_metrics,
            status="skipped",
            skip_reason=reason,
            method_version=method_version,
            backend=backend,
            device=device,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            checkpoint_path=checkpoint_path,
        )

    def to_summary_dict(self) -> dict[str, object]:
        """
        Convert the stage result into a lightweight summary dictionary.

        The AnnData object itself is intentionally excluded because it is not
        suitable for JSON provenance. The summary captures outputs, messages, and
        metrics only.

        Returns:
            JSON-friendly summary of the stage result.
        """

        # Return a JSON-friendly result summary.
        return {
            "status": self.status,
            "skip_reason": self.skip_reason,
            "method_version": self.method_version,
            "backend": self.backend,
            "device": self.device,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class StageSkipReason:
    """
    Explain why a stage was skipped.

    Skips should be explicit, not silent. A stage may be skipped because the user
    disabled it, required metadata is absent, a backend is unavailable, the
    dataset is too small for a method, or the method is biologically/statistically
    inappropriate for the current design.

    Args:
        stage_name: Stable stage name.
        reason: Human-readable explanation for the skip.
        details: Optional structured details for provenance and reports.
    """

    # Store the skipped stage name.
    stage_name: str

    # Store the human-readable skip reason.
    reason: str

    # Store optional structured skip details.
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the skip reason to a dictionary.

        Returns:
            JSON-friendly skip reason dictionary.
        """

        # Return the skip reason as a JSON-friendly dictionary.
        return {
            "stage_name": self.stage_name,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class StageExecutionError:
    """
    Store a structured stage execution error.

    Exceptions should not be dumped into reports as raw tracebacks only. This
    object captures the stage name, error type, message, and optional structured
    details so failed runs remain auditable.

    Args:
        stage_name: Stable stage name.
        error_type: Exception type or error category.
        message: Human-readable error message.
        details: Optional structured details for provenance and reports.
    """

    # Store the failed stage name.
    stage_name: str

    # Store the error type or category.
    error_type: str

    # Store the human-readable error message.
    message: str

    # Store optional structured error details.
    details: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        *,
        stage_name: str,
        error: Exception,
        details: dict[str, object] | None = None,
    ) -> StageExecutionError:
        """
        Build a structured execution error from an exception.

        Args:
            stage_name: Stable stage name.
            error: Exception raised during stage execution.
            details: Optional structured details for provenance and reports.

        Returns:
            StageExecutionError containing the exception type and message.
        """

        # Return a structured error from the exception.
        return cls(
            stage_name=stage_name,
            error_type=type(error).__name__,
            message=str(error),
            details={} if details is None else dict(details),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert the execution error to a dictionary.

        Returns:
            JSON-friendly error dictionary.
        """

        # Return the execution error as a JSON-friendly dictionary.
        return {
            "stage_name": self.stage_name,
            "error_type": self.error_type,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class StageExecutionRecord:
    """
    Store the lifecycle record for one stage execution decision.

    A stage record is produced whether the stage succeeds, is skipped, or fails.
    This is the object that makes stage execution auditable. It captures status,
    timing, backend use, inputs, outputs, warnings, notes, metrics, skip reasons,
    and structured errors.

    Args:
        stage_name: Stable stage name.
        status: Stage execution status.
        started_at_utc: UTC timestamp when the stage decision/execution began.
        ended_at_utc: UTC timestamp when the stage decision/execution ended.
        duration_seconds: Stage duration in seconds.
        backend_used: Optional backend used by the stage.
        input_artifacts: Artifacts consumed by the stage.
        output_artifacts: Artifacts produced by the stage.
        notes: Non-critical notes emitted by the stage.
        warnings: Important caveats emitted by the stage.
        metrics: Structured metrics emitted by the stage.
        skip_reason: Structured skip reason when status is skipped.
        error: Structured error when status is failed.
    """

    # Store the stable stage name.
    stage_name: str

    # Store the execution status.
    status: StageStatus

    # Store the UTC start timestamp.
    started_at_utc: datetime

    # Store the UTC end timestamp.
    ended_at_utc: datetime

    # Store stage duration in seconds.
    duration_seconds: float

    # Store the backend used by the stage, when applicable.
    backend_used: str | None = None

    # Store the method implementation version, when reported by the stage.
    method_version: str | None = None

    # Store the execution device used by the stage, when reported by the stage.
    device: str | None = None

    # Store a fingerprint of the stage inputs, when reported by the stage.
    input_fingerprint: str | None = None

    # Store a fingerprint of the stage outputs, when reported by the stage.
    output_fingerprint: str | None = None

    # Store an optional checkpoint path for resumable execution.
    checkpoint_path: Path | None = None

    # Store artifacts consumed by the stage.
    input_artifacts: list[StageArtifact] = field(default_factory=list)

    # Store artifacts produced by the stage.
    output_artifacts: list[StageArtifact] = field(default_factory=list)

    # Store non-critical stage notes.
    notes: list[str] = field(default_factory=list)

    # Store important stage warnings.
    warnings: list[str] = field(default_factory=list)

    # Store structured stage metrics.
    metrics: dict[str, object] = field(default_factory=dict)

    # Store the skip reason when the stage is skipped.
    skip_reason: StageSkipReason | None = None

    # Store the structured execution error when the stage fails.
    error: StageExecutionError | None = None

    @classmethod
    def success(
        cls,
        *,
        stage_name: str,
        result: StageResult,
        started_at_utc: datetime,
        ended_at_utc: datetime,
        backend_used: str | None = None,
        input_artifacts: list[StageArtifact] | None = None,
    ) -> StageExecutionRecord:
        """
        Build a successful stage execution record.

        Args:
            stage_name: Stable stage name.
            result: StageResult returned by the stage.
            started_at_utc: UTC timestamp when execution began.
            ended_at_utc: UTC timestamp when execution ended.
            backend_used: Optional backend used by the stage.
            input_artifacts: Optional artifacts consumed by the stage.

        Returns:
            Successful StageExecutionRecord.
        """

        # Return a successful execution record.
        return cls(
            stage_name=stage_name,
            status="success",
            started_at_utc=_ensure_utc_datetime(started_at_utc),
            ended_at_utc=_ensure_utc_datetime(ended_at_utc),
            duration_seconds=_duration_seconds(started_at_utc, ended_at_utc),
            backend_used=result.backend or backend_used,
            method_version=result.method_version,
            device=result.device,
            input_fingerprint=result.input_fingerprint,
            output_fingerprint=result.output_fingerprint,
            checkpoint_path=result.checkpoint_path,
            input_artifacts=[] if input_artifacts is None else list(input_artifacts),
            output_artifacts=list(result.artifacts),
            notes=list(result.notes),
            warnings=list(result.warnings),
            metrics=dict(result.metrics),
            skip_reason=None,
            error=None,
        )

    @classmethod
    def skipped(
        cls,
        *,
        stage_name: str,
        reason: str,
        started_at_utc: datetime | None = None,
        ended_at_utc: datetime | None = None,
        backend_used: str | None = None,
        input_artifacts: list[StageArtifact] | None = None,
        details: dict[str, object] | None = None,
        notes: list[str] | None = None,
        warnings: list[str] | None = None,
        method_version: str | None = None,
        device: str | None = None,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> StageExecutionRecord:
        """
        Build a skipped stage execution record.

        Args:
            stage_name: Stable stage name.
            reason: Human-readable skip reason.
            started_at_utc: Optional UTC timestamp when skip decision began.
            ended_at_utc: Optional UTC timestamp when skip decision ended.
            backend_used: Optional backend considered by the stage.
            input_artifacts: Optional artifacts considered by the stage.
            details: Optional structured skip details.
            notes: Optional notes emitted during skip decision.
            warnings: Optional warnings emitted during skip decision.

        Returns:
            Skipped StageExecutionRecord.
        """

        # Resolve the start timestamp.
        resolved_start = _ensure_utc_datetime(started_at_utc or datetime.now(UTC))

        # Resolve the end timestamp.
        resolved_end = _ensure_utc_datetime(ended_at_utc or resolved_start)

        # Build the structured skip reason.
        skip_reason = StageSkipReason(
            stage_name=stage_name,
            reason=reason,
            details={} if details is None else dict(details),
        )

        # Return a skipped execution record.
        return cls(
            stage_name=stage_name,
            status="skipped",
            started_at_utc=resolved_start,
            ended_at_utc=resolved_end,
            duration_seconds=_duration_seconds(resolved_start, resolved_end),
            backend_used=backend_used,
            method_version=method_version,
            device=device,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            checkpoint_path=checkpoint_path,
            input_artifacts=[] if input_artifacts is None else list(input_artifacts),
            output_artifacts=[],
            notes=[] if notes is None else list(notes),
            warnings=[] if warnings is None else list(warnings),
            metrics={},
            skip_reason=skip_reason,
            error=None,
        )

    @classmethod
    def failed(
        cls,
        *,
        stage_name: str,
        error: Exception | StageExecutionError,
        started_at_utc: datetime,
        ended_at_utc: datetime,
        backend_used: str | None = None,
        input_artifacts: list[StageArtifact] | None = None,
        output_artifacts: list[StageArtifact] | None = None,
        notes: list[str] | None = None,
        warnings: list[str] | None = None,
        metrics: dict[str, object] | None = None,
    ) -> StageExecutionRecord:
        """
        Build a failed stage execution record.

        Args:
            stage_name: Stable stage name.
            error: Exception or structured StageExecutionError.
            started_at_utc: UTC timestamp when execution began.
            ended_at_utc: UTC timestamp when execution ended.
            backend_used: Optional backend used by the stage.
            input_artifacts: Optional artifacts consumed before failure.
            output_artifacts: Optional artifacts produced before failure.
            notes: Optional notes emitted before failure.
            warnings: Optional warnings emitted before failure.
            metrics: Optional metrics emitted before failure.

        Returns:
            Failed StageExecutionRecord.
        """

        # Convert exceptions into structured execution errors.
        structured_error = (
            error
            if isinstance(error, StageExecutionError)
            else StageExecutionError.from_exception(stage_name=stage_name, error=error)
        )

        # Return a failed execution record.
        return cls(
            stage_name=stage_name,
            status="failed",
            started_at_utc=_ensure_utc_datetime(started_at_utc),
            ended_at_utc=_ensure_utc_datetime(ended_at_utc),
            duration_seconds=_duration_seconds(started_at_utc, ended_at_utc),
            backend_used=backend_used,
            input_artifacts=[] if input_artifacts is None else list(input_artifacts),
            output_artifacts=[] if output_artifacts is None else list(output_artifacts),
            notes=[] if notes is None else list(notes),
            warnings=[] if warnings is None else list(warnings),
            metrics={} if metrics is None else dict(metrics),
            skip_reason=None,
            error=structured_error,
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert the stage execution record to a dictionary.

        Returns:
            JSON-friendly execution record dictionary.
        """

        # Return a JSON-friendly execution record.
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "started_at_utc": self.started_at_utc.isoformat(),
            "ended_at_utc": self.ended_at_utc.isoformat(),
            "duration_seconds": self.duration_seconds,
            "backend_used": self.backend_used,
            "method_version": self.method_version,
            "device": self.device,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "input_artifacts": [artifact.to_dict() for artifact in self.input_artifacts],
            "output_artifacts": [artifact.to_dict() for artifact in self.output_artifacts],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
            "skip_reason": None if self.skip_reason is None else self.skip_reason.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }


class PipelineStage(Protocol):
    """
    Define the interface every CellQuorum stage must implement.

    Concrete stages should receive a PipelineContext and return a StageResult.
    The context is typed as object here to avoid circular imports while the
    execution spine is being bootstrapped.
    """

    # Store the stable stage name.
    name: str

    def run(self, context: object) -> StageResult:
        """
        Execute the stage.

        Args:
            context: Pipeline execution context containing data, config, paths,
                backend registry, artifact manager, and provenance metadata.

        Returns:
            StageResult containing the updated AnnData object and stage outputs.
        """
        ...


def _ensure_utc_datetime(value: datetime) -> datetime:
    """
    Return a timezone-aware UTC datetime.

    Args:
        value: Datetime to normalize.

    Returns:
        Timezone-aware UTC datetime.
    """

    # Attach UTC to naive datetimes.
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)

    # Convert timezone-aware datetimes to UTC.
    return value.astimezone(UTC)


def _duration_seconds(started_at_utc: datetime, ended_at_utc: datetime) -> float:
    """
    Compute a non-negative duration in seconds.

    Args:
        started_at_utc: Stage start timestamp.
        ended_at_utc: Stage end timestamp.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the end timestamp occurs before the start timestamp.
    """

    # Normalize the start timestamp to UTC.
    start = _ensure_utc_datetime(started_at_utc)

    # Normalize the end timestamp to UTC.
    end = _ensure_utc_datetime(ended_at_utc)

    # Reject impossible negative durations.
    if end < start:
        raise ValueError(
            "Stage execution end time cannot be earlier than start time. "
            f"Start: {start.isoformat()}, end: {end.isoformat()}."
        )

    # Return the duration in seconds.
    return (end - start).total_seconds()
