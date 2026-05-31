"""Shared typing utilities for CellQuorum."""

from __future__ import annotations

# Import Path for path-like type aliases.
from pathlib import Path

# Import runtime-checkable protocols and typed dictionary helpers.
from typing import Protocol, TypedDict, runtime_checkable

# Define filesystem path-like inputs accepted by public APIs.
type PathLike = str | Path

# Define scalar values commonly used in configuration, metadata, and metrics.
type ScalarValue = str | int | float | bool | None

# Define primitive JSON-compatible values.
type JsonPrimitive = str | int | float | bool | None

# Define recursive JSON-compatible values.
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

# Define simple metric payloads used by stages and reports.
type MetricValue = int | float | str | bool | None

# Define stage metric dictionaries.
type MetricsDict = dict[str, MetricValue | list[MetricValue] | dict[str, MetricValue]]

# Define metadata dictionaries for sample, run, and artifact metadata.
type MetadataDict = dict[str, ScalarValue | list[ScalarValue] | dict[str, ScalarValue]]

# Define generic string-keyed parameter dictionaries.
type ParameterDict = dict[str, ScalarValue | list[ScalarValue]]


class ArtifactPayload(TypedDict):
    """
    Represent a JSON-friendly artifact payload.

    This typed dictionary mirrors the serialized form of a CellQuorum artifact.
    It is useful for provenance, report context, and tests that inspect artifact
    dictionaries rather than artifact objects.

    Args:
        name: Stable artifact name.
        path: Artifact path serialized as a string.
        kind: Artifact kind, such as csv, json, figure, h5ad, or directory.
        description: Human-readable explanation of the artifact.
    """

    # Store the stable artifact name.
    name: str

    # Store the artifact path as a string.
    path: str

    # Store the artifact kind.
    kind: str

    # Store the artifact description.
    description: str


class StageExecutionPayload(TypedDict, total=False):
    """
    Represent a JSON-friendly stage execution payload.

    The payload is intentionally permissive because skipped, failed, and
    successful stage records have overlapping but not identical fields.

    Args:
        stage_name: Stable stage name.
        status: Stage execution status.
        started_at_utc: UTC start timestamp serialized as a string.
        ended_at_utc: UTC end timestamp serialized as a string.
        duration_seconds: Stage duration in seconds.
        backend_used: Backend used by the stage, when applicable.
        input_artifacts: Serialized input artifact records.
        output_artifacts: Serialized output artifact records.
        notes: Non-critical notes emitted by the stage.
        warnings: Important warnings emitted by the stage.
        metrics: Structured stage metrics.
        skip_reason: Optional structured skip reason.
        error: Optional structured execution error.
    """

    # Store the stable stage name.
    stage_name: str

    # Store the stage status.
    status: str

    # Store the UTC start timestamp.
    started_at_utc: str

    # Store the UTC end timestamp.
    ended_at_utc: str

    # Store the stage duration in seconds.
    duration_seconds: float

    # Store the backend used by the stage.
    backend_used: str | None

    # Store serialized input artifacts.
    input_artifacts: list[ArtifactPayload]

    # Store serialized output artifacts.
    output_artifacts: list[ArtifactPayload]

    # Store stage notes.
    notes: list[str]

    # Store stage warnings.
    warnings: list[str]

    # Store structured stage metrics.
    metrics: dict[str, JsonValue]

    # Store an optional skip reason payload.
    skip_reason: dict[str, JsonValue] | None

    # Store an optional error payload.
    error: dict[str, JsonValue] | None


@runtime_checkable
class SupportsToDict(Protocol):
    """
    Define the protocol for objects that can serialize themselves to dictionaries.

    This protocol is useful for provenance and artifact code that accepts richer
    objects but needs to convert them into JSON-friendly dictionaries.
    """

    def to_dict(self) -> dict[str, object]:
        """
        Convert the object into a dictionary.

        Returns:
            Dictionary representation of the object.
        """
        ...


@runtime_checkable
class SupportsName(Protocol):
    """
    Define the protocol for objects exposing a stable name.

    Stage objects, backend objects, artifacts, and registry entries commonly need
    a stable name for lookup, provenance, and reporting.
    """

    # Store the stable object name.
    name: str


@runtime_checkable
class SupportsRun(Protocol):
    """
    Define the protocol for objects exposing a run method.

    This lightweight protocol is intentionally generic. More specific stage
    contracts are defined in `cellquorum.core.stage`.
    """

    def run(self, context: object) -> object:
        """
        Execute the object against a runtime context.

        Args:
            context: Runtime context object.

        Returns:
            Execution result.
        """
        ...


def is_path_like(value: object) -> bool:
    """
    Return whether a value is path-like.

    Args:
        value: Candidate value.

    Returns:
        True when the value is a string or Path object, otherwise False.
    """

    # Return whether the value is a supported path-like type.
    return isinstance(value, str | Path)


def is_scalar_value(value: object) -> bool:
    """
    Return whether a value is a supported scalar metadata value.

    Args:
        value: Candidate value.

    Returns:
        True when the value is a supported scalar value, otherwise False.
    """

    # Return whether the value is one of the supported scalar types.
    return value is None or isinstance(value, str | int | float | bool)


__all__ = [
    "ArtifactPayload",
    "JsonPrimitive",
    "JsonValue",
    "MetadataDict",
    "MetricValue",
    "MetricsDict",
    "ParameterDict",
    "PathLike",
    "ScalarValue",
    "StageExecutionPayload",
    "SupportsName",
    "SupportsRun",
    "SupportsToDict",
    "is_path_like",
    "is_scalar_value",
]
