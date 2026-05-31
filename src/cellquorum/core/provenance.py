"""Reusable provenance helpers for CellQuorum."""

from __future__ import annotations

# Import JSON for serializability validation.
import json

# Import dataclass helpers for structured provenance records.
from dataclasses import asdict, dataclass, is_dataclass

# Import datetime utilities for UTC provenance timestamps.
from datetime import UTC, datetime

# Import Path for filesystem-safe provenance payload handling.
from pathlib import Path

# Import shared provenance exception.
from cellquorum.core.exceptions import CellQuorumProvenanceError

# Define primitive JSON-compatible values.
type JsonPrimitive = str | int | float | bool | None

# Define recursive JSON-compatible values.
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Store one structured provenance record.

    A provenance record is a small, JSON-safe description of something that
    happened during a CellQuorum run. This object is intentionally generic so it
    can represent run initialization, config validation, backend checks, manifest
    validation, stage execution, report rendering, or future method gates.

    Args:
        name: Stable provenance record name.
        kind: Record category, such as run, config, backend, manifest, stage, or report.
        payload: JSON-safe structured payload.
        description: Human-readable explanation of the record.
        created_at_utc: UTC timestamp when the record was created.
    """

    # Store the stable provenance record name.
    name: str

    # Store the provenance record category.
    kind: str

    # Store the JSON-safe structured payload.
    payload: JsonValue

    # Store the human-readable provenance record description.
    description: str

    # Store the UTC creation timestamp.
    created_at_utc: datetime

    def to_dict(self) -> dict[str, JsonValue]:
        """
        Convert the provenance record to a JSON-safe dictionary.

        Returns:
            JSON-safe dictionary representation of the provenance record.

        Raises:
            CellQuorumProvenanceError: If the payload cannot be represented as JSON.
        """

        # Normalize the creation timestamp to a UTC ISO string.
        created_at = ensure_utc_datetime(self.created_at_utc).isoformat()

        # Build the JSON-safe provenance payload.
        record: dict[str, JsonValue] = {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "created_at_utc": created_at,
            "payload": self.payload,
        }

        # Validate the record before returning it.
        validate_json_payload(record, context=f"provenance record '{self.name}'")

        # Return the validated record.
        return record


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC datetime.

    Returns:
        Current UTC datetime.
    """

    # Return the current UTC datetime.
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-formatted string.

    Returns:
        Current UTC timestamp as an ISO string.
    """

    # Return the current UTC datetime as an ISO string.
    return utc_now().isoformat()


def ensure_utc_datetime(value: datetime) -> datetime:
    """
    Normalize a datetime to timezone-aware UTC.

    Args:
        value: Datetime to normalize.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        TypeError: If value is not a datetime.
    """

    # Validate the datetime input type.
    if not isinstance(value, datetime):
        raise TypeError(
            "ensure_utc_datetime expected a datetime object. " f"Received: {type(value).__name__}."
        )

    # Attach UTC to naive datetimes.
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)

    # Convert timezone-aware datetimes to UTC.
    return value.astimezone(UTC)


def make_provenance_record(
    *,
    name: str,
    kind: str,
    payload: object,
    description: str,
    created_at_utc: datetime | None = None,
) -> ProvenanceRecord:
    """
    Build a validated provenance record from an arbitrary payload.

    The payload is converted into a JSON-safe representation before the record is
    created. This keeps downstream artifact writing predictable and avoids
    failures from non-serializable objects such as Path or datetime.

    Args:
        name: Stable provenance record name.
        kind: Record category.
        payload: Structured payload to convert into JSON-safe form.
        description: Human-readable explanation of the record.
        created_at_utc: Optional UTC timestamp. Current UTC time is used when absent.

    Returns:
        Validated ProvenanceRecord.

    Raises:
        CellQuorumProvenanceError: If name, kind, description, or payload is invalid.
    """

    # Validate the record name.
    record_name = _require_non_empty_string(name, field_name="name")

    # Validate the record kind.
    record_kind = _require_non_empty_string(kind, field_name="kind")

    # Validate the record description.
    record_description = _require_non_empty_string(description, field_name="description")

    # Convert the payload into a JSON-safe object.
    json_safe_payload = to_json_safe(payload)

    # Validate the JSON-safe payload.
    validate_json_payload(json_safe_payload, context=f"provenance record '{record_name}'")

    # Resolve the creation timestamp.
    created_at = utc_now() if created_at_utc is None else ensure_utc_datetime(created_at_utc)

    # Return the validated provenance record.
    return ProvenanceRecord(
        name=record_name,
        kind=record_kind,
        payload=json_safe_payload,
        description=record_description,
        created_at_utc=created_at,
    )


def to_json_safe(value: object) -> JsonValue:
    """
    Convert common Python objects into JSON-safe values.

    This helper is intentionally conservative. It supports primitive JSON values,
    dictionaries with string-like keys, sequences, sets, Paths, datetimes,
    dataclasses, and objects exposing `to_dict()`. Unsupported objects raise a
    provenance error instead of being silently stringified.

    Args:
        value: Value to convert into a JSON-safe representation.

    Returns:
        JSON-safe value.

    Raises:
        CellQuorumProvenanceError: If the value cannot be converted safely.
    """

    # Return primitive JSON values directly.
    if value is None or isinstance(value, str | int | float | bool):
        return value

    # Convert Path objects into strings.
    if isinstance(value, Path):
        return str(value)

    # Convert datetimes into UTC ISO strings.
    if isinstance(value, datetime):
        return ensure_utc_datetime(value).isoformat()

    # Convert dataclasses into dictionaries before recursive conversion.
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))

    # Convert objects with a to_dict method before recursive conversion.
    if hasattr(value, "to_dict"):
        # Retrieve the to_dict method.
        to_dict_method = value.to_dict

        # Validate that to_dict is callable.
        if callable(to_dict_method):
            return to_json_safe(to_dict_method())

    # Convert dictionaries recursively.
    if isinstance(value, dict):
        return _dict_to_json_safe(value)

    # Convert list and tuple values recursively.
    if isinstance(value, list | tuple):
        return [to_json_safe(item) for item in value]

    # Convert sets into sorted JSON-safe lists for deterministic output.
    if isinstance(value, set):
        return sorted([to_json_safe(item) for item in value], key=str)

    # Raise a clear error for unsupported objects.
    raise CellQuorumProvenanceError(
        "Value cannot be converted into a JSON-safe provenance payload. "
        f"Received type: {type(value).__name__}."
    )


def validate_json_payload(payload: object, *, context: str = "payload") -> None:
    """
    Validate that a payload can be serialized to JSON.

    Args:
        payload: Candidate JSON payload.
        context: Human-readable context for error messages.

    Raises:
        CellQuorumProvenanceError: If the payload cannot be serialized.
    """

    # Try to serialize the payload using the standard JSON encoder.
    try:
        json.dumps(payload)

    # Convert serialization failures into CellQuorum provenance errors.
    except (TypeError, ValueError) as error:
        raise CellQuorumProvenanceError(
            f"Failed to serialize {context} as JSON: {error}"
        ) from error


def _dict_to_json_safe(value: dict[object, object]) -> dict[str, JsonValue]:
    """
    Convert a dictionary into a JSON-safe dictionary.

    Args:
        value: Dictionary to convert.

    Returns:
        JSON-safe dictionary with string keys.

    Raises:
        CellQuorumProvenanceError: If a key is missing or unsupported.
    """

    # Initialize the converted dictionary.
    converted: dict[str, JsonValue] = {}

    # Iterate over key-value pairs.
    for key, item in value.items():
        # Convert the dictionary key into a JSON-safe string.
        converted_key = _json_key_to_string(key)

        # Convert the dictionary value recursively.
        converted[converted_key] = to_json_safe(item)

    # Return the converted dictionary.
    return converted


def _json_key_to_string(key: object) -> str:
    """
    Convert a dictionary key into a JSON object key string.

    Args:
        key: Candidate dictionary key.

    Returns:
        String dictionary key.

    Raises:
        CellQuorumProvenanceError: If the key cannot be represented safely.
    """

    # Reject missing keys.
    if key is None:
        raise CellQuorumProvenanceError("JSON provenance dictionary keys cannot be None.")

    # Preserve string keys after stripping harmless whitespace.
    if isinstance(key, str):
        # Strip harmless whitespace.
        stripped_key = key.strip()

        # Reject empty keys.
        if not stripped_key:
            raise CellQuorumProvenanceError("JSON provenance dictionary keys cannot be empty.")

        # Return the cleaned key.
        return stripped_key

    # Convert simple scalar keys to strings.
    if isinstance(key, int | float | bool):
        return str(key)

    # Reject complex keys because they make provenance ambiguous.
    raise CellQuorumProvenanceError(
        "JSON provenance dictionary keys must be strings or simple scalar values. "
        f"Received key type: {type(key).__name__}."
    )


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    """
    Validate and clean a required non-empty string.

    Args:
        value: Candidate string value.
        field_name: Field name used in error messages.

    Returns:
        Cleaned string.

    Raises:
        CellQuorumProvenanceError: If the value is not a non-empty string.
    """

    # Reject non-string values.
    if not isinstance(value, str):
        raise CellQuorumProvenanceError(
            f"Provenance field '{field_name}' must be a string. "
            f"Received: {type(value).__name__}."
        )

    # Strip harmless whitespace.
    cleaned_value = value.strip()

    # Reject empty strings.
    if not cleaned_value:
        raise CellQuorumProvenanceError(f"Provenance field '{field_name}' cannot be empty.")

    # Return the cleaned string.
    return cleaned_value


__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "ProvenanceRecord",
    "ensure_utc_datetime",
    "make_provenance_record",
    "to_json_safe",
    "utc_now",
    "utc_now_iso",
    "validate_json_payload",
]
