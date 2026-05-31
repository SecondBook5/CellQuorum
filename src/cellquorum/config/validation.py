"""Reusable configuration validation helpers for CellQuorum."""

from __future__ import annotations

# Import Mapping and Sequence for defensive runtime validation.
from collections.abc import Mapping, Sequence

# Import dataclass for structured validation issues.
from dataclasses import dataclass, field

# Import Literal for constrained severity labels.
from typing import Literal

# Import shared configuration exception type.
from cellquorum.core.exceptions import CellQuorumConfigError

ValidationSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """
    Store one structured configuration validation issue.

    This object gives config modules a consistent way to represent validation
    findings before deciding whether they should fail execution or be surfaced as
    warnings. It is intentionally generic so it can be used by QC, preprocessing,
    reporting, plotting, method gates, and future plugin-style modules.

    Args:
        field_path: Dot-separated field path, such as `qc.mito.max_percent`.
        message: Human-readable validation message.
        severity: Issue severity, either error or warning.
        details: Optional structured details for provenance or reports.
    """

    # Store the dot-separated field path.
    field_path: str

    # Store the human-readable validation message.
    message: str

    # Store the validation issue severity.
    severity: ValidationSeverity = "error"

    # Store optional structured validation details.
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the validation issue into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the validation issue.
        """

        # Return a JSON-friendly validation issue dictionary.
        return {
            "field_path": self.field_path,
            "message": self.message,
            "severity": self.severity,
            "details": dict(self.details),
        }


class ConfigValidationError(CellQuorumConfigError):
    """
    Report reusable configuration validation failures.

    This error is used by helper functions in this module when a value has the
    wrong type, is missing, is outside an allowed range, or uses an unsupported
    option.
    """


def format_field_path(*parts: str | int) -> str:
    """
    Build a dot-separated field path.

    Args:
        parts: Field path components.

    Returns:
        Dot-separated field path.

    Raises:
        ConfigValidationError: If no path parts are supplied.
    """

    # Reject empty field paths.
    if not parts:
        raise ConfigValidationError("Field path must contain at least one component.")

    # Convert every path component into a stripped string.
    cleaned_parts = [str(part).strip() for part in parts]

    # Reject empty path components.
    if any(not part for part in cleaned_parts):
        raise ConfigValidationError(f"Field path contains an empty component: {cleaned_parts}.")

    # Return the dot-separated field path.
    return ".".join(cleaned_parts)


def require_mapping(value: object, *, field_path: str) -> dict[str, object]:
    """
    Validate that a config value is a mapping.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.

    Returns:
        Dictionary copy of the mapping.

    Raises:
        ConfigValidationError: If value is not a mapping.
    """

    # Reject non-mapping values.
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be a mapping. "
            f"Received: {type(value).__name__}."
        )

    # Return a plain dictionary copy.
    return dict(value)


def require_required_keys(
    mapping: Mapping[str, object],
    *,
    required_keys: Sequence[str],
    field_path: str,
) -> None:
    """
    Validate that required keys exist in a mapping.

    Args:
        mapping: Mapping to inspect.
        required_keys: Required key names.
        field_path: Field path used in error messages.

    Raises:
        ConfigValidationError: If one or more required keys are absent.
    """

    # Identify required keys that are missing.
    missing_keys = [key for key in required_keys if key not in mapping]

    # Raise a clear error when required keys are missing.
    if missing_keys:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' is missing required key(s): "
            f"{', '.join(missing_keys)}."
        )


def reject_unknown_keys(
    mapping: Mapping[str, object],
    *,
    allowed_keys: Sequence[str],
    field_path: str,
) -> None:
    """
    Validate that a mapping contains only allowed keys.

    Args:
        mapping: Mapping to inspect.
        allowed_keys: Allowed key names.
        field_path: Field path used in error messages.

    Raises:
        ConfigValidationError: If unknown keys are present.
    """

    # Convert allowed keys into a set for lookup.
    allowed_key_set = set(allowed_keys)

    # Identify unknown keys.
    unknown_keys = sorted(key for key in mapping if key not in allowed_key_set)

    # Raise a clear error when unknown keys exist.
    if unknown_keys:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' contains unsupported key(s): "
            f"{', '.join(unknown_keys)}."
        )


def require_non_empty_string(value: object, *, field_path: str) -> str:
    """
    Validate that a config value is a non-empty string.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.

    Returns:
        Cleaned string value.

    Raises:
        ConfigValidationError: If value is not a non-empty string.
    """

    # Reject non-string values.
    if not isinstance(value, str):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be a string. "
            f"Received: {type(value).__name__}."
        )

    # Strip harmless surrounding whitespace.
    cleaned_value = value.strip()

    # Reject empty strings.
    if not cleaned_value:
        raise ConfigValidationError(f"Configuration field '{field_path}' cannot be empty.")

    # Return the cleaned string.
    return cleaned_value


def require_bool(value: object, *, field_path: str) -> bool:
    """
    Validate that a config value is boolean.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.

    Returns:
        Boolean value.

    Raises:
        ConfigValidationError: If value is not boolean.
    """

    # Reject non-boolean values.
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be a boolean. "
            f"Received: {type(value).__name__}."
        )

    # Return the validated boolean.
    return value


def require_int(
    value: object,
    *,
    field_path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """
    Validate that a config value is an integer within optional bounds.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        Integer value.

    Raises:
        ConfigValidationError: If value is not an integer or violates bounds.
    """

    # Reject booleans because bool is a subclass of int in Python.
    if isinstance(value, bool):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be an integer, not boolean."
        )

    # Reject non-integer values.
    if not isinstance(value, int):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be an integer. "
            f"Received: {type(value).__name__}."
        )

    # Validate the lower bound when supplied.
    if minimum is not None and value < minimum:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be >= {minimum}. " f"Received: {value}."
        )

    # Validate the upper bound when supplied.
    if maximum is not None and value > maximum:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be <= {maximum}. " f"Received: {value}."
        )

    # Return the validated integer.
    return value


def require_float(
    value: object,
    *,
    field_path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Validate that a config value is numeric within optional bounds.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        Float value.

    Raises:
        ConfigValidationError: If value is not numeric or violates bounds.
    """

    # Reject booleans because they behave numerically but are not valid floats here.
    if isinstance(value, bool):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be numeric, not boolean."
        )

    # Reject non-numeric values.
    if not isinstance(value, int | float):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be numeric. "
            f"Received: {type(value).__name__}."
        )

    # Convert the value to a float.
    float_value = float(value)

    # Validate the lower bound when supplied.
    if minimum is not None and float_value < minimum:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be >= {minimum}. " f"Received: {float_value}."
        )

    # Validate the upper bound when supplied.
    if maximum is not None and float_value > maximum:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be <= {maximum}. " f"Received: {float_value}."
        )

    # Return the validated float.
    return float_value


def require_probability(value: object, *, field_path: str) -> float:
    """
    Validate that a config value is a probability in [0, 1].

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.

    Returns:
        Float probability.

    Raises:
        ConfigValidationError: If value is not numeric or outside [0, 1].
    """

    # Validate the value as a bounded float.
    return require_float(value, field_path=field_path, minimum=0.0, maximum=1.0)


def require_allowed_value(
    value: object,
    *,
    allowed_values: Sequence[str],
    field_path: str,
) -> str:
    """
    Validate that a string config value is one of a finite set of options.

    Args:
        value: Candidate value.
        allowed_values: Allowed string values.
        field_path: Field path used in error messages.

    Returns:
        Cleaned allowed string value.

    Raises:
        ConfigValidationError: If value is not a string or is unsupported.
    """

    # Validate the value as a non-empty string.
    cleaned_value = require_non_empty_string(value, field_path=field_path)

    # Convert allowed values into a set for lookup.
    allowed_value_set = set(allowed_values)

    # Reject unsupported values.
    if cleaned_value not in allowed_value_set:
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be one of: "
            f"{', '.join(allowed_values)}. Received: '{cleaned_value}'."
        )

    # Return the validated allowed value.
    return cleaned_value


def require_string_list(value: object, *, field_path: str) -> list[str]:
    """
    Validate that a config value is a list of non-empty strings.

    Args:
        value: Candidate value.
        field_path: Field path used in error messages.

    Returns:
        Cleaned list of strings.

    Raises:
        ConfigValidationError: If value is not a sequence of strings.
    """

    # Reject strings because they are sequences but not valid string lists.
    if isinstance(value, str):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be a list of strings, not a string."
        )

    # Reject non-sequence values.
    if not isinstance(value, Sequence):
        raise ConfigValidationError(
            f"Configuration field '{field_path}' must be a list of strings. "
            f"Received: {type(value).__name__}."
        )

    # Initialize the cleaned string list.
    cleaned_values: list[str] = []

    # Validate each list element.
    for index, item in enumerate(value):
        # Build the item field path.
        item_path = format_field_path(field_path, index)

        # Validate and append the cleaned item.
        cleaned_values.append(require_non_empty_string(item, field_path=item_path))

    # Return the cleaned list.
    return cleaned_values


def collect_validation_issues(
    *,
    errors: Sequence[ValidationIssue] | None = None,
    warnings: Sequence[ValidationIssue] | None = None,
) -> list[ValidationIssue]:
    """
    Combine validation errors and warnings into one ordered list.

    Args:
        errors: Optional error issues.
        warnings: Optional warning issues.

    Returns:
        Combined list of validation issues.
    """

    # Initialize the combined issue list.
    issues: list[ValidationIssue] = []

    # Add errors when provided.
    if errors is not None:
        issues.extend(errors)

    # Add warnings when provided.
    if warnings is not None:
        issues.extend(warnings)

    # Return the combined issue list.
    return issues


def raise_if_errors(issues: Sequence[ValidationIssue], *, context: str) -> None:
    """
    Raise a configuration error if any validation issue has error severity.

    Args:
        issues: Validation issues to inspect.
        context: Human-readable validation context.

    Raises:
        ConfigValidationError: If any issue has severity `error`.
    """

    # Keep only error-severity issues.
    errors = [issue for issue in issues if issue.severity == "error"]

    # Return when no errors exist.
    if not errors:
        return

    # Build readable error messages.
    messages = [f"{issue.field_path}: {issue.message}" for issue in errors]

    # Raise one combined validation error.
    raise ConfigValidationError(
        f"{context} failed validation with {len(errors)} error(s): " + " | ".join(messages)
    )


__all__ = [
    "ConfigValidationError",
    "ValidationIssue",
    "collect_validation_issues",
    "format_field_path",
    "raise_if_errors",
    "reject_unknown_keys",
    "require_allowed_value",
    "require_bool",
    "require_float",
    "require_int",
    "require_mapping",
    "require_non_empty_string",
    "require_probability",
    "require_required_keys",
    "require_string_list",
]
