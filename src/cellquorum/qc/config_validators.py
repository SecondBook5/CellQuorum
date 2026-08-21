"""Reusable field-coercion helpers for the QC configuration models.

The QC configuration models in :mod:`cellquorum.qc.config` share a small set of
input-coercion patterns — "optional non-negative integer", "bounded float",
"list of non-empty strings", "stripped non-empty string" — that were previously
copied, near-identically, across a dozen ``@field_validator`` methods. These
free functions capture each pattern once so the config models stay declarative
(fields plus model-level constraints) and the coercion logic lives in one
individually testable place.

Every helper takes the raw candidate value plus the exact error-message text the
calling field uses, so the observable validation behavior — which errors are
raised, with which wording, and what value is returned — is unchanged from the
per-model validators these replace. The helpers depend only on the standard
library, so importing them from :mod:`cellquorum.qc.config` introduces no import
cycle (``qc.thresholds`` already imports ``qc.config``).
"""

from __future__ import annotations


def _type_name(value: object) -> str:
    """Return the runtime type name used in "Received: ..." error suffixes."""

    # Mirror the ``type(value).__name__`` interpolation the validators used.
    return type(value).__name__


def coerce_percent_top(value: object) -> list[int]:
    """
    Coerce and validate a ``percent_top`` setting.

    Args:
        value: Candidate percent_top value.

    Returns:
        Sorted list of unique positive integer ranks.

    Raises:
        ValueError: If the value is not a non-empty list of positive integers.
    """

    # Reject a missing percent_top list.
    if value is None:
        raise ValueError("percent_top cannot be None.")

    # Reject strings because they are sequences but not valid integer lists.
    if isinstance(value, str):
        raise ValueError("percent_top must be a list of positive integers, not a string.")

    # Reject non-list and non-tuple values.
    if not isinstance(value, list | tuple):
        raise ValueError(
            f"percent_top must be a list of positive integers. Received: {_type_name(value)}."
        )

    # Reject empty percent_top lists.
    if not value:
        raise ValueError("percent_top must contain at least one positive integer.")

    # Initialize the cleaned percent_top list.
    cleaned_values: list[int] = []

    # Iterate over each candidate value.
    for item in value:
        # Reject booleans because bool is a subclass of int.
        if isinstance(item, bool):
            raise ValueError("percent_top values must be integers, not booleans.")

        # Reject non-integer values.
        if not isinstance(item, int):
            raise ValueError(f"percent_top values must be integers. Received: {_type_name(item)}.")

        # Reject non-positive values.
        if item <= 0:
            raise ValueError("percent_top values must be > 0.")

        # Store the cleaned integer.
        cleaned_values.append(item)

    # Return the sorted unique values for deterministic metric names.
    return sorted(set(cleaned_values))


def coerce_stripped_string(
    value: object,
    *,
    optional: bool,
    type_message: str,
    empty_message: str,
) -> str | None:
    """
    Coerce a candidate into a stripped, non-empty string.

    Args:
        value: Candidate string value.
        optional: Whether ``None`` is preserved (``True``) or rejected as a
            non-string (``False``).
        type_message: Message prefix raised when the value is not a string; the
            runtime type is appended as `` Received: <type>.``.
        empty_message: Message raised when the stripped value is empty.

    Returns:
        The stripped string, or ``None`` when ``optional`` and the value is None.

    Raises:
        ValueError: If the value is a non-string (or empty after stripping).
    """

    # Preserve absent values only when the field is optional.
    if optional and value is None:
        return None

    # Reject non-string values.
    if not isinstance(value, str):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    # Strip harmless whitespace.
    cleaned_value = value.strip()

    # Reject empty strings.
    if not cleaned_value:
        raise ValueError(empty_message)

    # Return the cleaned string.
    return cleaned_value


def coerce_string_list(
    value: object,
    *,
    not_a_list_message: str,
    wrong_container_message: str,
    item_type_message: str,
    empty_item_message: str,
) -> list[str]:
    """
    Coerce a candidate into a list of stripped, non-empty strings.

    Args:
        value: Candidate string list (``None`` becomes an empty list).
        not_a_list_message: Message raised when a bare string is supplied.
        wrong_container_message: Message prefix raised for non-list/tuple values;
            the runtime type is appended as `` Received: <type>.``.
        item_type_message: Message prefix raised for a non-string entry; the
            runtime type is appended as `` Received: <type>.``.
        empty_item_message: Message raised when an entry is empty after stripping.

    Returns:
        The cleaned list of strings.

    Raises:
        ValueError: If the value is not a list of non-empty strings.
    """

    # Return an empty list when an optional list is omitted.
    if value is None:
        return []

    # Reject a single string because callers must provide a list explicitly.
    if isinstance(value, str):
        raise ValueError(not_a_list_message)

    # Reject non-list and non-tuple values.
    if not isinstance(value, list | tuple):
        raise ValueError(f"{wrong_container_message} Received: {_type_name(value)}.")

    # Initialize the cleaned list.
    cleaned_values: list[str] = []

    # Iterate over candidate entries.
    for item in value:
        # Reject non-string entries.
        if not isinstance(item, str):
            raise ValueError(f"{item_type_message} Received: {_type_name(item)}.")

        # Strip harmless whitespace.
        cleaned_item = item.strip()

        # Reject empty entries.
        if not cleaned_item:
            raise ValueError(empty_item_message)

        # Store the cleaned entry.
        cleaned_values.append(cleaned_item)

    # Return the cleaned values.
    return cleaned_values


def coerce_non_negative_int(
    value: object,
    *,
    optional: bool,
    bool_message: str,
    type_message: str,
    negative_message: str,
) -> int | None:
    """
    Coerce a candidate into a non-negative integer.

    Args:
        value: Candidate integer value.
        optional: Whether ``None`` is preserved (``True``) or rejected.
        bool_message: Message raised when the value is a boolean.
        type_message: Message prefix raised for a non-integer value; the runtime
            type is appended as `` Received: <type>.``.
        negative_message: Message raised when the value is negative.

    Returns:
        The validated integer, or ``None`` when ``optional`` and the value is None.

    Raises:
        ValueError: If the value is boolean, non-integer, or negative.
    """

    # Preserve absent values only when the field is optional.
    if optional and value is None:
        return None

    # Reject booleans because bool is a subclass of int.
    if isinstance(value, bool):
        raise ValueError(bool_message)

    # Reject non-integer values.
    if not isinstance(value, int):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    # Reject negative values.
    if value < 0:
        raise ValueError(negative_message)

    # Return the validated integer.
    return value


def coerce_float_in_range(
    value: object,
    *,
    optional: bool,
    low: float,
    high: float,
    bool_message: str,
    type_message: str,
    range_message: str,
) -> float | None:
    """
    Coerce a candidate into a float within the inclusive ``[low, high]`` range.

    Args:
        value: Candidate numeric value.
        optional: Whether ``None`` is preserved (``True``) or rejected.
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        bool_message: Message raised when the value is a boolean.
        type_message: Message prefix raised for a non-numeric value; the runtime
            type is appended as `` Received: <type>.``.
        range_message: Message raised when the value falls outside the range.

    Returns:
        The validated float, or ``None`` when ``optional`` and the value is None.

    Raises:
        ValueError: If the value is boolean, non-numeric, or out of range.
    """

    # Preserve absent values only when the field is optional.
    if optional and value is None:
        return None

    # Reject booleans because they behave numerically but are invalid here.
    if isinstance(value, bool):
        raise ValueError(bool_message)

    # Reject non-numeric values.
    if not isinstance(value, int | float):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    # Convert the value to float.
    float_value = float(value)

    # Reject values outside the valid range.
    if float_value < low or float_value > high:
        raise ValueError(range_message)

    # Return the validated float.
    return float_value


def coerce_positive_float(
    value: object,
    *,
    bool_message: str,
    type_message: str,
    nonpositive_message: str,
) -> float:
    """
    Coerce a candidate into a strictly positive float.

    Args:
        value: Candidate numeric value.
        bool_message: Message raised when the value is a boolean.
        type_message: Message prefix raised for a non-numeric value; the runtime
            type is appended as `` Received: <type>.``.
        nonpositive_message: Message raised when the value is not strictly positive.

    Returns:
        The validated positive float.

    Raises:
        ValueError: If the value is boolean, non-numeric, or non-positive.
    """

    # Reject booleans because they behave numerically but are invalid here.
    if isinstance(value, bool):
        raise ValueError(bool_message)

    # Reject non-numeric values.
    if not isinstance(value, int | float):
        raise ValueError(f"{type_message} Received: {_type_name(value)}.")

    # Convert the value to float.
    float_value = float(value)

    # Reject non-positive values.
    if float_value <= 0.0:
        raise ValueError(nonpositive_message)

    # Return the validated float.
    return float_value


__all__ = [
    "coerce_float_in_range",
    "coerce_non_negative_int",
    "coerce_percent_top",
    "coerce_positive_float",
    "coerce_string_list",
    "coerce_stripped_string",
]
