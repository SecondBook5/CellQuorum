"""Direct unit tests for the shared QC field-coercion helpers.

The QC config models delegate their ``@field_validator`` bodies to these free
functions (consolidation #187, Move 2). The models' own tests exercise them
transitively, but these tests pin the reusable helper contract directly — the
happy path and every rejection branch — so future callers can rely on it.
"""

from __future__ import annotations

import re

import pytest

from cellquorum.qc.config_validators import (
    coerce_float_in_range,
    coerce_non_negative_int,
    coerce_percent_top,
    coerce_positive_float,
    coerce_string_list,
    coerce_stripped_string,
)


class TestCoercePercentTop:
    def test_sorts_and_dedupes_positive_ints(self):
        assert coerce_percent_top([50, 20, 20, 100]) == [20, 50, 100]

    def test_accepts_tuple(self):
        assert coerce_percent_top((20,)) == [20]

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (None, "percent_top cannot be None."),
            ("20", "percent_top must be a list of positive integers, not a string."),
            (20, "percent_top must be a list of positive integers. Received: int."),
            ([], "percent_top must contain at least one positive integer."),
            ([True], "percent_top values must be integers, not booleans."),
            ([1.5], "percent_top values must be integers. Received: float."),
            ([0], "percent_top values must be > 0."),
            ([-3], "percent_top values must be > 0."),
        ],
    )
    def test_rejections(self, value, message):
        with pytest.raises(ValueError, match=r"^" + re.escape(message) + r"$"):
            coerce_percent_top(value)


class TestCoerceStrippedString:
    def test_strips_and_returns(self):
        assert (
            coerce_stripped_string("  layer  ", optional=True, type_message="t", empty_message="e")
            == "layer"
        )

    def test_optional_none_passthrough(self):
        assert (
            coerce_stripped_string(None, optional=True, type_message="t", empty_message="e") is None
        )

    def test_required_none_is_type_error(self):
        with pytest.raises(ValueError, match="must be a string. Received: NoneType."):
            coerce_stripped_string(
                None,
                optional=False,
                type_message="mito_metric must be a string.",
                empty_message="mito_metric cannot be empty.",
            )

    def test_non_string_type_message(self):
        with pytest.raises(ValueError, match="layer must be a string. Received: int."):
            coerce_stripped_string(
                5, optional=True, type_message="layer must be a string.", empty_message="e"
            )

    def test_empty_message(self):
        with pytest.raises(ValueError, match="^cannot be empty$"):
            coerce_stripped_string(
                "   ", optional=True, type_message="t", empty_message="cannot be empty"
            )


_LIST_MESSAGES = dict(
    not_a_list_message="must be a list, not a string.",
    wrong_container_message="must be a list of strings.",
    item_type_message="entries must be strings.",
    empty_item_message="entries cannot be empty.",
)


class TestCoerceStringList:
    def test_none_becomes_empty_list(self):
        assert coerce_string_list(None, **_LIST_MESSAGES) == []

    def test_strips_entries(self):
        assert coerce_string_list([" a ", "b"], **_LIST_MESSAGES) == ["a", "b"]

    def test_preserves_order_and_duplicates(self):
        # Unlike percent_top, string lists are not sorted or de-duplicated.
        assert coerce_string_list(["b", "a", "b"], **_LIST_MESSAGES) == ["b", "a", "b"]

    def test_rejects_bare_string(self):
        with pytest.raises(ValueError, match="^must be a list, not a string.$"):
            coerce_string_list("abc", **_LIST_MESSAGES)

    def test_rejects_wrong_container(self):
        with pytest.raises(ValueError, match="must be a list of strings. Received: int."):
            coerce_string_list(5, **_LIST_MESSAGES)

    def test_rejects_non_string_item(self):
        with pytest.raises(ValueError, match="entries must be strings. Received: int."):
            coerce_string_list(["a", 3], **_LIST_MESSAGES)

    def test_rejects_empty_item(self):
        with pytest.raises(ValueError, match="^entries cannot be empty.$"):
            coerce_string_list(["a", "  "], **_LIST_MESSAGES)


_INT_MESSAGES = dict(
    bool_message="cannot be boolean.",
    type_message="must be an integer.",
    negative_message="must be >= 0.",
)


class TestCoerceNonNegativeInt:
    def test_passes_non_negative(self):
        assert coerce_non_negative_int(0, optional=True, **_INT_MESSAGES) == 0
        assert coerce_non_negative_int(200, optional=False, **_INT_MESSAGES) == 200

    def test_optional_none_passthrough(self):
        assert coerce_non_negative_int(None, optional=True, **_INT_MESSAGES) is None

    def test_required_none_is_type_error(self):
        with pytest.raises(ValueError, match="^must be an integer. Received: NoneType.$"):
            coerce_non_negative_int(None, optional=False, **_INT_MESSAGES)

    def test_rejects_bool_before_int(self):
        with pytest.raises(ValueError, match="^cannot be boolean.$"):
            coerce_non_negative_int(True, optional=True, **_INT_MESSAGES)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match="must be an integer. Received: float."):
            coerce_non_negative_int(1.0, optional=True, **_INT_MESSAGES)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="^must be >= 0.$"):
            coerce_non_negative_int(-1, optional=True, **_INT_MESSAGES)


_RANGE_MESSAGES = dict(
    bool_message="cannot be boolean.",
    type_message="must be numeric.",
    range_message="must be between 0 and 1.",
)


class TestCoerceFloatInRange:
    def test_passes_and_converts_to_float(self):
        result = coerce_float_in_range(1, optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES)
        assert result == 1.0
        assert isinstance(result, float)

    def test_accepts_bounds_inclusive(self):
        assert coerce_float_in_range(0, optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES) == 0.0
        assert (
            coerce_float_in_range(100, optional=True, low=0.0, high=100.0, **_RANGE_MESSAGES)
            == 100.0
        )

    def test_optional_none_passthrough(self):
        assert (
            coerce_float_in_range(None, optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES) is None
        )

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="^cannot be boolean.$"):
            coerce_float_in_range(True, optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="must be numeric. Received: str."):
            coerce_float_in_range("x", optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="^must be between 0 and 1.$"):
            coerce_float_in_range(1.5, optional=True, low=0.0, high=1.0, **_RANGE_MESSAGES)


_POS_MESSAGES = dict(
    bool_message="cannot be boolean.",
    type_message="must be numeric.",
    nonpositive_message="must be > 0.",
)


class TestCoercePositiveFloat:
    def test_passes_and_converts(self):
        result = coerce_positive_float(5, **_POS_MESSAGES)
        assert result == 5.0
        assert isinstance(result, float)

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="^cannot be boolean.$"):
            coerce_positive_float(True, **_POS_MESSAGES)

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="must be numeric. Received: str."):
            coerce_positive_float("x", **_POS_MESSAGES)

    def test_rejects_zero_and_negative(self):
        with pytest.raises(ValueError, match="^must be > 0.$"):
            coerce_positive_float(0, **_POS_MESSAGES)
        with pytest.raises(ValueError, match="^must be > 0.$"):
            coerce_positive_float(-2.0, **_POS_MESSAGES)
