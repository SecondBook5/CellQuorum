"""Tests for the one canonical string form of a grouping label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellquorum.core.labels import as_label_strings


def test_an_integral_float_label_loses_the_decimal_point() -> None:
    """A subcluster column is float64 because NaN forced it; "1.0" is not a state name."""

    labels = as_label_strings(pd.Series([1.0, 2.0, np.nan, 8.0]))

    assert list(labels[:2]) == ["1", "2"]
    assert labels.iloc[3] == "8"


def test_missing_stays_missing() -> None:
    """The count-based methods exclude unlabelled cells; they have to be able to see them."""

    labels = as_label_strings(pd.Series([1.0, np.nan]))

    assert labels.isna().tolist() == [False, True]


def test_a_genuinely_fractional_label_keeps_its_form() -> None:
    """Only integral floats are rewritten -- a 0.5 resolution label is not an integer."""

    labels = as_label_strings(pd.Series([1.5, 2.0]))

    assert list(labels) == ["1.5", "2"]


def test_strings_and_categoricals_pass_through_unchanged() -> None:
    """The common case is already canonical and must not be disturbed."""

    assert list(as_label_strings(pd.Series(["LEC", "BEC"]))) == ["LEC", "BEC"]

    categorical = pd.Series(pd.Categorical(["Capillary", "Valve", "Capillary"]))
    assert list(as_label_strings(categorical)) == ["Capillary", "Valve", "Capillary"]


def test_a_categorical_of_integers_renders_without_a_decimal_point() -> None:
    """Leiden labels arrive as an integer-valued categorical."""

    categorical = pd.Series(pd.Categorical([0, 1, 1], categories=[0, 1]))

    assert list(as_label_strings(categorical)) == ["0", "1", "1"]


def test_numpy_integers_render_as_integers() -> None:
    """np.int64 is not an int subclass, so it needs the same treatment."""

    labels = as_label_strings(pd.Series(np.array([3, 4], dtype=np.int64)))

    assert list(labels) == ["3", "4"]


def test_the_result_is_object_dtype_so_a_merge_key_is_stable() -> None:
    """Two frames joined on a label must not disagree about the key's dtype."""

    left = as_label_strings(pd.Series([1.0, 2.0]))
    right = as_label_strings(pd.Series(["1", "2"]))

    assert left.dtype == right.dtype == object
    assert list(left) == list(right)
