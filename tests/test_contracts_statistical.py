"""Tests for value-level statistical sanity checks."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.statistical import (
    assert_integer_valued,
    assert_log_range,
    assert_non_integer_or_zero,
    assert_non_negative,
)


def test_integer_valued_passes_on_counts():
    X = sp.csr_matrix(np.array([[0, 3], [5, 0]], dtype=np.float32))
    assert_integer_valued(X, layer="counts")  # no raise


def test_integer_valued_raises_on_fractional():
    X = np.array([[0.0, 1.5], [2.0, 0.0]], dtype=np.float32)
    with pytest.raises(CellQuorumContractError, match="counts"):
        assert_integer_valued(X, layer="counts")


def test_non_integer_or_zero_raises_on_raw_counts_mislabeled_as_lognorm():
    # This is the exact lekc bug: integer counts sitting in a 'lognorm' layer.
    X = sp.csr_matrix(np.array([[0, 3, 12], [5, 0, 7]], dtype=np.float32))
    with pytest.raises(CellQuorumContractError, match="lognorm"):
        assert_non_integer_or_zero(X, layer="lognorm")


def test_non_integer_or_zero_passes_on_real_lognorm():
    X = np.array([[0.0, 0.73], [1.42, 0.0]], dtype=np.float32)
    assert_non_integer_or_zero(X, layer="lognorm")  # no raise


def test_non_negative_raises_on_negative():
    X = np.array([[0.0, -1.0]], dtype=np.float32)
    with pytest.raises(CellQuorumContractError):
        assert_non_negative(X, layer="lognorm")


def test_log_range_raises_on_unlogged():
    X = np.array([[0.0, 5000.0]], dtype=np.float32)
    with pytest.raises(CellQuorumContractError, match="exceeds"):
        assert_log_range(X, layer="lognorm", max_value=30.0)


def test_log_range_passes_on_logged():
    X = np.array([[0.0, 4.2]], dtype=np.float32)
    assert_log_range(X, layer="lognorm", max_value=30.0)
