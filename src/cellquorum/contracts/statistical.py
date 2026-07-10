"""Cheap value-level checks that catch mislabeled expression matrices.

These checks exist because a layer name lies. The lekc project stored raw
integer counts in a layer named ``lognorm``; every downstream statistic was
wrong. ``assert_non_integer_or_zero`` is the specific guard: log-normalized
data is (essentially) never all-integer, so an all-integer 'lognorm' layer is
raised on immediately.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cellquorum.contracts.exceptions import CellQuorumContractError


def _to_sample(X: np.ndarray | sp.spmatrix, n: int = 10000) -> np.ndarray:
    """
    Return a flat float sample of the matrix's values for cheap checks.

    For sparse matrices only stored (nonzero) values are sampled, which is both
    faster and the right domain for integer-ness / log-range checks. The sample
    is deterministic (a head slice, no RNG) so tests and reruns are stable.

    Args:
            X: Expression matrix, dense or sparse.
            n: Maximum number of values to inspect.

    Returns:
            1-D float64 array of up to ``n`` values.
    """

    # Extract stored values for sparse, or a raveled dense view.
    if sp.issparse(X):
        values = X.data
    else:
        values = np.asarray(X).ravel()

    # Deterministically cap the sample size for speed on large matrices.
    if values.size > n:
        values = values[:n]

    # Return as float64 for uniform downstream arithmetic.
    return values.astype(np.float64, copy=False)


def assert_non_negative(X: np.ndarray | sp.spmatrix, *, layer: str) -> None:
    """
    Assert all values are >= 0.

    Args:
            X: Expression matrix.
            layer: Layer name for the error message.

    Raises:
            CellQuorumContractError: If any value is negative.
    """

    # Sample and check the minimum.
    sample = _to_sample(X)
    if sample.size and sample.min() < 0:
        raise CellQuorumContractError(
            f"Layer '{layer}' contains negative values (min={sample.min():.4g}); "
            "expression layers must be non-negative."
        )


def assert_integer_valued(X: np.ndarray | sp.spmatrix, *, layer: str) -> None:
    """
    Assert every value is a non-negative integer (a valid counts layer).

    Args:
            X: Expression matrix.
            layer: Layer name for the error message.

    Raises:
            CellQuorumContractError: If any value is negative or non-integer.
    """

    # Sample once and reuse for both checks.
    sample = _to_sample(X)
    if sample.size == 0:
        return

    # Reject negatives.
    if sample.min() < 0:
        raise CellQuorumContractError(
            f"Layer '{layer}' declared as counts but contains negatives (min={sample.min():.4g})."
        )

    # Reject fractional values.
    if not np.allclose(sample, np.round(sample)):
        raise CellQuorumContractError(
            f"Layer '{layer}' declared as counts but contains non-integer values."
        )


def assert_non_integer_or_zero(X: np.ndarray | sp.spmatrix, *, layer: str) -> None:
    """
    Assert the layer is not the all-integer signature of raw counts.

    Log-normalized expression is effectively never all-integer. If every
    sampled nonzero value is an integer, the layer almost certainly holds raw
    counts under a normalized name — the exact lekc bug.

    Args:
            X: Expression matrix.
            layer: Layer name for the error message.

    Raises:
            CellQuorumContractError: If all nonzero values are integers.
    """

    # Sample nonzero values (zeros are integers in both cases; ignore them).
    sample = _to_sample(X)
    nonzero = sample[sample != 0]
    if nonzero.size == 0:
        return

    # All-integer nonzero values => raw counts masquerading as normalized.
    if np.allclose(nonzero, np.round(nonzero)):
        raise CellQuorumContractError(
            f"Layer '{layer}' is tagged as log-normalized but all values are integers — "
            "this is the signature of raw counts stored under a normalized layer name."
        )


def assert_log_range(
    X: np.ndarray | sp.spmatrix,
    *,
    layer: str,
    max_value: float = 30.0,
) -> None:
    """
    Assert log-normalized values do not exceed a sane ceiling.

    Un-logged CP10k/CPM data reaches into the thousands; genuine log1p values
    almost never exceed ~15. A ceiling of 30 catches un-logged data without
    false positives.

    Args:
            X: Expression matrix.
            layer: Layer name for the error message.
            max_value: Maximum tolerated value.

    Raises:
            CellQuorumContractError: If any value exceeds ``max_value``.
    """

    # Sample and check the maximum.
    sample = _to_sample(X)
    if sample.size and sample.max() > max_value:
        raise CellQuorumContractError(
            f"Layer '{layer}' max value {sample.max():.4g} exceeds log ceiling {max_value}; "
            "values look un-logged (raw CP10k/CPM?)."
        )


__all__ = [
    "assert_integer_valued",
    "assert_log_range",
    "assert_non_integer_or_zero",
    "assert_non_negative",
]
