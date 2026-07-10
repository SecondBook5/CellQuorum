"""Exception type for CellQuorum data-contract violations."""

from __future__ import annotations

# Import the data error base class for contract-specific inheritance.
from cellquorum.core.exceptions import CellQuorumDataError


class CellQuorumContractError(CellQuorumDataError):
    """
    Report a data-contract violation at a stage boundary.

    Raised when an AnnData object handed into or out of a stage fails its
    declared contract: a missing layer/obs column/embedding (structural), a
    layer whose provenance tag does not match the required recipe (semantic),
    or a layer whose values are statistically inconsistent with its declared
    kind, e.g. integer-valued data tagged as log-normalized (statistical).
    """


__all__ = ["CellQuorumContractError"]
