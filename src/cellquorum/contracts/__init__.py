"""Fail-loud data contracts for CellQuorum stage boundaries."""

from __future__ import annotations

# Import the contract violation exception for public export.
from cellquorum.contracts.exceptions import CellQuorumContractError

__all__ = ["CellQuorumContractError"]
