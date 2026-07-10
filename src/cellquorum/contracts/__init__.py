"""Fail-loud data contracts for CellQuorum stage boundaries."""

from __future__ import annotations

from cellquorum.contracts.data_contract import DataContract
from cellquorum.contracts.exceptions import CellQuorumContractError

__all__ = ["CellQuorumContractError", "DataContract"]
