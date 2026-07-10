"""Tests for the contract exception type."""

from __future__ import annotations

# Import pytest for exception-catching assertions.
import pytest

# Import the contract exception and its base classes for inheritance tests.
from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.core.exceptions import CellQuorumDataError, CellQuorumError


def test_contract_error_is_data_error():
    err = CellQuorumContractError("bad layer")
    assert isinstance(err, CellQuorumDataError)
    assert isinstance(err, CellQuorumError)
    assert "bad layer" in str(err)


def test_contract_error_catchable_as_base():
    with pytest.raises(CellQuorumError):
        raise CellQuorumContractError("boom")
