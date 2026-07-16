"""Fail-loud data contracts for CellQuorum stage boundaries."""

from __future__ import annotations

from cellquorum.contracts.data_contract import DataContract
from cellquorum.contracts.exceptions import CellQuorumContractError
from cellquorum.contracts.labels import LabelContract
from cellquorum.contracts.layer_tags import (
    get_layer_tag,
    get_normalization_recipe,
    set_layer_tag,
)
from cellquorum.contracts.magic_guard import assert_not_imputed
from cellquorum.contracts.statistical import assert_statistical_input

__all__ = [
    "CellQuorumContractError",
    "DataContract",
    "LabelContract",
    "assert_not_imputed",
    "assert_statistical_input",
    "get_layer_tag",
    "get_normalization_recipe",
    "set_layer_tag",
]
