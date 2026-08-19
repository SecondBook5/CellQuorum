"""Fail-loud data contracts for CellQuorum stage boundaries."""

from __future__ import annotations

from cellquorum.core.contracts.data_contract import DataContract
from cellquorum.core.contracts.exceptions import CellQuorumContractError
from cellquorum.core.contracts.labels import LabelContract
from cellquorum.core.contracts.layer_tags import (
    get_layer_tag,
    get_normalization_recipe,
    set_layer_tag,
)
from cellquorum.core.contracts.magic_guard import assert_not_imputed
from cellquorum.core.contracts.statistical import assert_statistical_input

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
