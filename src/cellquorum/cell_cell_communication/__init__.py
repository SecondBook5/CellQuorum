"""Cell-cell communication (LR co-expression) stage package."""

from __future__ import annotations

from cellquorum.cell_cell_communication.config import CellCellCommunicationConfig
from cellquorum.cell_cell_communication.liana_method import LianaMethod
from cellquorum.cell_cell_communication.multinichenet_method import MultiNicheNetMethod
from cellquorum.cell_cell_communication.nichenet_method import NicheNetMethod
from cellquorum.cell_cell_communication.tensor_c2c_method import TensorCell2CellMethod
from cellquorum.methods.registry import METHOD_REGISTRY

for _method in (LianaMethod, TensorCell2CellMethod, MultiNicheNetMethod, NicheNetMethod):
    if not METHOD_REGISTRY.has("cell_cell_communication", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "CellCellCommunicationConfig",
    "LianaMethod",
    "TensorCell2CellMethod",
    "MultiNicheNetMethod",
    "NicheNetMethod",
]
