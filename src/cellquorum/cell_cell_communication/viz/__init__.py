"""CCC visualization stage package."""

from __future__ import annotations

from cellquorum.cell_cell_communication.viz._viz_methods import (
    ChordVizMethod,
    DotplotVizMethod,
    NetworkVizMethod,
    SankeyVizMethod,
    SummaryVizMethod,
)
from cellquorum.cell_cell_communication.viz.config import CccVizConfig
from cellquorum.methods.registry import METHOD_REGISTRY

for _method in (
    DotplotVizMethod,
    ChordVizMethod,
    SankeyVizMethod,
    NetworkVizMethod,
    SummaryVizMethod,
):
    if not METHOD_REGISTRY.has("ccc_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = [
    "CccVizConfig",
    "ChordVizMethod",
    "DotplotVizMethod",
    "NetworkVizMethod",
    "SankeyVizMethod",
    "SummaryVizMethod",
]
