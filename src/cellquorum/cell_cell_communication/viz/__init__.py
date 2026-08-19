"""CCC visualization stage package."""

from __future__ import annotations

from cellquorum.cell_cell_communication.viz.chord_viz import ChordVizMethod
from cellquorum.cell_cell_communication.viz.config import CccVizConfig
from cellquorum.cell_cell_communication.viz.dotplot_viz import DotplotVizMethod
from cellquorum.cell_cell_communication.viz.network_viz import NetworkVizMethod
from cellquorum.cell_cell_communication.viz.sankey_viz import SankeyVizMethod
from cellquorum.cell_cell_communication.viz.summary_viz import SummaryVizMethod
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
