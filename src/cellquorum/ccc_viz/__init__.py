"""CCC visualization stage package."""

from __future__ import annotations

from cellquorum.ccc_viz.chord_viz import ChordVizMethod
from cellquorum.ccc_viz.config import CccVizConfig
from cellquorum.ccc_viz.dotplot_viz import DotplotVizMethod
from cellquorum.ccc_viz.network_viz import NetworkVizMethod
from cellquorum.ccc_viz.sankey_viz import SankeyVizMethod
from cellquorum.ccc_viz.summary_viz import SummaryVizMethod
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
