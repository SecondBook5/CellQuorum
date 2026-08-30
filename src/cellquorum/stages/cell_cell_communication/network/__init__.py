"""Cell-cell communication network (topology + curvature) stage package."""

from __future__ import annotations

from cellquorum.stages.cell_cell_communication.network.config import CCCNetworkConfig
from cellquorum.stages.cell_cell_communication.network.ricci_method import RicciMethod
from cellquorum.stages.cell_cell_communication.network.topology_method import TopologyMethod
from cellquorum.methods.registry import METHOD_REGISTRY

for _method in (TopologyMethod, RicciMethod):
    if not METHOD_REGISTRY.has("ccc_network", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = ["CCCNetworkConfig", "RicciMethod", "TopologyMethod"]
