"""State-scoring visualization stage package."""

from __future__ import annotations

from cellquorum.methods.registry import METHOD_REGISTRY
from cellquorum.stages.state_scoring.viz._viz_methods import StateScoreViolinMethod
from cellquorum.stages.state_scoring.viz.config import StateScoringVizConfig

for _method in (StateScoreViolinMethod,):
    if not METHOD_REGISTRY.has("state_scoring_viz", _method.name):
        METHOD_REGISTRY.register(_method)

__all__ = ["StateScoringVizConfig", "StateScoreViolinMethod"]
