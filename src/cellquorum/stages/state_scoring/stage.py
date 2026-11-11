# Pipeline step (order=170): state_scoring — score cell states via the configured method(s).
"""State-scoring stage: dispatches to the configured state-scoring method(s)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.state_scoring  # noqa: F401
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


@register_stage(
    name="state_scoring",
    order=170,
    config_flag="state_scoring",
    config_field="state_scoring",
    category="state_scoring",
)
class StateScoringStage(DefaultMethodStage):
    """Score cell-state programs with the configured method(s)."""

    default_method = "score_genes"

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        """Default to running both methods (obs scores + obsm AUCs) when unset."""
        augmented = dict(stage_config)
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "score_genes"}, {"method": "aucell"}]
        return augmented


__all__ = ["StateScoringStage"]
