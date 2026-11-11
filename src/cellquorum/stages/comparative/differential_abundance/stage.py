# Pipeline step (order=220): differential_abundance — test abundance shifts (DA methods).
"""Differential-abundance stage: dispatches to the configured DA method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.comparative.differential_abundance  # noqa: F401
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import with_study_contrast
from cellquorum.methods.stage_base import DefaultMethodStage

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


@register_stage(
    name="differential_abundance",
    order=220,
    config_flag="differential_abundance",
    config_field="differential_abundance",
    category="differential_abundance",
)
class DifferentialAbundanceStage(DefaultMethodStage):
    """Run the configured differential-abundance method(s)."""

    default_method = "milo"

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        """Resolve the primary contrast and the default abundance methods."""

        augmented = with_study_contrast(context, stage_config)

        # Inject the default methods list if the config is bare (no methods, no method).
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [
                {"method": "milo"},
                {"method": "sccoda"},
                {"method": "propeller"},
                {"method": "proportion_ttest"},
            ]

        return augmented
