# Pipeline step (order=230): enrichment — score pathway/gene-set enrichment.
"""Enrichment stage: dispatches to the configured enrichment method(s)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.comparative.enrichment  # noqa: F401
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.context_access import with_study_contrast
from cellquorum.methods.stage_base import DefaultMethodStage
from cellquorum.stages.comparative.enrichment.config import default_enrichment_methods

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext


@register_stage(
    name="enrichment",
    order=230,
    config_flag="enrichment",
    config_field="enrichment",
    category="enrichment",
)
class EnrichmentStage(DefaultMethodStage):
    """Run the configured enrichment / pathway-activity method(s)."""

    default_method = "gsea"

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        """Resolve the contrast, organism and default enrichment methods."""
        augmented = with_study_contrast(context, stage_config)
        config = getattr(context, "config", None)

        # Organism is always bridged (gene-set fetches are species-specific).
        organism = getattr(getattr(config, "project", None), "organism", None) or getattr(
            config, "organism", None
        )
        if organism is not None and not augmented.get("organism"):
            augmented["organism"] = organism

        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = default_enrichment_methods(
                by_cell_type=bool(
                    getattr(getattr(config, "differential_expression", None), "cell_type_col", None)
                )
            )

        return augmented


__all__ = ["EnrichmentStage"]
