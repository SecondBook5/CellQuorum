"""Enrichment-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

# Import the package so the methods register themselves as a side effect.
import cellquorum.enrichment_viz  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage

# Keys bridged from config.enrichment_viz into each dispatched viz method's config.
_VIZ_CONFIG_KEYS = ("enabled", "top_k", "figure_formats", "dpi", "collections", "resources")


class EnrichmentVizStage(MethodDispatchStage):
    """Render publication figures from the enrichment stage's CSV outputs."""

    name = "enrichment_viz"
    stage_category = "enrichment_viz"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "gsea_viz")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        """Bridge config.enrichment_viz into the method config; inject method list."""
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "enrichment_viz", None)
        if viz_cfg is not None:
            for key in _VIZ_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(viz_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [
                {"method": "gsea_viz"},
                {"method": "ora_viz"},
                {"method": "gsva_viz"},
                {"method": "activity_viz"},
            ]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["EnrichmentVizStage"]
