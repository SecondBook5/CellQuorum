# Pipeline step (order=152): annotation_viz — figures that audit the annotation labels.
"""Annotation figure stage: dispatches the configured annotation viz method(s).

Runs after every stage that writes or revises the cell-type label (annotation 90,
annotation_consensus 130, adjudication 145, population_identity 150) and before subclustering
(155) narrows into single populations, so the figure audits the labels the rest of the run uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.annotation.viz  # noqa: F401  (side-effect: registers methods)
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

_VIZ_CONFIG_KEYS = (
    "enabled",
    "group_by",
    "highlight_expected",
    "vmax_percentile",
    "title",
    "figure_formats",
    "dpi",
)
_DEFAULT_METHODS = ["marker_dotplot", "lineage_dotplot"]


@register_stage(
    name="annotation_viz",
    order=152,
    config_flag="annotation_viz",
    config_field="annotation_viz",
    category="annotation_viz",
)
class AnnotationVizStage(DefaultMethodStage):
    """Render the marker dotplots from the annotation stage's own panel."""

    default_method = "marker_dotplot"

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "annotation_viz", None)
        if viz_cfg is not None:
            for key in _VIZ_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(viz_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        annotation = getattr(config, "annotation", None)
        if annotation is not None:
            augmented.setdefault("marker_panels", getattr(annotation, "marker_panels", {}) or {})
            augmented.setdefault("layer", getattr(annotation, "score_layer", None))
            if not augmented.get("group_by"):
                augmented["group_by"] = getattr(annotation, "key_added", None)
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": m} for m in _DEFAULT_METHODS]
        return augmented


__all__ = ["AnnotationVizStage"]
