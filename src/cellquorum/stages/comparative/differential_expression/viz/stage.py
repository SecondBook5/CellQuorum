# Pipeline step (order=250): de_viz — render differential-expression results.
"""DE-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cellquorum.stages.comparative.differential_expression.viz  # noqa: F401  (side-effect: method registration)
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

_VIZ_CONFIG_KEYS = (
    "enabled",
    "fc_cut",
    "fdr_cut",
    "top_n_labels",
    "figure_formats",
    "dpi",
    "x_label",
    "case_color",
    "control_color",
    "highlight_genes",
)


@register_stage(
    name="de_viz", order=250, config_flag="de_viz", config_field="de_viz", category="de_viz"
)
class DeVizStage(DefaultMethodStage):
    """Render publication figures from the DE stage's CSV outputs."""

    default_method = "volcano_viz"

    def _augment_config(self, context: PipelineContext, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "de_viz", None)
        if viz_cfg is not None:
            for key in _VIZ_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(viz_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        # Bridge the design block so the volcano can label case/control direction.
        design = getattr(config, "design", None)
        if design is not None:
            if not augmented.get("case"):
                augmented["case"] = getattr(design, "case", None)
            if not augmented.get("control"):
                augmented["control"] = getattr(design, "control", None)
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": "volcano_viz"}]
        return augmented


__all__ = ["DeVizStage"]
