"""DE-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

import cellquorum.differential_expression.viz  # noqa: F401  (side-effect: method registration)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

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
)


@register_stage(
    name="de_viz", order=250, config_flag="de_viz", config_field="de_viz", category="de_viz"
)
class DeVizStage(MethodDispatchStage):
    """Render publication figures from the DE stage's CSV outputs."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "volcano_viz")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
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

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["DeVizStage"]
