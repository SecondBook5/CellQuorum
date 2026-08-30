# Pipeline step (order=350): ccc_viz — render cell-cell communication results.
"""CCC-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

import cellquorum.stages.cell_cell_communication.viz  # noqa: F401  (side-effect: registers methods)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage

_VIZ_CONFIG_KEYS = ("enabled", "top_k", "figure_formats", "dpi", "sources", "levels")
_DEFAULT_METHODS = ["dotplot_viz", "chord_viz", "sankey_viz", "network_viz", "summary_viz"]


@register_stage(
    name="ccc_viz", order=350, config_flag="ccc_viz", config_field="ccc_viz", category="ccc_viz"
)
class CccVizStage(MethodDispatchStage):
    """Render publication figures from the CCC stages' CSV/uns outputs."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "dotplot_viz")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        config = getattr(context, "config", None)
        viz_cfg = getattr(config, "ccc_viz", None)
        if viz_cfg is not None:
            for key in _VIZ_CONFIG_KEYS:
                if key not in augmented:
                    value = getattr(viz_cfg, key, None)
                    if value is not None:
                        augmented[key] = value
        if not augmented.get("methods") and "method" not in augmented:
            augmented["methods"] = [{"method": m} for m in _DEFAULT_METHODS]
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No-op: this stage writes figures, no obs/var postcondition."""


__all__ = ["CccVizStage"]
