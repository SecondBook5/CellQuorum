"""CCC-visualization stage: dispatches the configured viz method(s)."""

from __future__ import annotations

import cellquorum.ccc_viz  # noqa: F401  (side-effect: registers methods)
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage

_VIZ_CONFIG_KEYS = ("enabled", "top_k", "figure_formats", "dpi", "sources", "levels")
_DEFAULT_METHODS = ["dotplot_viz", "chord_viz", "sankey_viz", "network_viz", "summary_viz"]


class CccVizStage(MethodDispatchStage):
    """Render publication figures from the CCC stages' CSV/uns outputs."""

    name = "ccc_viz"
    stage_category = "ccc_viz"

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
