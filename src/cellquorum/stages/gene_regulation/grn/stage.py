# Pipeline step (order=270): grn — infer gene-regulatory networks via the configured method.
"""GRN stage: dispatches to the configured GRN method."""

from __future__ import annotations

import cellquorum.stages.gene_regulation.grn  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(name="grn", order=270, config_flag="grn", config_field="grn", category="grn")
class GrnStage(MethodDispatchStage):
    """Run the configured GRN (pySCENIC) method."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "pyscenic")

    def _augment_config(self, context: object, stage_config: dict) -> dict:
        augmented = dict(stage_config)
        # Bridge the design block so the radial GRN figure can label
        # "up in <case>" / "up in <control>" generically instead of a
        # hardcoded study-specific condition name (same pattern as de_viz).
        config = getattr(context, "config", None)
        design = getattr(config, "design", None)
        if design is not None:
            if not augmented.get("case"):
                augmented["case"] = getattr(design, "case", None)
            if not augmented.get("control"):
                augmented["control"] = getattr(design, "control", None)
        return augmented

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes tables + figures."""


__all__ = ["GrnStage"]
