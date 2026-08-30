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

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes tables + figures."""


__all__ = ["GrnStage"]
