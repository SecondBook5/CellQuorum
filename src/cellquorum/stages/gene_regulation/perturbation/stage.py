"""Perturbation stage: dispatches to the configured perturbation method."""

from __future__ import annotations

import cellquorum.stages.gene_regulation.perturbation  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="perturbation",
    order=280,
    config_flag="perturbation",
    config_field="perturbation",
    category="perturbation",
)
class PerturbationStage(MethodDispatchStage):
    """Run the configured in-silico perturbation (CellOracle) method."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "celloracle")

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes tables + figures."""


__all__ = ["PerturbationStage"]
