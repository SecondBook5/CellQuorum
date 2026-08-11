"""Perturbation stage: dispatches to the configured perturbation method."""

from __future__ import annotations

import cellquorum.perturbation  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class PerturbationStage(MethodDispatchStage):
    """Run the configured in-silico perturbation (CellOracle) method."""

    name = "perturbation"
    stage_category = "perturbation"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "celloracle")

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes tables + figures."""


__all__ = ["PerturbationStage"]
