"""Multicellular programs stage: dispatches DIALOGUE."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.multicellular_programs  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class MulticellularProgramsStage(MethodDispatchStage):
    """Run the configured multicellular programs method: DIALOGUE."""

    name = "multicellular_programs"
    stage_category = "multicellular_programs"

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "dialogue")

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes table artifacts + obs keys, no postcondition."""


__all__ = ["MulticellularProgramsStage"]
