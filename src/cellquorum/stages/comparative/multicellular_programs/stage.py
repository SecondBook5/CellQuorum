# Pipeline step (order=330): multicellular_programs — detect multicellular programs via DIALOGUE.
"""Multicellular programs stage: dispatches DIALOGUE."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.stages.comparative.multicellular_programs  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="multicellular_programs",
    order=330,
    config_flag="multicellular_programs",
    config_field="multicellular_programs",
    category="multicellular_programs",
)
class MulticellularProgramsStage(MethodDispatchStage):
    """Run the configured multicellular programs method: DIALOGUE."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "dialogue")

    def _validate_output(self, result: StageResult) -> None:
        """No-op: writes table artifacts only (no obs keys), no postcondition."""


__all__ = ["MulticellularProgramsStage"]
