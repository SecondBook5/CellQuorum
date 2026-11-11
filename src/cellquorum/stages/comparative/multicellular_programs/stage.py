# Pipeline step (order=330): multicellular_programs — detect multicellular programs via DIALOGUE.
"""Multicellular programs stage: dispatches DIALOGUE."""

from __future__ import annotations

import cellquorum.stages.comparative.multicellular_programs  # noqa: F401
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage


@register_stage(
    name="multicellular_programs",
    order=330,
    config_flag="multicellular_programs",
    config_field="multicellular_programs",
    category="multicellular_programs",
)
class MulticellularProgramsStage(DefaultMethodStage):
    """Run the configured multicellular programs method: DIALOGUE."""

    default_method = "dialogue"


__all__ = ["MulticellularProgramsStage"]
