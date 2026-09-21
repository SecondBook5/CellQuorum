# Pipeline step (order=148): metacells -- partition cells into near-homogeneous
# metacells via MetaCell2, feeding atlas-level QC/identity figures.
"""Metacells stage: dispatches to the configured metacell method."""

from __future__ import annotations

import cellquorum.stages.metacells  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import CellScope, CellScopePolicy, register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="metacells",
    order=148,
    config_flag="metacells",
    config_field="metacells",
    category="metacells",
    cell_scope=CellScopePolicy(
        fit_scope=CellScope.ALL,
        reason=(
            "This stage's whole purpose is to characterize unclassified/outlier "
            "structure across the object -- restricting its input to only "
            "QC-core cells would hide exactly the contamination it exists to "
            "surface, so it deliberately partitions every cell, including "
            "QC-flagged ones, rather than only the biological-reference subset."
        ),
    ),
)
class MetacellsStage(MethodDispatchStage):
    """Run the configured metacell (MetaCell2) method."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "metacells")

    def _validate_output(self, result: StageResult) -> None:
        """No structural postcondition; the stage writes a metacell_id obs column plus
        tables/figures."""


__all__ = ["MetacellsStage"]
