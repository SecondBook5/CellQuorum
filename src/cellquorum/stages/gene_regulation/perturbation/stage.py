# Pipeline step (order=280): perturbation — run in-silico perturbation via the configured method.
"""Perturbation stage: dispatches to the configured perturbation method."""

from __future__ import annotations

import cellquorum.stages.gene_regulation.perturbation  # noqa: F401  (registers the method as a side effect)
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage


@register_stage(
    name="perturbation",
    order=280,
    config_flag="perturbation",
    config_field="perturbation",
    category="perturbation",
)
class PerturbationStage(DefaultMethodStage):
    """Run the configured CellOracle or scTenifoldKnk perturbation method."""

    default_method = "celloracle"


__all__ = ["PerturbationStage"]
