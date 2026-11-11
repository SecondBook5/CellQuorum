# Pipeline step (order=180): discovery — discover de-novo programs via consensus NMF.
"""Discovery stage: de-novo program discovery via consensus NMF."""

from __future__ import annotations

import cellquorum.stages.discovery  # noqa: F401
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage


@register_stage(
    name="discovery",
    order=180,
    config_flag="discovery",
    config_field="discovery",
    category="discovery",
)
class DiscoveryStage(DefaultMethodStage):
    """Discover unbiased gene programs with the configured method (default NMF)."""

    default_method = "nmf"


__all__ = ["DiscoveryStage"]
