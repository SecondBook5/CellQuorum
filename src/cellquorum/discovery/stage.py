"""Discovery stage: de-novo program discovery via consensus NMF."""

from __future__ import annotations

# Import the package so the method registers itself as a side effect.
import cellquorum.discovery  # noqa: F401
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import MethodDispatchStage


@register_stage(
    name="discovery",
    order=180,
    config_flag="discovery",
    config_field="discovery",
    category="discovery",
)
class DiscoveryStage(MethodDispatchStage):
    """Discover unbiased gene programs with the configured method (default NMF)."""

    def _select_method_name(self, config: dict) -> str:
        return config.get("method", "nmf")

    def _validate_output(self, result: StageResult) -> None:
        """No-op: discovery writes obsm/uns + tables, no strict postcondition."""


__all__ = ["DiscoveryStage"]
