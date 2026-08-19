"""Integration-benchmark evaluation stage: dispatch to scib-metrics method."""

from __future__ import annotations

from cellquorum.contracts import CellQuorumContractError
from cellquorum.core.stage import StageResult
from cellquorum.methods.stage_base import MethodDispatchStage


class IntegrationBenchmarkStage(MethodDispatchStage):
    """Config-driven integration-quality evaluation stage.

    Measures batch-correction quality (iLISI/kBET/pcr) and biological-structure
    preservation (cLISI/silhouette/graph-connectivity/NMI) over multiple
    integration embeddings. READ-ONLY: never modifies obsm/obs. Returns ranking
    + per-embedding metrics as StageResult.metrics only.
    """

    name = "integration_benchmark"
    stage_category = "integration_benchmark"

    def _select_method_name(self, config: dict) -> str:
        """Return the configured benchmark method (default 'scib_benchmark')."""
        return config.get("method", "scib_benchmark")

    def _validate_output(self, result: StageResult) -> None:
        """Assert embeddings metrics were recorded (non-skip only)."""
        if result.metrics.get("skipped"):
            return
        # The embeddings key must exist in metrics.
        if "embeddings" not in result.metrics:
            raise CellQuorumContractError(
                "integration_benchmark did not produce 'embeddings' in metrics."
            )


__all__ = ["IntegrationBenchmarkStage"]
