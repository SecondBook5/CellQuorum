# Pipeline step (order=160): integration_benchmark — score integration quality via scib-metrics.
"""Integration-benchmark evaluation stage: dispatch to scib-metrics method."""

from __future__ import annotations

import pandas as pd

from cellquorum.core.artifacts import ArtifactManager
from cellquorum.core.context import PipelineContext
from cellquorum.core.contracts import CellQuorumContractError
from cellquorum.core.stage import StageResult
from cellquorum.core.stage_catalog import register_stage
from cellquorum.methods.stage_base import DefaultMethodStage


@register_stage(
    name="integration_benchmark",
    order=160,
    config_flag="integration_benchmark",
    config_field="integration_benchmark",
    category="integration_benchmark",
)
class IntegrationBenchmarkStage(DefaultMethodStage):
    """Config-driven integration-quality evaluation stage.

    Measures batch-correction quality (iLISI/kBET/pcr) and biological-structure
    preservation (cLISI/silhouette/graph-connectivity/NMI) over multiple
    integration embeddings. READ-ONLY: never modifies obsm/obs. Returns ranking
    and per-embedding measurements in the run report.
    """

    default_method = "scib_benchmark"

    def run(self, context: PipelineContext) -> StageResult:
        result = super().run(context)
        scores = result.metrics.get("embeddings")
        if isinstance(scores, dict):
            rows = [
                (embedding, category, metric, value)
                for embedding, domains in scores.items()
                for category in ("batch", "bio")
                for metric, value in domains[category].items()
            ]
            table = pd.DataFrame(rows, columns=["embedding", "category", "metric", "value"])
            result.artifacts.append(
                ArtifactManager.from_root(context.paths.root).write_dataframe(
                    table,
                    name="integration_quality",
                    relative_path="reports/integration_quality.csv",
                    description="Batch mixing and biological identity scores for each embedding.",
                )
            )
        return result

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
