"""PAGA compute method: writes uns['paga'] over a resolved grouping column."""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.embeddings import compute
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class PagaMethod(AnalysisMethod):
    """Compute PAGA connectivity over cell-type or cluster groups."""

    name = "paga"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        groupby = compute.resolve_paga_groupby(
            adata,
            config.get("paga_groupby"),
            cell_type_key=config.get("cell_type_key", "cell_type"),
            cluster_key=config.get("cluster_key", "leiden"),
        )
        if groupby is None:
            return self._skip("no grouping column (cell_type/leiden) present")
        try:
            compute.compute_paga(adata, groupby=groupby)
        except compute.EmbeddingsComputeError as exc:
            return self._skip(f"{exc}")
        return StageResult(
            adata=adata,
            notes=[f"paga: wrote uns['paga'] over '{groupby}'"],
            metrics={"method": self.name, "groupby": groupby},
            backend="python",
        )


__all__ = ["PagaMethod"]
