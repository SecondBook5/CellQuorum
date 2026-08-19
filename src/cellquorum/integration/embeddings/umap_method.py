"""UMAP compute method: writes obsm['X_umap'] on the existing neighbors graph."""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.integration.embeddings import compute
from cellquorum.methods.base import AnalysisMethod, MethodSkip


def _seed(config: dict, context: object) -> int:
    """Prefer config random_state, else context.random_seed, else 1337."""
    if config.get("random_state") is not None:
        return int(config["random_state"])
    return int(getattr(context, "random_seed", 1337))


class UmapMethod(AnalysisMethod):
    """Compute UMAP coordinates from the neighbors graph."""

    name = "umap"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            compute.compute_umap(
                adata,
                min_dist=float(config.get("umap_min_dist", 0.3)),
                random_state=_seed(config, context),
            )
        except compute.EmbeddingsComputeError as exc:
            return self._skip(f"{exc}")
        return StageResult(
            adata=adata,
            notes=["umap: wrote obsm['X_umap']"],
            metrics={"method": self.name},
            backend="python",
        )


__all__ = ["UmapMethod"]
