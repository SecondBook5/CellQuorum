"""PHATE compute method: writes obsm['X_phate']; skips if phate unavailable."""

from __future__ import annotations

import anndata as ad

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.embeddings import compute
from cellquorum.embeddings.umap_method import _seed
from cellquorum.methods.base import AnalysisMethod, MethodSkip


class PhateMethod(AnalysisMethod):
    """Compute PHATE coordinates from a representation."""

    name = "phate"
    stage_category = "embeddings"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        return DataContract()

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult | MethodSkip:
        try:
            compute.compute_phate(
                adata,
                use_rep=config.get("use_rep", "X_pca_harmony"),
                knn=int(config.get("phate_knn", 15)),
                decay=int(config.get("phate_decay", 40)),
                random_state=_seed(config, context),
            )
        except compute.EmbeddingsComputeError as exc:
            return MethodSkip(reason=f"phate skipped: {exc}", details={"method": self.name})
        return StageResult(
            adata=adata,
            notes=["phate: wrote obsm['X_phate']"],
            metrics={"method": self.name},
            backend="python",
        )


__all__ = ["PhateMethod"]
