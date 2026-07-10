"""Leiden clustering method: kNN graph on the PCA embedding, then Leiden.

LeidenMethod is an AnalysisMethod strategy. It requires a PCA embedding
(obsm["X_pca"]) on input — enforced by its contract — builds a neighbors graph on
that embedding, and runs Leiden at the configured resolution.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


class LeidenMethod(AnalysisMethod):
    """Leiden clustering strategy over a PCA embedding."""

    # Registry identity.
    name = "leiden"
    stage_category = "clustering"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Leiden requires a PCA embedding to build the neighbor graph."""

        # Require the PCA embedding produced by the dimensionality stage.
        return DataContract(required_obsm=["X_pca"])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Build the neighbor graph and run Leiden clustering.

        Args:
            adata: Input AnnData carrying obsm["X_pca"].
            config: Resolved clustering config sub-block.
            context: Pipeline context (unused here).

        Returns:
            StageResult with cluster labels in obs and cluster-count metrics.
        """

        # Resolve settings with defaults matching ClusteringConfig.
        n_neighbors = int(config.get("n_neighbors", 15))
        resolution = float(config.get("resolution", 1.0))
        random_state = int(config.get("random_state", 0))
        key_added = config.get("key_added", "leiden")

        # Build the kNN graph on the PCA embedding.
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep="X_pca",
            random_state=random_state,
        )

        # Run Leiden at the configured resolution (flavor pinned for determinism).
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=random_state,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        # Count clusters for provenance.
        n_clusters = int(adata.obs[key_added].nunique())
        return StageResult(
            adata=adata,
            metrics={"n_clusters": n_clusters, "resolution": resolution},
            notes=[f"Leiden found {n_clusters} clusters at resolution {resolution}."],
        )


__all__ = ["LeidenMethod"]
