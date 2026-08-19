"""Leiden clustering method: kNN graph on the PCA embedding, then Leiden.

LeidenMethod is an AnalysisMethod strategy. It requires a PCA embedding
(obsm["X_pca"]) on input — enforced by its contract — builds a neighbors graph on
that embedding, and runs Leiden at the configured resolution.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


def _run_cpu_neighbors_leiden(
    adata: ad.AnnData,
    n_neighbors: int,
    use_rep: str,
    resolution: float,
    random_state: int,
    key_added: str,
) -> None:
    """Run neighbors + Leiden on CPU via scanpy (shared by CPU path and GPU fallback)."""

    # Build the kNN graph on the chosen embedding.
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state)
    # Run Leiden with the deterministic igraph flavor.
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added=key_added,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )


class LeidenMethod(AnalysisMethod):
    """Leiden clustering strategy over a PCA embedding."""

    # Registry identity.
    name = "leiden"
    stage_category = "clustering"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Leiden requires the configured embedding to build the neighbor graph."""

        # Cluster on the configured embedding (default X_pca; X_pca_harmony after integration).
        use_rep = config.get("use_rep", "X_pca")
        return DataContract(required_obsm=[use_rep])

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Build the neighbor graph and run Leiden clustering.

        Args:
            adata: Input AnnData carrying obsm["X_pca"].
            config: Resolved clustering config sub-block.
            context: Pipeline context.

        Returns:
            StageResult with cluster labels in obs and cluster-count metrics.
        """

        # Resolve settings with defaults matching ClusteringConfig.
        n_neighbors = int(config.get("n_neighbors", 15))
        resolution = float(config.get("resolution", 1.0))
        random_state = int(config.get("random_state", 0))
        key_added = config.get("key_added", "leiden")
        use_rep = config.get("use_rep", "X_pca")

        from cellquorum.compute.router import resolve_compute

        routing = resolve_compute(context)
        compute_used = "cpu"
        gpu_fallback_note = None

        if routing["use_gpu"]:
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_GPU(adata)
                rsc.pp.neighbors(
                    adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state
                )
                rsc.tl.leiden(
                    adata, resolution=resolution, random_state=random_state, key_added=key_added
                )
                rsc.get.anndata_to_CPU(adata)
                compute_used = "gpu"
            except Exception as exc:  # noqa: BLE001
                if not routing["fallback_to_cpu"]:
                    raise
                try:
                    import rapids_singlecell as rsc

                    rsc.get.anndata_to_CPU(adata)
                except Exception:
                    pass
                gpu_fallback_note = (
                    f"GPU clustering failed ({type(exc).__name__}: {str(exc)[:80]}); "
                    "fell back to CPU."
                )
                _run_cpu_neighbors_leiden(
                    adata, n_neighbors, use_rep, resolution, random_state, key_added
                )
        else:
            _run_cpu_neighbors_leiden(
                adata, n_neighbors, use_rep, resolution, random_state, key_added
            )

        # Count clusters for provenance.
        n_clusters = int(adata.obs[key_added].nunique())
        notes = [f"Leiden found {n_clusters} clusters at resolution {resolution}."]
        if gpu_fallback_note:
            notes.append(gpu_fallback_note)

        return StageResult(
            adata=adata,
            metrics={"n_clusters": n_clusters, "resolution": resolution, "compute": compute_used},
            notes=notes,
        )


__all__ = ["LeidenMethod"]
