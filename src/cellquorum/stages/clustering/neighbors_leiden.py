"""Leiden clustering method: kNN graph on the PCA embedding, then Leiden.

LeidenMethod is an AnalysisMethod strategy. It requires a PCA embedding
(obsm["X_pca"]) on input — enforced by its contract — builds a neighbors graph on
that embedding, and runs Leiden at the configured resolution.

## Who gets to define a cluster

A cluster boundary is a cohort-derived quantity: it is inferred from every cell that took
part in the graph. A damaged cell sitting between two populations can bridge them, or a group
of them can form a cluster of their own that then gets annotated as a cell type. So the
partition is fitted on the cells QC permits to fit, and every other cell receives its label
by nearest-neighbour transfer.

The neighbors graph itself stays at full size. That is deliberate and not a compromise: the
embeddings stage raises ``NeighborsMissing`` unless ``obsp`` covers every cell, so UMAP and
PAGA depend on it. A pairwise graph is also not a fitted parameter — what needed protecting
was the partition derived from it, and Leiden never sees the full graph.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod
from cellquorum.stages.qc.eligibility import Analysis, fitting_cells

logger = logging.getLogger(__name__)

#: Companion column recording how each cell got its label. Provenance rather than decoration:
#: without it, a transferred label is indistinguishable from a fitted one, and a reader cannot
#: tell which cluster assignments were inferred for cells that had no say in the partition.
LABEL_SOURCE_COLUMN = "cellquorum_cluster_label_source"


def transfer_cluster_labels(
    reference: np.ndarray,
    reference_labels: pd.Series,
    query: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Assign each query cell the majority label of its nearest reference cells.

    The out-of-sample step for a partition. Leiden has no transform of its own — a partition
    is a labelling of the cells that were present, not a function that can be applied to new
    ones — so nearest-neighbour transfer stands in for it. That is the same primitive
    reference-mapping tools use to label a query against an atlas, and it is appropriate for
    the same reason: the reference structure is fixed and the query is scored against it.

    Args:
        reference: Fitted cells' coordinates in the clustering representation.
        reference_labels: Their Leiden labels.
        query: Coordinates of the cells being labelled.
        n_neighbors: Neighbours to vote, capped at the reference size.

    Returns:
        One label per query cell, as strings drawn from the reference's own categories.
    """
    from sklearn.neighbors import KNeighborsClassifier

    k = int(min(n_neighbors, len(reference)))
    classifier = KNeighborsClassifier(n_neighbors=max(1, k))
    classifier.fit(reference, reference_labels.astype(str).to_numpy())
    return np.asarray(classifier.predict(query), dtype=object)


def _neighbors_and_leiden(
    adata: ad.AnnData,
    *,
    n_neighbors: int,
    use_rep: str,
    resolution: float,
    random_state: int,
    key_added: str,
) -> None:
    """Build the kNN graph and run Leiden on CPU (shared by CPU path and GPU fallback)."""
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state)
    # The igraph flavor is deterministic, which the whole reproducibility story depends on.
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

        from cellquorum.backends.compute import resolve_compute

        routing = resolve_compute(context)
        compute_used = "cpu"
        gpu_fallback_note = None

        # Cluster boundaries are inferred from whoever takes part, so the partition is fitted
        # on the cells QC permits to fit. This stage declares fit_scope=CORE at registration;
        # this is what honours it. CLUSTERING is its own analysis in the eligibility table
        # because a cell may legitimately be projected into the manifold and still be barred
        # from shaping clusters.
        fitting = fitting_cells(adata.obs, Analysis.CLUSTERING)

        # The object Leiden partitions. When a fit population is declared, it is a copy
        # carrying its own kNN graph — an induced subgraph would strip each core cell of the
        # edges it had to non-core cells, leaving it artificially isolated.
        target = adata if fitting is None else adata[fitting].copy()

        if routing["use_gpu"]:
            try:
                import rapids_singlecell as rsc

                rsc.get.anndata_to_GPU(target)
                rsc.pp.neighbors(
                    target, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state
                )
                rsc.tl.leiden(
                    target, resolution=resolution, random_state=random_state, key_added=key_added
                )
                rsc.get.anndata_to_CPU(target)
                compute_used = "gpu"
            except Exception as exc:  # noqa: BLE001
                if not routing["fallback_to_cpu"]:
                    raise
                try:
                    import rapids_singlecell as rsc

                    rsc.get.anndata_to_CPU(target)
                except Exception:
                    pass
                gpu_fallback_note = (
                    f"GPU clustering failed ({type(exc).__name__}: {str(exc)[:80]}); "
                    "fell back to CPU."
                )
                _neighbors_and_leiden(
                    target,
                    n_neighbors=n_neighbors,
                    use_rep=use_rep,
                    resolution=resolution,
                    random_state=random_state,
                    key_added=key_added,
                )
        else:
            _neighbors_and_leiden(
                target,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                resolution=resolution,
                random_state=random_state,
                key_added=key_added,
            )

        notes: list[str] = []
        if target is adata:
            adata.obs[LABEL_SOURCE_COLUMN] = "fitted"
        else:
            notes.append(
                self._transfer_and_graph(
                    adata,
                    target,
                    fitting=fitting,  # type: ignore[arg-type]
                    key_added=key_added,
                    use_rep=use_rep,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )
            )

        # Count clusters for provenance.
        n_clusters = int(adata.obs[key_added].nunique())
        notes.insert(0, f"Leiden found {n_clusters} clusters at resolution {resolution}.")
        if gpu_fallback_note:
            notes.append(gpu_fallback_note)

        return StageResult(
            adata=adata,
            metrics={"n_clusters": n_clusters, "resolution": resolution, "compute": compute_used},
            notes=notes,
        )

    def _transfer_and_graph(
        self,
        adata: ad.AnnData,
        fitted: ad.AnnData,
        *,
        fitting: pd.Series,
        key_added: str,
        use_rep: str,
        n_neighbors: int,
        random_state: int,
    ) -> str:
        """Carry a core-only partition back to every cell, and leave a full-size graph.

        Args:
            adata: The full object, mutated in place.
            fitted: The fit-population copy that Leiden partitioned.
            fitting: Boolean mask of the fitted cells.
            key_added: obs column holding the labels.
            use_rep: Representation the transfer votes in.
            n_neighbors: Neighbours to vote.
            random_state: Seed for the full-object graph.

        Returns:
            A provenance note naming how many labels were fitted and how many transferred.
        """
        mask = fitting.to_numpy(dtype=bool)
        coordinates = np.asarray(adata.obsm[use_rep])

        labels = pd.Series(pd.NA, index=adata.obs_names, dtype=object)
        labels.loc[fitted.obs_names] = fitted.obs[key_added].astype(str).to_numpy()

        # Every cell keeps a label. Withholding one would delete non-core cells from every
        # downstream stage that groups by cluster — a silent drop, which is exactly what the
        # graded model replaced. Whether a transferred label may inform a *conclusion* is the
        # eligibility masks' job, not this stage's.
        if (~mask).any():
            labels.loc[~mask] = transfer_cluster_labels(
                coordinates[mask],
                fitted.obs[key_added],
                coordinates[~mask],
                n_neighbors,
            )

        categories = sorted(set(fitted.obs[key_added].astype(str)))
        adata.obs[key_added] = pd.Categorical(labels, categories=categories)
        adata.obs[LABEL_SOURCE_COLUMN] = np.where(mask, "fitted", "transferred")

        # The embeddings stage raises NeighborsMissing unless obsp covers every cell, so the
        # full graph is rebuilt here for UMAP and PAGA. Leiden never saw it.
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state)

        note = (
            f"Clusters fitted on {int(mask.sum())} QC-permitted cells; "
            f"{int((~mask).sum())} further cells labelled by {n_neighbors}-NN transfer "
            f"without influencing the partition."
        )
        logger.info(note)
        return note


__all__ = ["LABEL_SOURCE_COLUMN", "LeidenMethod", "transfer_cluster_labels"]
