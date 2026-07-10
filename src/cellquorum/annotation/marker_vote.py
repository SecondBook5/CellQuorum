"""Marker-vote annotation: assign each cluster the best-scoring cell type.

For each configured cell-type panel, score every cell (scanpy score_genes on the
log-normalized layer), average per cluster, and assign each cluster the argmax
cell type. Deterministic, offline, CPU — the default annotation method.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from cellquorum.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod


class MarkerVoteMethod(AnalysisMethod):
    """Per-cluster argmax-of-panel-scores annotation strategy."""

    # Registry identity.
    name = "marker_vote"
    stage_category = "annotation"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Require the cluster column and a log-normalized score layer."""

        # Read the cluster key + score layer from config.
        cluster_key = config.get("cluster_key", "leiden")
        score_layer = config.get("score_layer", "cellquorum_normalized")

        # Require the cluster labels and that the score layer is lognorm.
        return DataContract(
            required_obs=[cluster_key],
            required_layers=[score_layer],
            expression_layer=score_layer,
            expected_kind="lognorm",
        )

    def requires_obs(self, config: dict) -> list[str]:
        """Return the cluster key that must exist for annotation to run."""

        # Read the cluster key from config.
        cluster_key = config.get("cluster_key", "leiden")

        # Require the cluster column to exist.
        return [cluster_key]

    def _run(self, adata: ad.AnnData, config: dict, context: object) -> StageResult:
        """
        Score each cell-type panel, average per cluster, assign the argmax.

        Args:
            adata: Clustered AnnData with a lognorm score layer.
            config: Resolved annotation config sub-block.
            context: Pipeline context (unused).

        Returns:
            StageResult with obs[key_added] set and assignment metrics.
        """

        # Resolve settings.
        cluster_key = config.get("cluster_key", "leiden")
        score_layer = config.get("score_layer", "cellquorum_normalized")
        key_added = config.get("key_added", "cell_type")
        panels = config.get("marker_panels", {}) or {}
        random_state = int(config.get("random_state", 0))

        # Score genes on the log-normalized layer for each cell-type panel.
        # Use a temporary object whose .X is the score layer so score_genes reads it.
        scored = adata.copy()
        scored.X = scored.layers[score_layer]
        score_cols = {}
        for cell_type, genes in panels.items():
            present = [g for g in genes if g in scored.var_names]
            col = f"_score_{cell_type}"
            if present:
                sc.tl.score_genes(scored, present, score_name=col, random_state=random_state)
            else:
                scored.obs[col] = 0.0
            score_cols[cell_type] = col

        # Average each panel score per cluster, then argmax to assign a type.
        clusters = adata.obs[cluster_key].astype(str)
        assignments = {}
        for cluster in clusters.unique():
            mask = (clusters == cluster).to_numpy()
            best_type, best_score = None, -np.inf
            for cell_type, col in score_cols.items():
                mean_score = float(scored.obs.loc[mask, col].mean())
                if mean_score > best_score:
                    best_type, best_score = cell_type, mean_score
            assignments[cluster] = best_type

        # Write the per-cell assignment.
        adata.obs[key_added] = clusters.map(assignments).astype("category")

        return StageResult(
            adata=adata,
            metrics={
                "n_types": len(panels),
                "cluster_key": cluster_key,
                "assignments": assignments,
                "key_added": key_added,
            },
            notes=[f"marker_vote assigned {len(assignments)} clusters -> {key_added}."],
        )


__all__ = ["MarkerVoteMethod"]
