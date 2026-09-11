"""Marker-vote annotation: assign each cluster the best-scoring cell type.

For each configured cell-type panel, score every cell (scanpy score_genes on the
log-normalized layer), average per cluster, and assign each cluster the argmax
cell type. Deterministic, offline, CPU — the default annotation method.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.core.contracts import DataContract
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

        # No panels means no evidence, so say so instead of writing a column of NaN.
        #
        # `marker_panels` defaults to `{}`, and with it empty the scoring loop below simply does
        # not execute: every cell got NaN, the stage reported success with `n_types: 0`, and
        # `cell_type_markers` was an all-null categorical. Marker voting is the *mechanistic*
        # leg of this annotation — the one that is neither a trained model nor an atlas — so a
        # silently empty column removes the only independent check without anyone noticing.
        if not panels:
            return self._skip(
                "no marker_panels configured, so there is nothing to score. Marker voting is "
                "the mechanistic evidence in a multi-source annotation; an empty panel set "
                "produces an all-null column that looks like a computed answer. Supply "
                "annotation.marker_panels as {cell_type: [genes]}."
            )

        # Score genes on the log-normalized layer for each cell-type panel.
        # A MINIMAL object whose .X is the score layer, so score_genes reads it.
        #
        # `adata.copy()` duplicated the whole cohort -- two sparse layers over 201,871 x 33,417,
        # every obsm and ~90 obs columns -- to change which matrix is .X. score_genes reads the
        # matrix and the gene names and nothing else. This is the same fault that made
        # state_scoring allocate 12 GB in one minute and sit at zero free memory until the VM
        # died; the panels this stage now scores make it run for real rather than no-op.
        scored = ad.AnnData(
            X=adata.layers[score_layer],
            obs=pd.DataFrame(index=adata.obs_names),
            var=pd.DataFrame(index=adata.var_names),
        )
        score_cols = {}
        # Panels with zero present genes are excluded from candidacy entirely, not scored
        # as a flat 0.0. A "neutral" 0.0 is not neutral against real panels: score_genes
        # scores are gene-set expression minus a matched control set, which is legitimately
        # negative for a cluster that matches neither known type -- exactly a genuinely
        # novel population, the case this method most needs to get right. A absent-gene
        # panel sitting at 0.0 would silently outscore every real candidate there and win
        # an unsupported label. This is virtually always a config problem (typo, wrong
        # species/ID convention), so it is reported as a warning, not swallowed.
        missing_panels: list[str] = []
        for cell_type, genes in panels.items():
            present = [g for g in genes if g in scored.var_names]
            if not present:
                missing_panels.append(cell_type)
                continue
            col = f"_score_{cell_type}"
            sc.tl.score_genes(scored, present, score_name=col, random_state=random_state)
            score_cols[cell_type] = col

        if not score_cols:
            return self._skip(
                "none of the configured marker_panels have any gene present in the data "
                f"(checked: {sorted(panels)}). Check gene symbols and species convention."
            )

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

        warnings = []
        if missing_panels:
            warnings.append(
                f"{len(missing_panels)} marker panel(s) had zero genes present in the data "
                f"and were excluded from voting (not scored as a fake neutral 0.0): "
                f"{sorted(missing_panels)}. Check gene symbols and species convention."
            )

        return StageResult(
            adata=adata,
            metrics={
                "n_types": len(score_cols),
                "cluster_key": cluster_key,
                "assignments": assignments,
                "key_added": key_added,
            },
            notes=[f"marker_vote assigned {len(assignments)} clusters -> {key_added}."],
            warnings=warnings,
        )


__all__ = ["MarkerVoteMethod"]
