"""Marker-vote annotation: assign each cluster the best-scoring, significant cell type.

For each configured cell-type panel, score every cell (scanpy score_genes on the
log-normalized layer), average per cluster, and assign each cluster the argmax cell
type -- among panels whose score is statistically significant against a permutation
null, not merely the relatively-highest score. Deterministic, offline, CPU — the
default annotation method.

## Why a significance test, not just the highest score

A panel whose genes exist in the data but carry no real signal there (present, but
expressed in a fraction of a percent of cells) can still score higher than every other
candidate on a cluster that matches none of them well -- score_genes' relative,
background-adjusted score has no notion of "this is indistinguishable from chance."
Confirmed on real data (GSE140819's HTAPP-StJude-SMP-PDX1): an "Immune" panel present
in var_names but essentially unexpressed (CD68/LYZ detected in ~0.2% of cells) won 30%
of cells by relative score alone, with every candidate's absolute score near zero.

The fix is not a hand-picked minimum score (which would just be a different arbitrary
threshold tuned to look right on one dataset) but a permutation test: for each cluster,
compare a panel's score against the scores of many random gene sets of the same size,
scored on the same cells. Only a panel whose score beats that null at the configured
significance level (default 0.05, the standard convention -- not tuned per dataset) may
win a cluster; if none do, the cluster is honestly "Unassigned" rather than given an
unsupported label. This subsumes the zero-present-gene case (its "signal" trivially
fails to beat the null) as one instance of the general problem, rather than a special
case handled separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cellquorum.core.contracts import DataContract
from cellquorum.core.stage import StageResult
from cellquorum.methods.base import AnalysisMethod, MethodSkip

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

#: Sentinel written to key_added when no candidate panel beats the permutation null.
UNASSIGNED = "Unassigned"


def _empirical_p_value(observed: float, null: np.ndarray) -> float:
    """One-sided empirical p-value: P(a random gene set scores >= observed).

    Add-one smoothing on both numerator and denominator avoids a p-value of exactly
    0.0 from a finite permutation sample, which would overstate significance no matter
    how many permutations were run.
    """
    return float((int(np.sum(null >= observed)) + 1) / (len(null) + 1))


def _null_score_distribution(
    scored: ad.AnnData,
    *,
    panel_size: int,
    cluster_masks: dict[str, np.ndarray],
    n_permutations: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Per-cluster null: mean score of a random gene set of ``panel_size``, many times.

    Shared across every real panel of this size and every cluster: one score_genes call
    per permutation scores every cell at once, so every cluster's null draw for that
    permutation comes for free from the same call, rather than needing its own.

    Args:
        scored: Minimal AnnData built from the score layer (see caller).
        panel_size: Number of genes in the real panel(s) this null distribution serves.
        cluster_masks: cluster_id -> boolean mask over ``scored.obs_names``.
        n_permutations: Random gene sets to draw.
        random_state: Base seed; combined with the permutation index for a distinct,
            reproducible draw each time.

    Returns:
        cluster_id -> array of ``n_permutations`` mean scores.
    """
    rng = np.random.default_rng(random_state)
    all_genes = scored.var_names.to_numpy()
    null_scores: dict[str, list[float]] = {cluster: [] for cluster in cluster_masks}
    for i in range(n_permutations):
        random_genes = rng.choice(all_genes, size=panel_size, replace=False)
        sc.tl.score_genes(
            scored, list(random_genes), score_name="_null_score", random_state=random_state + i
        )
        for cluster, mask in cluster_masks.items():
            null_scores[cluster].append(float(scored.obs.loc[mask, "_null_score"].mean()))
    return {cluster: np.asarray(values) for cluster, values in null_scores.items()}


class MarkerVoteMethod(AnalysisMethod):
    """Per-cluster argmax-of-panel-scores annotation strategy."""

    # Registry identity.
    name = "marker_vote"
    stage_category = "annotation"
    backend = "python"

    def input_contract(self, config: dict) -> DataContract:
        """Require the cluster column and a log-normalized score layer."""

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

        cluster_key = config.get("cluster_key", "leiden")

        # Require the cluster column to exist.
        return [cluster_key]

    def _run(
        self, adata: ad.AnnData, config: dict, context: PipelineContext
    ) -> StageResult | MethodSkip:
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
        n_permutations = int(config.get("n_permutations", 100))
        significance_alpha = float(config.get("significance_alpha", 0.05))

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
        score_cols: dict[str, str] = {}
        panel_sizes: dict[str, int] = {}
        # Panels with zero present genes are excluded from candidacy entirely -- the
        # degenerate case of the significance test below (zero real genes trivially
        # cannot beat a random-gene-set null), handled up front so score_genes is never
        # asked to score an empty list.
        missing_panels: list[str] = []
        for cell_type, genes in panels.items():
            present = [g for g in genes if g in scored.var_names]
            if not present:
                missing_panels.append(cell_type)
                continue
            col = f"_score_{cell_type}"
            sc.tl.score_genes(scored, present, score_name=col, random_state=random_state)
            score_cols[cell_type] = col
            panel_sizes[cell_type] = len(present)

        if not score_cols:
            return self._skip(
                "none of the configured marker_panels have any gene present in the data "
                f"(checked: {sorted(panels)}). Check gene symbols and species convention."
            )

        clusters = adata.obs[cluster_key].astype(str)
        cluster_masks = {cluster: (clusters == cluster).to_numpy() for cluster in clusters.unique()}

        # One null distribution per distinct panel SIZE, not per panel: a random gene
        # set's score depends on how many genes it has, not which real panel it stands
        # in for, and scoring one permutation draw covers every cluster and every
        # same-sized panel at once. Sizes are typically few relative to panel count, so
        # this reuse is the difference between the check being affordable at all.
        null_by_size: dict[int, dict[str, np.ndarray]] = {}
        for size in set(panel_sizes.values()):
            null_by_size[size] = _null_score_distribution(
                scored,
                panel_size=size,
                cluster_masks=cluster_masks,
                n_permutations=n_permutations,
                random_state=random_state,
            )

        # Per cluster: only panels whose score is significant against the permutation
        # null (p < significance_alpha) may compete; argmax among those. A panel with
        # real genes but no real signal here (score indistinguishable from a random
        # gene set of the same size) cannot win merely for being relatively less bad
        # than the others -- that is exactly how an absent population used to get a
        # confident, unsupported label. No candidate passing is UNASSIGNED, not a guess.
        assignments: dict[str, str] = {}
        unassigned_clusters: list[str] = []
        for cluster, mask in cluster_masks.items():
            best_type, best_score = None, -np.inf
            for cell_type, col in score_cols.items():
                mean_score = float(scored.obs.loc[mask, col].mean())
                null = null_by_size[panel_sizes[cell_type]][cluster]
                if _empirical_p_value(mean_score, null) >= significance_alpha:
                    continue
                if mean_score > best_score:
                    best_type, best_score = cell_type, mean_score
            if best_type is None:
                unassigned_clusters.append(cluster)
                best_type = UNASSIGNED
            assignments[cluster] = best_type

        adata.obs[key_added] = clusters.map(assignments).astype("category")

        warnings = []
        if missing_panels:
            warnings.append(
                f"{len(missing_panels)} marker panel(s) had zero genes present in the data "
                f"and were excluded from voting (not scored as a fake neutral 0.0): "
                f"{sorted(missing_panels)}. Check gene symbols and species convention."
            )
        if unassigned_clusters:
            warnings.append(
                f"{len(unassigned_clusters)} cluster(s) had no marker panel significant at "
                f"p<{significance_alpha} and were labeled '{UNASSIGNED}' rather than given "
                f"an unsupported best-guess label: {sorted(unassigned_clusters)}."
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
