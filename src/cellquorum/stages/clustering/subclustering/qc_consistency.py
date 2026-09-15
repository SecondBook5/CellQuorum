"""Technical-QC consistency check for subclustering.

A subcluster's *identity* label answers "what marker program does it
express"; this answers a different, prior question: "is this cluster made of
cells the QC evidence already distrusts, more than expected?" Global QC runs
once, across every lineage, in filter mode -- a threshold calibrated across
cell types with very different baseline RNA content will pass a degraded
fibroblast that would fail as an obvious outlier against other fibroblasts
specifically. This does not re-threshold anything (that was tried and
rejected: see the retired `qc.mode: report_only`, which let every lineage
publish its own bound for cells one global pass had already judged). It only
compares a lineage's own subclusters against the lineage's own population,
using QC evidence the global pass already computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

# metric -> direction of a BAD value. "high_is_bad": flag when a cluster's
# value is a z-outlier on the high side (e.g. %mito, multiplet rate).
# "low_is_bad": flag on the low side (e.g. counts, genes -- fewer is worse).
_DEFAULT_METRICS: dict[str, str] = {
    "pct_counts_mito": "high_is_bad",
    "total_counts": "low_is_bad",
    "n_genes_by_counts": "low_is_bad",
    "qc_probable_multiplet": "high_is_bad",
}

# Boolean obs columns are a rate (proportion test), not a continuous median.
_RATE_COLUMNS: frozenset[str] = frozenset({"qc_probable_multiplet"})

# Asymptotic SE(median) / SE(mean-of-robust-scale) constant relating a MAD
# (scaled to be a normal-equivalent sigma via *1.4826) to the standard error
# of a SAMPLE MEDIAN of n draws: SE(median) ~= 1.2533 * sigma / sqrt(n).
_MEDIAN_SE_CONSTANT = 1.2533

# A pure z-test alone is wrong at both ends of realistic cluster sizes: with
# a handful of cells its standard error is huge and nothing ever flags; with
# the thousands of cells a real subcluster usually has, the standard error
# shrinks toward zero and EVERYTHING becomes "significant", including a
# cluster whose value is practically indistinguishable from the population.
# Flagging additionally requires a minimum PRACTICAL effect size -- how many
# population-MADs away a continuous metric sits, or how many percentage
# points away a rate sits -- independent of the cluster's own cell count, so
# a huge cluster that is merely a little off does not trip it and a small
# cluster that is dramatically off still can.
_MIN_EFFECT_SIZE_MADS = 1.5
_MIN_EFFECT_SIZE_RATE_PP = 0.02


def subcluster_qc_consistency(
    adata: AnnData,
    cluster_key: str,
    *,
    metrics: dict[str, str] | None = None,
    z_threshold: float = 2.5,
) -> dict:
    """
    Compare each subcluster's QC evidence against the lineage's own population.

    For each configured metric with a matching obs column, this tests
    whether a cluster's summary value (median for a continuous metric,
    proportion for a boolean/rate metric) differs from the WHOLE lineage's
    own value by more than sampling noise would explain, given that
    cluster's cell count -- a standard-error-scaled z-test, not a comparison
    of the (often very few) per-cluster summary values against each other.
    That matters at low cluster counts: with only 2-3 subclusters, a
    cluster's own median is always exactly 1.4826 MADs from a shared
    median/MAD computed over just those summary values, which makes any
    such test structurally unable to flag anything regardless of how
    extreme the real difference is. Testing against the full per-cell
    population (thousands of cells, not a handful of cluster summaries)
    has no such degeneracy and correctly demands a bigger effect size from
    a smaller cluster before flagging it.

    Args:
        adata: AnnData with obs[cluster_key] and QC evidence columns.
        cluster_key: obs column with cluster labels.
        metrics: {obs_column: "high_is_bad" | "low_is_bad"}. Defaults to
            pct_counts_mito, total_counts, n_genes_by_counts,
            qc_probable_multiplet.
        z_threshold: z-score magnitude that trips a flag.

    Returns:
        dict with:
        - clusters: {cluster_id: {n_cells, metrics: {name: {value, z,
          flagged}}, qc_pass, qc_reason}}
        - summary: {n_pass, n_fail, metrics_used, skipped}
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs")

    configured = dict(metrics) if metrics is not None else dict(_DEFAULT_METRICS)
    metrics_used = [col for col in configured if col in adata.obs.columns]

    if not metrics_used:
        return {
            "clusters": {},
            "summary": {"n_pass": 0, "n_fail": 0, "metrics_used": [], "skipped": True},
        }

    cluster_labels = adata.obs[cluster_key].astype(str)
    cluster_ids = sorted(cluster_labels.unique())

    # Population (whole-lineage) reference stats per metric, computed once.
    pop_stats: dict[str, tuple] = {}
    for col in metrics_used:
        if col in _RATE_COLUMNS:
            all_vals = adata.obs[col].astype(bool)
            pop_stats[col] = ("rate", float(all_vals.mean()))
        else:
            all_vals = pd.to_numeric(adata.obs[col], errors="coerce")
            pop_median = float(all_vals.median())
            pop_mad = float((all_vals - pop_median).abs().median())
            pop_stats[col] = ("continuous", pop_median, pop_mad)

    clusters: dict[str, dict] = {}
    n_pass = 0
    n_fail = 0
    for cid in cluster_ids:
        mask = (cluster_labels == cid).to_numpy()
        n_cells = int(mask.sum())
        cluster_metrics: dict[str, dict] = {}
        flagged_names: list[str] = []

        for col in metrics_used:
            direction = configured[col]
            kind = pop_stats[col][0]

            if kind == "rate":
                _, pop_p = pop_stats[col]
                cluster_p = float(adata.obs[col].astype(bool).to_numpy()[mask].mean())
                # Two-proportion z-test of this cluster's rate against the
                # population rate it is itself part of. A floor on the
                # variance term keeps a population rate of exactly 0 or 1
                # from dividing by zero.
                variance = max(pop_p * (1 - pop_p), 1e-6)
                se = np.sqrt(variance / max(n_cells, 1))
                z = (cluster_p - pop_p) / se if se > 0 else 0.0
                value = cluster_p
                effect_size = abs(cluster_p - pop_p) / _MIN_EFFECT_SIZE_RATE_PP
            else:
                _, pop_median, pop_mad = pop_stats[col]
                col_values = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy()[mask]
                cluster_median = float(np.median(col_values))
                sigma = 1.4826 * pop_mad
                # A relative floor on sigma: a metric whose population MAD
                # happens to be ~0 (near-constant) would otherwise divide by
                # ~0 and flag any tiny, meaningless difference.
                sigma = max(sigma, 0.01 * abs(pop_median), 1e-9)
                se = _MEDIAN_SE_CONSTANT * sigma / np.sqrt(max(n_cells, 1))
                z = (cluster_median - pop_median) / se if se > 0 else 0.0
                value = cluster_median
                effect_size = abs(cluster_median - pop_median) / (sigma * _MIN_EFFECT_SIZE_MADS)

            statistically_off = (
                (z > z_threshold) if direction == "high_is_bad" else (z < -z_threshold)
            )
            practically_off = effect_size >= 1.0
            flagged = statistically_off and practically_off
            cluster_metrics[col] = {"value": value, "z": float(z), "flagged": flagged}
            if flagged:
                flagged_names.append(col)

        qc_pass = not flagged_names
        if qc_pass:
            n_pass += 1
            qc_reason = "within normal range for the lineage"
        else:
            n_fail += 1
            qc_reason = (
                f"differs from the lineage population (|z| > {z_threshold}) on: "
                f"{', '.join(flagged_names)}"
            )

        clusters[cid] = {
            "n_cells": n_cells,
            "metrics": cluster_metrics,
            "qc_pass": qc_pass,
            "qc_reason": qc_reason,
        }

    return {
        "clusters": clusters,
        "summary": {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "metrics_used": metrics_used,
            "skipped": False,
        },
    }


__all__ = ["subcluster_qc_consistency"]
