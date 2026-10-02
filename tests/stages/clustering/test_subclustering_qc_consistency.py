"""Tests for subclustering technical-QC consistency check.

Real motivating case: a fibroblast subcluster (n=3,866) had 5.0% probable-
multiplet cells (vs 0.2-1.6% in every other subcluster of the same lineage),
roughly a third the median counts/genes of its siblings, and no elevated
stress-response signature -- a QC-confounded population masquerading as a
distinct biological state, caught only by a manual, one-off check. This
module is that check made standard and automatic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from cellquorum.stages.clustering.subclustering.qc_consistency import (
    subcluster_qc_consistency,
)


def make_qc_confounded_adata() -> AnnData:
    """Three clusters: C0 and C1 have similar, healthy QC profiles; C2 has
    much lower counts/genes and a much higher probable-multiplet rate --
    the same shape as the real cluster-6 case."""
    rng = np.random.default_rng(0)

    n0, n1, n2 = 80, 80, 40
    clusters = ["C0"] * n0 + ["C1"] * n1 + ["C2"] * n2

    total_counts = np.concatenate(
        [
            rng.normal(8000, 500, n0),
            rng.normal(7800, 500, n1),
            rng.normal(1800, 200, n2),  # C2: much lower
        ]
    )
    n_genes = np.concatenate(
        [
            rng.normal(2800, 150, n0),
            rng.normal(2700, 150, n1),
            rng.normal(1000, 100, n2),  # C2: much lower
        ]
    )
    pct_mito = np.concatenate(
        [
            rng.normal(2.5, 0.3, n0),
            rng.normal(2.7, 0.3, n1),
            rng.normal(5.5, 0.5, n2),  # C2: much higher
        ]
    )
    # C2: much higher multiplet rate; C0/C1: low and similar.
    multiplet = np.concatenate(
        [
            rng.random(n0) < 0.01,
            rng.random(n1) < 0.015,
            rng.random(n2) < 0.05,
        ]
    )

    obs = pd.DataFrame(
        {
            "cl": clusters,
            "total_counts": total_counts,
            "n_genes_by_counts": n_genes,
            "pct_counts_mito": pct_mito,
            "qc_probable_multiplet": multiplet,
        },
        index=[f"cell_{i}" for i in range(n0 + n1 + n2)],
    )
    return AnnData(X=rng.normal(0, 1, (n0 + n1 + n2, 20)), obs=obs)


def test_flags_the_cluster_with_degraded_qc_and_elevated_multiplet_rate() -> None:
    adata = make_qc_confounded_adata()

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    assert result["clusters"]["C2"]["qc_pass"] is False
    assert result["clusters"]["C0"]["qc_pass"] is True
    assert result["clusters"]["C1"]["qc_pass"] is True
    assert result["summary"]["n_fail"] == 1
    assert result["summary"]["n_pass"] == 2


def test_flagged_cluster_reason_names_the_offending_metrics() -> None:
    adata = make_qc_confounded_adata()

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    reason = result["clusters"]["C2"]["qc_reason"]
    assert "total_counts" in reason or "n_genes_by_counts" in reason or "pct_counts_mito" in reason


def test_does_not_flag_clusters_with_similar_qc_profiles() -> None:
    rng = np.random.default_rng(1)
    n = 60
    obs = pd.DataFrame(
        {
            "cl": ["A"] * n + ["B"] * n + ["C"] * n,
            "total_counts": rng.normal(8000, 400, 3 * n),
            "n_genes_by_counts": rng.normal(2800, 150, 3 * n),
            "pct_counts_mito": rng.normal(2.5, 0.3, 3 * n),
            "qc_probable_multiplet": rng.random(3 * n) < 0.01,
        },
        index=[f"cell_{i}" for i in range(3 * n)],
    )
    adata = AnnData(X=rng.normal(0, 1, (3 * n, 20)), obs=obs)

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    assert result["summary"]["n_fail"] == 0


def test_a_cluster_with_higher_counts_is_not_flagged_direction_aware() -> None:
    """High total_counts is GOOD, not a quality problem -- an outlier in the
    healthy direction must not trip the same flag as one in the bad
    direction."""
    rng = np.random.default_rng(2)
    n0, n1 = 80, 40
    obs = pd.DataFrame(
        {
            "cl": ["C0"] * n0 + ["C1"] * n1,
            "total_counts": np.concatenate(
                [rng.normal(8000, 500, n0), rng.normal(20000, 1000, n1)]
            ),
            "n_genes_by_counts": np.concatenate(
                [rng.normal(2800, 150, n0), rng.normal(4500, 200, n1)]
            ),
            "pct_counts_mito": rng.normal(2.5, 0.3, n0 + n1),
            "qc_probable_multiplet": rng.random(n0 + n1) < 0.01,
        },
        index=[f"cell_{i}" for i in range(n0 + n1)],
    )
    adata = AnnData(X=rng.normal(0, 1, (n0 + n1, 20)), obs=obs)

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    assert result["clusters"]["C1"]["qc_pass"] is True
    assert result["summary"]["n_fail"] == 0


def test_skips_metrics_whose_column_is_absent() -> None:
    adata = make_qc_confounded_adata()
    del adata.obs["pct_counts_mito"]

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    assert "pct_counts_mito" not in result["summary"]["metrics_used"]
    assert "total_counts" in result["summary"]["metrics_used"]
    # C2 is still bad enough on the remaining metrics to fail.
    assert result["clusters"]["C2"]["qc_pass"] is False


def test_returns_skipped_when_no_qc_columns_present() -> None:
    rng = np.random.default_rng(3)
    obs = pd.DataFrame({"cl": ["A"] * 10 + ["B"] * 10}, index=[f"c{i}" for i in range(20)])
    adata = AnnData(X=rng.normal(0, 1, (20, 5)), obs=obs)

    result = subcluster_qc_consistency(adata, cluster_key="cl")

    assert result["summary"]["skipped"] is True
    assert result["clusters"] == {}
