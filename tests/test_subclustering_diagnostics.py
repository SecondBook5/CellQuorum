"""Tests for subclustering diagnostic plots."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.clustering.subclustering.diagnostics import (
    plot_marker_identity_dotplot,
    plot_technical_qc_consistency,
    select_marker_genes,
)


def _identity_adata(n_per_cluster: int = 60, seed: int = 0) -> ad.AnnData:
    """Two clusters, each with its own clearly-elevated marker gene."""
    rng = np.random.default_rng(seed)
    n = n_per_cluster * 2
    clusters = np.repeat(["0", "1"], n_per_cluster)
    genes = ["MARKER_0", "MARKER_1", "HOUSEKEEP"]
    X = rng.poisson(1.0, size=(n, len(genes))).astype("float32")
    # Cluster 0 cells: MARKER_0 high. Cluster 1 cells: MARKER_1 high.
    X[clusters == "0", 0] += 20
    X[clusters == "1", 1] += 20
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"subcluster": clusters}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=genes),
    )
    a.layers["cellquorum_normalized"] = X.copy()
    return a


def _consistency_result() -> dict:
    return {
        "clusters": {
            "0": {
                "n_cells": 100,
                "qc_pass": True,
                "qc_reason": "within normal range on all checked metrics",
                "metrics": {
                    "pct_counts_mito": {"value": 2.5, "z": 0.1, "flagged": False},
                    "total_counts": {"value": 8000, "z": -0.2, "flagged": False},
                },
            },
            "6": {
                "n_cells": 40,
                "qc_pass": False,
                "qc_reason": "outlier vs. sibling subclusters on: total_counts",
                "metrics": {
                    "pct_counts_mito": {"value": 5.5, "z": 3.1, "flagged": True},
                    "total_counts": {"value": 1800, "z": -3.4, "flagged": True},
                },
            },
        },
        "summary": {"n_pass": 1, "n_fail": 1, "metrics_used": ["pct_counts_mito", "total_counts"]},
    }


def test_plot_technical_qc_consistency_writes_png_and_a_vector_twin(tmp_path) -> None:
    out = plot_technical_qc_consistency(_consistency_result(), tmp_path / "qc_consistency.png")

    assert out == tmp_path / "qc_consistency.png"
    assert out.exists()
    assert out.with_suffix(".pdf").exists()


def test_plot_technical_qc_consistency_skips_when_no_clusters(tmp_path) -> None:
    empty_result = {"clusters": {}, "summary": {"n_pass": 0, "n_fail": 0, "metrics_used": []}}

    out = plot_technical_qc_consistency(empty_result, tmp_path / "qc_consistency.png")

    assert not out.exists()


def test_select_marker_genes_finds_the_real_marker_per_cluster() -> None:
    a = _identity_adata()
    markers = select_marker_genes(
        a, cluster_key="subcluster", layer="cellquorum_normalized", n_top=1
    )

    assert markers["0"] == ["MARKER_0"]
    assert markers["1"] == ["MARKER_1"]


def test_plot_marker_identity_dotplot_writes_png(tmp_path) -> None:
    a = _identity_adata()
    out = plot_marker_identity_dotplot(
        a,
        cluster_key="subcluster",
        layer="cellquorum_normalized",
        out_path=tmp_path / "marker_dotplot.png",
        n_top=2,
    )

    assert out == tmp_path / "marker_dotplot.png"
    assert out.exists()


def test_plot_marker_identity_dotplot_applies_cluster_labels(tmp_path) -> None:
    a = _identity_adata()
    out = plot_marker_identity_dotplot(
        a,
        cluster_key="subcluster",
        layer="cellquorum_normalized",
        out_path=tmp_path / "marker_dotplot_labeled.png",
        n_top=2,
        cluster_labels={"0": "Papillary", "1": "Reticular"},
    )

    assert out.exists()
    # The rename is local to the plotting call -- the caller's own obs column
    # is never mutated in place, matching de_heatmap_method's convention of
    # building a display-only relabeled copy rather than touching the object.
    assert set(a.obs["subcluster"].astype(str).unique()) == {"0", "1"}


def test_plot_marker_identity_dotplot_skips_with_fewer_than_two_clusters(tmp_path) -> None:
    a = _identity_adata()
    a = a[a.obs["subcluster"] == "0"].copy()
    out = plot_marker_identity_dotplot(
        a,
        cluster_key="subcluster",
        layer="cellquorum_normalized",
        out_path=tmp_path / "marker_dotplot.png",
        n_top=2,
    )

    assert not out.exists()
