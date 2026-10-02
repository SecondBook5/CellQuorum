"""compute_resolution_stability on real (tiny) clustering runs -- not the pure math.

Confirms the orchestration actually produces a sane long-form result and that the
adaptive allocation spends its extra budget where the module claims it does: on
resolutions that are uncertain or sit next to a transition, not spread blindly.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.stages.clustering.resolution_diagnostic import (
    compute_resolution_stability,
    summarize_resolution_stability,
)


def _blobs(n_per_blob: int = 40, n_blobs: int = 3, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = n_per_blob * n_blobs
    coords = rng.normal(scale=0.4, size=(n, 5)).astype(np.float32)
    for i in range(n_blobs):
        coords[i * n_per_blob : (i + 1) * n_per_blob, :2] += rng.normal(
            loc=i * 15.0, scale=0.5, size=2
        )
    adata = ad.AnnData(X=rng.normal(size=(n, 10)).astype(np.float32))
    adata.obsm["X_pca"] = coords
    return adata


def test_non_adaptive_spends_exactly_n_bootstraps_per_resolution():
    adata = _blobs()
    long_df, n_clusters = compute_resolution_stability(
        adata,
        use_rep="X_pca",
        n_neighbors=10,
        resolutions=[0.5, 1.0],
        n_bootstraps=4,
        subsample_fraction=0.8,
        random_state=0,
        adaptive=False,
    )

    assert set(long_df["resolution"]) <= {0.5, 1.0}
    assert set(long_df["bootstrap"]) == {0, 1, 2, 3}
    assert set(n_clusters) == {0.5, 1.0}
    assert all(v >= 1 for v in n_clusters.values())


def test_adaptive_mode_produces_a_valid_summary():
    adata = _blobs()
    long_df, n_clusters = compute_resolution_stability(
        adata,
        use_rep="X_pca",
        n_neighbors=10,
        resolutions=[0.3, 0.6, 1.0, 1.5],
        n_bootstraps=6,
        subsample_fraction=0.8,
        random_state=0,
        adaptive=True,
        initial_fraction=0.5,
    )

    summary = summarize_resolution_stability(long_df, n_clusters)
    assert list(summary["resolution"]) == [0.3, 0.6, 1.0, 1.5]
    assert (summary["median_jaccard"].dropna() >= 0).all()
    assert (summary["median_jaccard"].dropna() <= 1).all()
    # Every resolution kept at least its round-1 observations -- adaptive never zeroes
    # a resolution out entirely, since the deliverable is a full curve.
    assert (summary["n_bootstrap_observations"] > 0).all()


def test_adaptive_allocation_favors_uncertain_and_transitioning_resolutions():
    """Round 2 must actually spend more where round 1 was noisy or near a jump.

    Built directly from a synthetic round-1 long_df (not a real clustering run) so the
    allocation logic is checked in isolation from clustering noise: three resolutions
    with identical, tight round-1 stability (nothing to chase) and one wild outlier
    (both uncertain AND a transition point) should draw the lion's share of round 2.
    """
    from cellquorum.stages.clustering.resolution_diagnostic import _round2_allocation

    long_df = pd.DataFrame(
        {
            "resolution": [0.5] * 4 + [1.0] * 4 + [1.5] * 4,
            "cluster": ["0"] * 12,
            "bootstrap": list(range(4)) * 3,
            # 0.5 and 1.5 are flat and tight; 1.0 is noisy and a transition point.
            "jaccard": [0.9, 0.9, 0.9, 0.9] + [0.9, 0.1, 0.5, 0.2] + [0.9, 0.9, 0.9, 0.9],
        }
    )

    allocation = _round2_allocation(long_df, [0.5, 1.0, 1.5], remaining_budget=30)

    assert allocation[1.0] > allocation[0.5]
    assert allocation[1.0] > allocation[1.5]


def test_round2_allocation_with_no_signal_does_not_exceed_the_budget():
    """When round 1 gave no signal to prioritize by, the fallback must respect the budget,
    not silently give every resolution 1 regardless of how small remaining_budget is."""
    from cellquorum.stages.clustering.resolution_diagnostic import _round2_allocation

    long_df = pd.DataFrame(
        {
            "resolution": [0.5, 1.0, 1.5, 2.0],
            "cluster": ["0"] * 4,
            "bootstrap": [0] * 4,
            "jaccard": [0.7, 0.7, 0.7, 0.7],  # identical -- no signal to prioritize by
        }
    )

    allocation = _round2_allocation(long_df, [0.5, 1.0, 1.5, 2.0], remaining_budget=2)

    assert sum(allocation.values()) <= 2, allocation
