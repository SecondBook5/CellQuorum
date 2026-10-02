"""TDD tests for the leiden_stability subclustering partition method.

An alternative to CHOIR: bootstrap-validated Leiden resolution selection
(chooseR-style), for when CHOIR is unavailable, fails, or its own formal
significance test rejects the partition. Reuses
cellquorum.stages.clustering.resolution_diagnostic (already tested) rather
than reimplementing the bootstrap sweep.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from cellquorum.methods.base import MethodSkip
from cellquorum.stages.clustering.subclustering.config import SubclusteringConfig
from cellquorum.stages.clustering.subclustering.partition import run_leiden_stability


def _adata(n: int = 300, n_dims: int = 10, seed: int = 0) -> ad.AnnData:
    """Three well-separated blobs in a low-dim reduction, so a real best
    resolution exists and bootstrap Jaccard has a genuine signal to find."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(3, n_dims))
    labels = rng.integers(0, 3, size=n)
    coords = centers[labels] + rng.normal(scale=0.5, size=(n, n_dims))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    adata = ad.AnnData(X=np.zeros((n, 1), dtype="float32"), obs=obs)
    adata.obsm["X_test_reduction"] = coords
    return adata


def test_skips_when_reduction_key_missing() -> None:
    adata = _adata()
    config = SubclusteringConfig(key_added="fibroblast_subcluster")
    result = run_leiden_stability(adata, config, reduction_key="X_does_not_exist")
    assert isinstance(result, MethodSkip)
    assert "reduction" in result.reason.lower()


def test_writes_key_added_labels_and_picks_a_real_resolution() -> None:
    adata = _adata(n=240)
    config = SubclusteringConfig(
        key_added="fibroblast_subcluster",
        partition={
            "method": "leiden_stability",
            "leiden_stability": {
                "resolutions": [0.2, 0.5, 1.5],
                "n_bootstraps": 4,
                "subsample_fraction": 0.8,
                "random_state": 0,
                "adaptive": False,
            },
        },
    )
    result = run_leiden_stability(adata, config, reduction_key="X_test_reduction")
    assert not isinstance(result, MethodSkip)
    assert "fibroblast_subcluster" in result.obs.columns
    assert result.obs["fibroblast_subcluster"].notna().all()
    # Real signal (3 separated blobs): more than one cluster found.
    assert result.obs["fibroblast_subcluster"].nunique() > 1


def test_records_the_chosen_resolution_and_stability_summary_in_uns() -> None:
    adata = _adata(n=240)
    config = SubclusteringConfig(
        key_added="fibroblast_subcluster",
        partition={
            "method": "leiden_stability",
            "leiden_stability": {
                "resolutions": [0.2, 0.5, 1.5],
                "n_bootstraps": 4,
                "adaptive": False,
            },
        },
    )
    result = run_leiden_stability(adata, config, reduction_key="X_test_reduction")
    record = result.uns["subclustering"]["leiden_stability"]
    assert record["chosen_resolution"] in [0.2, 0.5, 1.5]
    assert len(record["summary"]) == 3


def test_deterministic_across_repeated_calls_with_the_same_seed() -> None:
    config = SubclusteringConfig(
        key_added="fibroblast_subcluster",
        partition={
            "method": "leiden_stability",
            "leiden_stability": {
                "resolutions": [0.3, 1.0],
                "n_bootstraps": 3,
                "random_state": 7,
                "adaptive": False,
            },
        },
    )
    r1 = run_leiden_stability(_adata(n=180), config, reduction_key="X_test_reduction")
    r2 = run_leiden_stability(_adata(n=180), config, reduction_key="X_test_reduction")
    assert (
        r1.uns["subclustering"]["leiden_stability"]["chosen_resolution"]
        == (r2.uns["subclustering"]["leiden_stability"]["chosen_resolution"])
    )
    pd.testing.assert_series_equal(
        r1.obs["fibroblast_subcluster"].reset_index(drop=True),
        r2.obs["fibroblast_subcluster"].reset_index(drop=True),
        check_names=False,
    )
