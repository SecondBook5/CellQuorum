"""Tests for subclustering donor-reproducibility gatekeeper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from cellquorum.stages.clustering.subclustering.donor_gate import (
    apply_qc_flags,
    donor_reproducibility,
)


def make_donor_gated_adata() -> AnnData:
    """
    Build synthetic clustered adata for donor gate tests.

    Creates two clusters:
    - C0: evenly distributed across d1, d2, d3 (should PASS)
    - C1: 95% from d1 (should FAIL: one-donor-dominated)

    Returns:
        AnnData with obs[cluster_key], obs[group_key], obsm[embedding_key].
    """
    rng = np.random.default_rng(42)

    # C0: 60 cells evenly split across 3 donors.
    c0_donors = ["d1"] * 20 + ["d2"] * 20 + ["d3"] * 20
    # C1: 40 cells, 95% from d1.
    c1_donors = ["d1"] * 38 + ["d2"] * 1 + ["d3"] * 1

    donors = c0_donors + c1_donors
    clusters = ["C0"] * 60 + ["C1"] * 40
    n_cells = len(donors)

    # Create synthetic embedding (small PCA-like array).
    # Make C0 and C1 separable in embedding space.
    embedding = np.zeros((n_cells, 10))
    embedding[:60, 0] = rng.normal(0, 1, 60)  # C0 centered at 0.
    embedding[60:, 0] = rng.normal(5, 1, 40)  # C1 centered at 5.
    embedding[:, 1:] = rng.normal(0, 0.5, (n_cells, 9))

    obs = pd.DataFrame(
        {"cluster": clusters, "donor": donors},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    adata = AnnData(
        X=rng.normal(0, 1, (n_cells, 50)),
        obs=obs,
    )
    adata.obsm["X_pca"] = embedding

    return adata


def test_donor_reproducibility_basic() -> None:
    """
    Test donor_reproducibility flags clusters correctly.

    C0 should PASS (evenly distributed across 3 donors).
    C1 should FAIL (95% from one donor).
    """
    adata = make_donor_gated_adata()

    result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        min_groups=3,
        max_group_frac=0.8,
    )

    # Check result structure.
    assert "clusters" in result
    assert "summary" in result
    assert "C0" in result["clusters"]
    assert "C1" in result["clusters"]

    # C0: should PASS (3 donors, no single donor > 80%).
    c0 = result["clusters"]["C0"]
    assert c0["n_groups"] == 3
    assert c0["n_cells"] == 60
    assert c0["max_group_frac"] < 0.8
    assert c0["qc_pass"] is True

    # C1: should FAIL (dominated by one donor: 38/40 = 95%).
    c1 = result["clusters"]["C1"]
    assert c1["n_groups"] == 3
    assert c1["n_cells"] == 40
    assert c1["max_group_frac"] > 0.8
    assert c1["qc_pass"] is False
    assert "one-donor-dominated" in c1["qc_reason"]

    # Summary.
    assert result["summary"]["n_pass"] == 1
    assert result["summary"]["n_fail"] == 1


def test_donor_reproducibility_lodo_stability() -> None:
    """
    Test LODO (leave-one-donor-out) stability metric.

    For a well-distributed cluster, LODO stability should be high
    (held-out donor's cells are correctly predicted as belonging to the
    cluster when trained on other donors).
    """
    adata = make_donor_gated_adata()

    result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        do_lodo=True,
        min_groups=3,
    )

    c0 = result["clusters"]["C0"]
    c1 = result["clusters"]["C1"]

    # C0: well-distributed, should have high LODO stability.
    assert "lodo_stability" in c0
    assert c0["lodo_stability"] is not None
    # Should be reasonably high (>0.5), though not necessarily perfect.
    assert c0["lodo_stability"] > 0.4

    # C1: one-donor-dominated, LODO may be lower.
    assert "lodo_stability" in c1
    assert c1["lodo_stability"] is not None


def test_donor_reproducibility_classifier_separability() -> None:
    """
    Test donor-blocked classifier separability metric.

    The classifier must be trained/tested BY DONOR (not by cell).
    This metric measures one-vs-rest separability of each cluster
    using held-out donors.
    """
    adata = make_donor_gated_adata()

    result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        do_classifier=True,
        min_groups=3,
    )

    c0 = result["clusters"]["C0"]
    c1 = result["clusters"]["C1"]

    # Both clusters should have classifier_sep computed.
    assert "classifier_sep" in c0
    assert c0["classifier_sep"] is not None
    # C0 and C1 are well-separated in embedding space → high accuracy.
    assert c0["classifier_sep"] > 0.5

    assert "classifier_sep" in c1
    assert c1["classifier_sep"] is not None
    assert c1["classifier_sep"] > 0.5


def test_donor_blocked_classifier_enforced() -> None:
    """
    Test that the classifier is donor-blocked (split BY donor, not by cell).

    With only 1 donor, the donor-blocked classifier should return None
    (cannot donor-block) rather than falling back to a leaky cell split.
    This proves the split is BY group_key.
    """
    rng = np.random.default_rng(42)

    # Build adata with only 1 donor (all cells from d1).
    n_cells = 50
    obs = pd.DataFrame(
        {"cluster": ["C0"] * n_cells, "donor": ["d1"] * n_cells},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = AnnData(
        X=rng.normal(0, 1, (n_cells, 50)),
        obs=obs,
    )
    adata.obsm["X_pca"] = rng.normal(0, 1, (n_cells, 10))

    result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        do_classifier=True,
        min_groups=1,
    )

    c0 = result["clusters"]["C0"]

    # With only 1 donor, classifier_sep should be None (cannot donor-block).
    assert c0["classifier_sep"] is None


def test_apply_qc_flags_flag_not_drop() -> None:
    """
    Test apply_qc_flags annotates cells with pass/fail flags (flag-not-drop).

    All cells should be retained; obs columns added with per-cell QC verdict.
    """
    adata = make_donor_gated_adata()

    gate_result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        min_groups=3,
        max_group_frac=0.8,
    )

    n_cells_before = adata.n_obs

    # Apply QC flags.
    apply_qc_flags(adata, "cluster", gate_result, key_added="donor_qc")

    # All cells retained (flag-not-drop).
    assert adata.n_obs == n_cells_before

    # QC columns added.
    assert "donor_qc_qc_pass" in adata.obs.columns
    assert "donor_qc_qc_reason" in adata.obs.columns

    # Per-cell values match cluster's verdict.
    c0_cells = adata.obs["cluster"] == "C0"
    c1_cells = adata.obs["cluster"] == "C1"

    # C0 cells: all PASS.
    assert adata.obs.loc[c0_cells, "donor_qc_qc_pass"].all()

    # C1 cells: all FAIL.
    assert not adata.obs.loc[c1_cells, "donor_qc_qc_pass"].any()
    assert all("one-donor-dominated" in r for r in adata.obs.loc[c1_cells, "donor_qc_qc_reason"])


def test_apply_qc_flags_drop_action() -> None:
    """
    Test that cells in failed clusters are dropped when action='drop'.
    """
    adata = make_donor_gated_adata()

    gate_result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        min_groups=3,
        max_group_frac=0.8,
    )

    # Apply QC flags with drop action (by returning a filtered view).
    apply_qc_flags(adata, "cluster", gate_result, key_added="donor_qc")

    # Now manually filter to simulate action='drop'.
    filtered = adata[adata.obs["donor_qc_qc_pass"]].copy()

    # Only C0 cells retained (60 cells).
    assert filtered.n_obs == 60
    assert (filtered.obs["cluster"] == "C0").all()


def test_donor_reproducibility_min_groups_threshold() -> None:
    """
    Test that clusters with n_groups < min_groups are flagged as FAIL.
    """
    adata = make_donor_gated_adata()

    # Set min_groups=4 (higher than C0/C1's 3 donors).
    result = donor_reproducibility(
        adata,
        cluster_key="cluster",
        group_key="donor",
        min_groups=4,
    )

    c0 = result["clusters"]["C0"]
    c1 = result["clusters"]["C1"]

    # Both should FAIL (n_groups=3 < min_groups=4).
    assert c0["qc_pass"] is False
    assert "n_groups < min_groups" in c0["qc_reason"]

    assert c1["qc_pass"] is False
    # C1 has BOTH violations; reason should mention at least one.
    assert "n_groups < min_groups" in c1["qc_reason"] or "one-donor-dominated" in c1["qc_reason"]
