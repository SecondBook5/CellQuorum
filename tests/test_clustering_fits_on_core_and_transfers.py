"""Cluster boundaries must be fitted on core cells, with labels transferred to the rest.

A cluster boundary is a cohort-derived quantity — it is inferred from whoever took part in
the graph. The failure this protects against, measured on the fixture below rather than
assumed, is that a group of damaged cells forms a cluster of its own, which annotation then
gives a cell-type name and which gets reported as biology.

Clustering is a separate analysis from the manifold in the eligibility table precisely
because a cell can be legitimately projected into an embedding and still be barred from
shaping clusters. These tests hold both halves:

    1. Non-core cells cannot change the partition   (no leak)
    2. Non-core cells still receive a label         (no silent drop)

and one structural claim that is easy to break by accident: the neighbors graph left on the
object must still cover every cell, because the embeddings stage raises ``NeighborsMissing``
otherwise and UMAP would stop working for the whole pipeline.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellquorum.stages.clustering.neighbors_leiden import (
    LABEL_SOURCE_COLUMN,
    LeidenMethod,
    transfer_cluster_labels,
)
from cellquorum.stages.qc.eligibility import Analysis, EligibilityMasks, Permission

FIT_COLUMN = EligibilityMasks.column_name(Analysis.CLUSTERING, Permission.FIT)
REP = "X_pca"


def _cohort(
    n_per_cluster: int = 120,
    n_bridge: int = 60,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """Two well-separated populations plus non-core cells strung between them.

    The bridge spans the gap densely enough to be well connected to itself, which is what
    makes it form its own community when it is allowed into the fit. Spreading it along the
    gap rather than piling it at the midpoint also gives the transfer test two ends with
    unambiguous nearest populations.
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(-6.0, 0.45, size=(n_per_cluster, 10))
    right = rng.normal(6.0, 0.45, size=(n_per_cluster, 10))

    # A dense chain of cells spanning the gap, evenly spaced so consecutive ones are close.
    ramp = np.linspace(-6.0, 6.0, n_bridge)[:, None]
    bridge = ramp + rng.normal(0.0, 0.12, size=(n_bridge, 10))

    coordinates = np.vstack([left, right, bridge]).astype(np.float32)
    n_core = 2 * n_per_cluster
    obs = pd.DataFrame(
        {
            FIT_COLUMN: [True] * n_core + [False] * n_bridge,
            "is_bridge": [False] * n_core + [True] * n_bridge,
        },
        index=[f"cell_{i}" for i in range(len(coordinates))],
    )
    adata = ad.AnnData(
        X=np.zeros((len(coordinates), 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.obsm[REP] = coordinates
    return adata


def _cluster(adata: ad.AnnData, **overrides: object) -> object:
    """Run Leiden the way the stage would."""
    config: dict[str, object] = {
        "n_neighbors": 15,
        "resolution": 1.0,
        "random_state": 0,
        "key_added": "leiden",
        "use_rep": REP,
    }
    config.update(overrides)
    return LeidenMethod()._run(adata, config, context=None)


# ═══ 1. No leak: excluded cells cannot become a cluster of their own ════════════════
#
# The harm here was measured, not assumed. An earlier version of these tests asserted that
# the bridge *merges* the two populations, and it does not: a chain dense enough to connect
# them is also dense enough for Leiden to call it its own community. Unmasked, the 60 bridge
# cells produced two clusters that were 100% bridge (n=21 and n=31) while the populations
# stayed separate. So the failure to protect against is a spurious cluster made of excluded
# cells — which is worse than a merge, because annotation will give it a cell-type name and
# it will be reported as biology.


def _non_core_fraction_per_cluster(adata: ad.AnnData) -> pd.Series:
    """Fraction of each cluster that QC excluded from fitting."""
    non_core = ~adata.obs[FIT_COLUMN].to_numpy(bool)
    return (
        pd.Series(non_core, index=adata.obs_names)
        .groupby(adata.obs["leiden"].astype(str).to_numpy())
        .mean()
    )


def test_excluded_cells_do_not_form_clusters_of_their_own() -> None:
    """No cluster may consist of cells that had no say in the partition.

    Such a cluster is the one that gets annotated as a novel cell type and written up.
    """
    adata = _cohort()
    _cluster(adata)

    fractions = _non_core_fraction_per_cluster(adata)
    spurious = fractions[fractions > 0.9]
    assert spurious.empty, (
        f"clusters {list(spurious.index)} are >90% QC-excluded cells, so the partition was "
        f"shaped by cells forbidden from fitting it"
    )


def test_the_control_excluded_cells_really_do_form_their_own_clusters_unmasked() -> None:
    """Without the mask the leak is real, so the test above is not vacuous."""
    unmasked = _cohort()
    marks = unmasked.obs[FIT_COLUMN].copy()
    del unmasked.obs[FIT_COLUMN]
    _cluster(unmasked)
    unmasked.obs[FIT_COLUMN] = marks

    fractions = _non_core_fraction_per_cluster(unmasked)
    assert (fractions > 0.9).any(), (
        "the unmasked run produced no cluster dominated by excluded cells, so the masked "
        "assertion above does not demonstrate the mask is doing anything — strengthen the "
        "fixture"
    )


def test_the_two_real_populations_stay_separate() -> None:
    """Whatever else happens, the actual biology must not be merged."""
    adata = _cohort()
    _cluster(adata)

    labels = adata.obs["leiden"].astype(str).to_numpy()
    assert not (set(labels[:120]) & set(labels[120:240]))


def test_the_partition_matches_one_fitted_without_the_bridge_present() -> None:
    """Fitting with the bridge masked must equal fitting on a cohort that never had it.

    The reference-immutability property for a partition: not "similar clustering" but the
    same clustering, since Leiden must never have seen those cells.
    """
    full = _cohort()
    _cluster(full)

    core_only = _cohort()
    core_only = core_only[core_only.obs[FIT_COLUMN].to_numpy(bool)].copy()
    del core_only.obs[FIT_COLUMN]
    _cluster(core_only)

    fitted = full.obs.loc[core_only.obs_names, "leiden"].astype(str)
    reference = core_only.obs["leiden"].astype(str)
    assert (fitted.to_numpy() == reference.to_numpy()).all()


# ═══ 2. No silent drop: every cell keeps a label ════════════════════════════════════


def test_every_cell_receives_a_cluster_label() -> None:
    """Withholding labels would delete non-core cells from every cluster-grouped stage."""
    adata = _cohort()
    _cluster(adata)

    assert adata.obs["leiden"].notna().all()
    assert len(adata.obs["leiden"]) == adata.n_obs


def test_transferred_labels_come_from_the_fitted_category_set() -> None:
    """Transfer must not invent a cluster that Leiden never found."""
    adata = _cohort()
    _cluster(adata)

    core = adata.obs[FIT_COLUMN].to_numpy(bool)
    fitted_labels = set(adata.obs["leiden"].astype(str)[core])
    transferred = set(adata.obs["leiden"].astype(str)[~core])

    assert transferred <= fitted_labels


def test_label_source_records_which_cells_were_only_transferred() -> None:
    """A transferred label must be distinguishable from a fitted one after the run."""
    adata = _cohort()
    _cluster(adata)

    source = adata.obs[LABEL_SOURCE_COLUMN]
    core = adata.obs[FIT_COLUMN].to_numpy(bool)

    assert (source[core] == "fitted").all()
    assert (source[~core] == "transferred").all()


def test_bridge_cells_are_assigned_to_whichever_population_they_sit_nearest() -> None:
    """Transfer must be spatially sensible, not arbitrary.

    The bridge spans the gap, so its ends must inherit labels from the population they sit
    next to — otherwise the vote is not doing what it claims. Each population splits into
    sub-clusters at resolution 1.0, so the assertion is membership in that population's label
    set rather than equality with one nominated cell's label.
    """
    adata = _cohort()
    _cluster(adata)

    labels = adata.obs["leiden"].astype(str).to_numpy()
    left, right = set(labels[:120]), set(labels[120:240])
    bridge = labels[240:]

    assert bridge[0] in left, "the bridge end nearest the left population was mislabelled"
    assert bridge[-1] in right, "the bridge end nearest the right population was mislabelled"


# ═══ 3. The graph downstream stages depend on ═══════════════════════════════════════


def test_a_full_size_neighbors_graph_is_left_on_the_object() -> None:
    """UMAP and PAGA raise NeighborsMissing without it, for every cell.

    Fitting on a subset must not shrink what the next stage reads, or protecting the
    partition would break visualisation for the whole pipeline.
    """
    adata = _cohort()
    _cluster(adata)

    assert "connectivities" in adata.obsp
    assert adata.obsp["connectivities"].shape == (adata.n_obs, adata.n_obs)
    assert "neighbors" in adata.uns


# ═══ 4. Degradation and the transfer primitive ══════════════════════════════════════


def test_a_dataset_without_graded_qc_behaves_as_before() -> None:
    """Absent QC columns must not become a hidden dependency."""
    adata = _cohort()
    del adata.obs[FIT_COLUMN]
    result = _cluster(adata)

    assert adata.obs["leiden"].notna().all()
    assert (adata.obs[LABEL_SOURCE_COLUMN] == "fitted").all()
    assert not any("transfer" in note for note in result.notes)


def test_an_empty_fit_population_falls_back_rather_than_clustering_nothing() -> None:
    """An all-False mask is a misconfiguration, not an instruction to partition zero cells."""
    adata = _cohort()
    adata.obs[FIT_COLUMN] = False
    _cluster(adata)

    assert adata.obs["leiden"].notna().all()


def test_the_run_records_how_many_cells_fitted_and_how_many_were_transferred() -> None:
    """A scope decision nobody can see after the fact is the failure being designed out."""
    adata = _cohort(n_per_cluster=120, n_bridge=60)
    result = _cluster(adata)

    scope = [note for note in result.notes if "QC-permitted" in note]
    assert len(scope) == 1
    assert "240" in scope[0]
    assert "60" in scope[0]


@pytest.mark.parametrize("k", [1, 5, 15])
def test_transfer_assigns_the_nearest_reference_label(k: int) -> None:
    """The primitive on its own: a query beside a group of cells inherits their label.

    Each class needs more than ``k`` members, or the vote reaches across the gap and returns
    the global majority instead — correct kNN behaviour, but not what is being checked here.
    """
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 0.1, size=(20, 2))
    far = rng.normal(10.0, 0.1, size=(20, 2))
    reference = np.vstack([near, far])
    labels = pd.Series(["a"] * 20 + ["b"] * 20)

    got = transfer_cluster_labels(reference, labels, np.array([[0.0, 0.0], [10.0, 10.0]]), k)
    assert list(got) == ["a", "b"]


def test_transfer_caps_k_at_the_reference_size() -> None:
    """Asking for more neighbours than reference cells must not raise.

    A tiny core is a QC misconfiguration that should still produce labels; sklearn would
    otherwise raise on ``n_neighbors > n_samples``.
    """
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = pd.Series(["a", "b"])

    got = transfer_cluster_labels(reference, labels, np.array([[0.1, 0.1]]), n_neighbors=50)
    assert len(got) == 1
