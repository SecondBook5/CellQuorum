"""Specimen-first pseudobulk aggregation and full-universe support auditing.

The unit of pseudobulk inference is a biological *specimen*: one row of pseudobulk
counts is one ``sample_col`` value. These tests pin four things:

* a repeated-measures experiment (a subject contributing several specimens that
  share a subject and a between-subject factor level) keeps every specimen
  distinct, where the legacy donor x condition key would merge them;
* declared sample-level metadata must be constant AND completely observed within a
  specimen -- never silently ``first()``-ed and never coerced to ``"nan"``;
* the legacy donor x condition path still returns exactly the counts it always did;
* the support audit is defined against the *complete* specimen universe, so a
  specimen with zero cells of a population stays visible (``population_absent``)
  instead of vanishing when the object is subset to that population.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.comparative.differential_expression.pseudobulk import (
    PseudobulkResult,
    aggregate_pseudobulk,
    build_specimen_universe,
    evaluate_pseudobulk_support,
    restrict_to_included,
)

# --------------------------------------------------------------------------- #
# Synthetic 6-subject / 12-specimen fixture                                   #
# --------------------------------------------------------------------------- #

# subject -> (between-subject factor level)
_SUBJECTS = {"C1": "ctl", "C2": "ctl", "C3": "ctl", "P2": "trt", "P3": "trt", "P4": "trt"}
# within-subject factor level -> specimen-id suffix
_SITES = {"siteA": "A", "siteB": "B"}
_SPECIMENS = [f"{s}{suffix}" for s in _SUBJECTS for suffix in _SITES.values()]
_GENES = ["g1", "g2", "g3", "g4"]


def _cells(n_per_specimen: dict[str, int], *, seed: int = 0) -> ad.AnnData:
    """Build an AnnData with ``n_per_specimen[sample_id]`` cells per specimen."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for subject, group in _SUBJECTS.items():
        for site, suffix in _SITES.items():
            sample_id = f"{subject}{suffix}"
            for _ in range(n_per_specimen.get(sample_id, 0)):
                rows.append(
                    {"sample_id": sample_id, "subject": subject, "group": group, "site": site}
                )
    obs = pd.DataFrame(rows)
    obs.index = [f"cell{i}" for i in range(len(obs))]
    counts = rng.poisson(3.0, size=(len(obs), len(_GENES))).astype(float)
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs)
    adata.var_names = _GENES
    adata.layers["counts"] = sp.csr_matrix(counts)
    return adata


def _full(n: int = 10) -> ad.AnnData:
    return _cells({s: n for s in _SPECIMENS})


_META = ["subject", "group", "site"]


# --------------------------------------------------------------------------- #
# Specimen-first aggregation                                                  #
# --------------------------------------------------------------------------- #


def test_twelve_specimens_give_twelve_profiles():
    adata = _full()
    res = aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)

    assert res.counts.shape == (12, 4)
    assert set(res.counts.index) == set(_SPECIMENS)
    assert list(res.sample_meta.index) == list(res.counts.index)
    assert res.sample_meta.loc["P3B", "subject"] == "P3"
    assert res.sample_meta.loc["P3B", "group"] == "trt"
    assert res.sample_meta.loc["P3B", "site"] == "siteB"
    assert (res.n_cells == 10).all()


def test_specimen_counts_are_the_cell_sums():
    adata = _full()
    res = aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)
    dense = adata.layers["counts"].toarray()
    for sample_id in _SPECIMENS:
        mask = (adata.obs["sample_id"] == sample_id).to_numpy()
        np.testing.assert_array_equal(res.counts.loc[sample_id].to_numpy(), dense[mask].sum(axis=0))


def test_legacy_donor_condition_key_would_merge_specimens():
    # Documents WHY specimen-first exists: the legacy key (subject x group) merges a
    # subject's two specimens into one profile -> 6, not 12.
    adata = _full()
    legacy = aggregate_pseudobulk(adata, layer="counts", donor_col="subject", condition_col="group")
    assert legacy.counts.shape[0] == 6


def test_dense_layer_is_accepted():
    adata = _full()
    adata.layers["counts"] = adata.layers["counts"].toarray()
    res = aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)
    assert res.counts.shape == (12, 4)


def test_undeclared_cell_varying_column_is_ignored():
    # Only DECLARED sample-level columns are checked; a per-cell quantity that is
    # not requested must never become (or break) sample metadata.
    adata = _full()
    adata.obs["pct_mito"] = np.arange(adata.n_obs, dtype=float)
    res = aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)
    assert "pct_mito" not in res.sample_meta.columns


def test_declared_cell_varying_column_fails():
    adata = _full()
    adata.obs["pct_mito"] = np.arange(adata.n_obs, dtype=float)
    with pytest.raises(CellQuorumDataError, match="pct_mito"):
        aggregate_pseudobulk(
            adata, layer="counts", sample_col="sample_id", metadata_cols=[*_META, "pct_mito"]
        )


# --------------------------------------------------------------------------- #
# Strict metadata validation                                                  #
# --------------------------------------------------------------------------- #


def test_conflicting_metadata_within_specimen_fails_loudly():
    adata = _full()
    idx = np.flatnonzero((adata.obs["sample_id"] == "C1A").to_numpy())[:3]
    adata.obs.iloc[idx, adata.obs.columns.get_loc("group")] = "trt"
    with pytest.raises(CellQuorumDataError) as exc:
        aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)
    msg = str(exc.value)
    # Names the specimen, the column, the observed values and the cell count.
    assert "C1A" in msg and "group" in msg
    assert "ctl" in msg and "trt" in msg
    assert "10" in msg


@pytest.mark.parametrize("column", ["site", "subject"])
def test_partial_missingness_within_specimen_fails(column):
    adata = _full()
    idx = np.flatnonzero((adata.obs["sample_id"] == "C1A").to_numpy())[:2]
    adata.obs[column] = adata.obs[column].astype(object)
    adata.obs.iloc[idx, adata.obs.columns.get_loc(column)] = np.nan
    with pytest.raises(CellQuorumDataError, match=column):
        aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)


def test_entirely_missing_declared_field_fails():
    adata = _full()
    adata.obs["site"] = adata.obs["site"].astype(object)
    adata.obs.loc[adata.obs["sample_id"] == "C1A", "site"] = np.nan
    with pytest.raises(CellQuorumDataError, match="site"):
        aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)


def test_entirely_missing_optional_field_is_allowed():
    adata = _full()
    adata.obs["batch"] = np.nan
    res = aggregate_pseudobulk(
        adata,
        layer="counts",
        sample_col="sample_id",
        metadata_cols=_META,
        optional_metadata_cols=["batch"],
    )
    assert res.sample_meta["batch"].isna().all()


def test_partially_missing_optional_field_still_fails():
    adata = _full()
    adata.obs["batch"] = np.nan
    adata.obs["batch"] = adata.obs["batch"].astype(object)
    idx = np.flatnonzero((adata.obs["sample_id"] == "C1A").to_numpy())[:2]
    adata.obs.iloc[idx, adata.obs.columns.get_loc("batch")] = "b1"
    with pytest.raises(CellQuorumDataError, match="batch"):
        aggregate_pseudobulk(
            adata,
            layer="counts",
            sample_col="sample_id",
            metadata_cols=_META,
            optional_metadata_cols=["batch"],
        )


def test_null_sample_id_is_rejected_not_coerced_to_nan_string():
    adata = _full()
    adata.obs["sample_id"] = adata.obs["sample_id"].astype(object)
    adata.obs.iloc[:3, adata.obs.columns.get_loc("sample_id")] = np.nan
    with pytest.raises(CellQuorumDataError, match="sample_id"):
        aggregate_pseudobulk(adata, layer="counts", sample_col="sample_id", metadata_cols=_META)


def test_legacy_null_donor_is_rejected():
    adata = _full()
    adata.obs["subject"] = adata.obs["subject"].astype(object)
    adata.obs.iloc[:2, adata.obs.columns.get_loc("subject")] = np.nan
    with pytest.raises(CellQuorumDataError, match="subject"):
        aggregate_pseudobulk(adata, layer="counts", donor_col="subject", condition_col="group")


def test_legacy_declared_extra_obs_must_be_constant():
    adata = _full()
    adata.obs["age"] = np.arange(adata.n_obs, dtype=float)
    with pytest.raises(CellQuorumDataError, match="age"):
        aggregate_pseudobulk(
            adata, layer="counts", donor_col="subject", condition_col="group", extra_obs=["age"]
        )


# --------------------------------------------------------------------------- #
# Argument contract                                                           #
# --------------------------------------------------------------------------- #


def test_mixing_legacy_and_specimen_arguments_is_rejected():
    with pytest.raises(ValueError, match="either"):
        aggregate_pseudobulk(
            _full(),
            layer="counts",
            sample_col="sample_id",
            donor_col="subject",
            condition_col="group",
        )


def test_no_grouping_arguments_is_rejected():
    with pytest.raises(ValueError, match="either"):
        aggregate_pseudobulk(_full(), layer="counts")


def test_legacy_requires_both_donor_and_condition():
    with pytest.raises(ValueError, match="either"):
        aggregate_pseudobulk(_full(), layer="counts", donor_col="subject")


def test_missing_layer_and_column_raise_keyerror():
    adata = _full()
    with pytest.raises(KeyError, match="Layer"):
        aggregate_pseudobulk(adata, layer="nope", sample_col="sample_id")
    with pytest.raises(KeyError, match="absent_col"):
        aggregate_pseudobulk(
            adata, layer="counts", sample_col="sample_id", metadata_cols=["absent_col"]
        )


# --------------------------------------------------------------------------- #
# Legacy equivalence                                                          #
# --------------------------------------------------------------------------- #


def _reference_legacy(adata, *, layer, donor_col, condition_col, extra_obs):
    """The pre-refactor algorithm, kept verbatim as an oracle."""
    donor = adata.obs[donor_col].astype(str)
    condition = adata.obs[condition_col].astype(str)
    group_key = donor.to_numpy() + "__" + condition.to_numpy()
    groups = pd.Index(pd.unique(group_key))
    col_index = {g: i for i, g in enumerate(groups)}
    rows = np.fromiter((col_index[g] for g in group_key), dtype=np.int64, count=len(group_key))
    cols = np.arange(len(group_key), dtype=np.int64)
    data = np.ones(len(group_key), dtype=np.float64)
    indicator = sp.csr_matrix((data, (rows, cols)), shape=(len(groups), adata.n_obs))
    m = adata.layers[layer]
    m = sp.csr_matrix(m) if not sp.issparse(m) else m
    summed = indicator @ m
    counts = pd.DataFrame(
        np.rint(np.asarray(summed.todense())).astype(np.int64),
        index=groups,
        columns=list(adata.var_names),
    )
    meta_source = adata.obs.assign(_group=group_key)
    meta_cols = [donor_col, condition_col, *(extra_obs or [])]
    meta = meta_source.groupby("_group")[meta_cols].first().reindex(groups)
    meta.index.name = None
    return counts, meta


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_legacy_path_is_identical_to_pre_refactor_algorithm(seed):
    adata = _cells({s: 7 + (i * 3) % 11 for i, s in enumerate(_SPECIMENS)}, seed=seed)
    # A subject-level covariate that is constant within donor x condition.
    adata.obs["batch"] = adata.obs["subject"].map({"C1": "b1", "C2": "b1", "C3": "b2"}).fillna("b3")
    new = aggregate_pseudobulk(
        adata, layer="counts", donor_col="subject", condition_col="group", extra_obs=["batch"]
    )
    old_counts, old_meta = _reference_legacy(
        adata, layer="counts", donor_col="subject", condition_col="group", extra_obs=["batch"]
    )
    pd.testing.assert_frame_equal(new.counts, old_counts)
    pd.testing.assert_frame_equal(new.sample_meta, old_meta)


def test_legacy_result_still_constructible_without_new_fields():
    # test_de_pairing_resolution builds PseudobulkResult(counts=, sample_meta=) only.
    pb = PseudobulkResult(
        counts=pd.DataFrame([[1, 2]], index=["s"], columns=["a", "b"]),
        sample_meta=pd.DataFrame({"x": [1]}, index=["s"]),
    )
    assert pb.n_cells is None


# --------------------------------------------------------------------------- #
# Specimen universe + support audit                                           #
# --------------------------------------------------------------------------- #


def test_specimen_universe_lists_every_specimen_with_total_cells():
    uni = build_specimen_universe(_full().obs, sample_col="sample_id", metadata_cols=_META)
    assert len(uni) == 12
    assert set(uni.index) == set(_SPECIMENS)
    assert (uni["n_cells_total"] == 10).all()
    assert uni.loc["C2B", "subject"] == "C2"


def test_specimen_universe_rejects_a_sample_mapped_to_two_subjects():
    adata = _full()
    adata.obs["subject"] = adata.obs["subject"].astype(object)
    idx = np.flatnonzero((adata.obs["sample_id"] == "P2A").to_numpy())[:4]
    adata.obs.iloc[idx, adata.obs.columns.get_loc("subject")] = "P3"
    with pytest.raises(CellQuorumDataError, match="P2A"):
        build_specimen_universe(adata.obs, sample_col="sample_id", metadata_cols=_META)


def _population_subset(per_specimen: dict[str, int]) -> tuple[ad.AnnData, pd.DataFrame]:
    """A population subset of the full experiment plus the full specimen universe."""
    full = _full(30)
    universe = build_specimen_universe(full.obs, sample_col="sample_id", metadata_cols=_META)
    sub = _cells(per_specimen)
    return sub, universe


def test_absent_population_stays_visible_in_support_and_is_never_zero_profile():
    # 'C2A' has ZERO cells of this population; 'P3B' has only 8 (below threshold).
    n = {s: 25 for s in _SPECIMENS}
    n["C2A"] = 0
    n["P3B"] = 8
    sub, universe = _population_subset(n)

    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    # The aggregator only knows observed cells: no row for the absent specimen,
    # and in particular no all-zero pseudobulk row.
    assert "C2A" not in pb.counts.index
    assert (pb.counts.sum(axis=1) > 0).all()

    support = evaluate_pseudobulk_support(
        pb, sample_metadata=universe, subject_col="subject", min_cells_per_sample=20
    )
    table = support.table
    assert len(table) == 12  # the whole universe, not just observed specimens
    assert list(table.columns) == [
        "sample_id",
        "subject_id",
        "group",
        "site",
        "n_cells",
        "library_size",
        "observed_population",
        "included_for_expression",
        "exclusion_reason",
    ]
    by_id = table.set_index("sample_id")

    absent = by_id.loc["C2A"]
    assert absent["n_cells"] == 0 and absent["library_size"] == 0
    assert not absent["observed_population"]
    assert not absent["included_for_expression"]
    assert absent["exclusion_reason"] == "population_absent"
    assert absent["subject_id"] == "C2" and absent["group"] == "ctl"

    low = by_id.loc["P3B"]
    assert low["n_cells"] == 8 and low["observed_population"]
    assert not low["included_for_expression"]
    assert low["exclusion_reason"] == "below_min_cells"

    ok = by_id.loc["C1A"]
    assert ok["included_for_expression"] and pd.isna(ok["exclusion_reason"])
    assert ok["library_size"] == int(pb.counts.loc["C1A"].sum())

    assert support.n_included == 10
    assert set(support.excluded_ids) == {"C2A", "P3B"}

    kept = restrict_to_included(pb, support)
    assert kept.counts.shape[0] == 10
    assert "P3B" not in kept.counts.index and "C2A" not in kept.counts.index
    assert list(kept.sample_meta.index) == list(kept.counts.index)
    assert list(kept.n_cells.index) == list(kept.counts.index)


def test_support_without_threshold_only_excludes_absent_specimens():
    n = {s: 25 for s in _SPECIMENS}
    n["C2A"] = 0
    n["P3B"] = 2
    sub, universe = _population_subset(n)
    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    support = evaluate_pseudobulk_support(pb, sample_metadata=universe, subject_col="subject")
    assert support.n_included == 11
    assert support.excluded_ids == ["C2A"]


def test_zero_library_specimen_is_excluded_with_reason():
    n = {s: 25 for s in _SPECIMENS}
    sub, universe = _population_subset(n)
    mask = (sub.obs["sample_id"] == "P4A").to_numpy()
    dense = sub.layers["counts"].toarray()
    dense[mask] = 0
    sub.layers["counts"] = sp.csr_matrix(dense)
    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    support = evaluate_pseudobulk_support(pb, sample_metadata=universe, subject_col="subject")
    row = support.table.set_index("sample_id").loc["P4A"]
    assert row["observed_population"] and not row["included_for_expression"]
    assert row["exclusion_reason"] == "zero_library_size"


def test_support_rejects_a_specimen_outside_the_universe():
    sub = _full(30)
    universe = build_specimen_universe(sub.obs, sample_col="sample_id", metadata_cols=_META)
    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    with pytest.raises(CellQuorumDataError, match="not in the specimen universe"):
        evaluate_pseudobulk_support(pb, sample_metadata=universe.drop(index="P2B"))


def test_support_rejects_metadata_disagreeing_with_the_universe():
    sub = _full(30)
    universe = build_specimen_universe(sub.obs, sample_col="sample_id", metadata_cols=_META)
    universe.loc["C1A", "group"] = "trt"
    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    with pytest.raises(CellQuorumDataError, match="C1A"):
        evaluate_pseudobulk_support(pb, sample_metadata=universe, subject_col="subject")


def test_support_rejects_nonpositive_min_cells():
    sub = _full(30)
    universe = build_specimen_universe(sub.obs, sample_col="sample_id", metadata_cols=_META)
    pb = aggregate_pseudobulk(sub, layer="counts", sample_col="sample_id", metadata_cols=_META)
    with pytest.raises(ValueError, match="min_cells_per_sample"):
        evaluate_pseudobulk_support(pb, sample_metadata=universe, min_cells_per_sample=0)


def test_support_requires_aggregated_cell_counts():
    pb = PseudobulkResult(
        counts=pd.DataFrame([[1]], index=["s"], columns=["a"]),
        sample_meta=pd.DataFrame(index=["s"]),
    )
    with pytest.raises(ValueError, match="n_cells"):
        evaluate_pseudobulk_support(pb, sample_metadata=pd.DataFrame(index=["s"]))
