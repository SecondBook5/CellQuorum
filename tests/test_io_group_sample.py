"""Per-group capped reads of a large atlas.

The whole point of this reader is that three things which could be done silently are not: the
cap (which cells survived it), the gene restriction (which names were absent), and the
agreement gate (which groups could be gated at all). A second annotation column often has no
word for some of the first column's labels, and gating those groups anyway empties them --
that is the failure this reader exists to avoid, so it gets the most tests.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellquorum.core.contracts.layer_tags import get_layer_tag, set_layer_tag
from cellquorum.io.anndata import AnnDataLoadError, load_group_sample

CELL_TYPES = ["LEC"] * 40 + ["Fib"] * 200 + ["Mac"] * 60 + ["Neutrophils"] * 10
# 'Neutrophils' is deliberately outside the ref_state vocabulary: the second annotation has no
# word for it, exactly as in the real atlas.
REF_STATES = ["LEC"] * 35 + ["VEC"] * 5 + ["Fib"] * 190 + ["Other"] * 10 + ["Mac"] * 60 + [""] * 10
GENES = [f"G{i}" for i in range(30)]


@pytest.fixture
def atlas(tmp_path):
    rng = np.random.default_rng(0)
    n = len(CELL_TYPES)
    counts = sp.csr_matrix(rng.poisson(1.0, size=(n, len(GENES))).astype(np.float32))
    adata = ad.AnnData(
        X=sp.csr_matrix(np.zeros((n, len(GENES)), dtype=np.float32)),
        obs=pd.DataFrame(
            {"cell_type": CELL_TYPES, "ref_state": REF_STATES},
            index=[f"cell{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=GENES),
    )
    adata.layers["counts"] = counts
    path = tmp_path / "atlas.h5ad"
    adata.write_h5ad(path)
    return path


def _report(adata) -> dict[str, dict]:
    return {row["group"]: row for row in adata.uns["cellquorum_group_sample"]["groups"]}


def test_a_cap_takes_that_many_cells_and_none_means_all(atlas) -> None:
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None, "Fib": 25}, layer="counts"
    )
    counts = out.obs["cell_type"].value_counts().to_dict()
    assert counts == {"Fib": 25, "LEC": 40}


def test_unselected_cell_types_do_not_survive_as_empty_categories(atlas) -> None:
    """An empty level becomes an empty group in any per-group statistic built from the slice."""
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None, "Fib": 25}, layer="counts"
    )
    assert set(out.obs["cell_type"].cat.categories) == {"LEC", "Fib"}


def test_the_sample_is_reproducible_from_the_seed(atlas) -> None:
    kwargs = dict(group_column="cell_type", per_group={"Fib": 20}, layer="counts")
    first = load_group_sample(atlas, seed=7, **kwargs)
    same = load_group_sample(atlas, seed=7, **kwargs)
    other = load_group_sample(atlas, seed=8, **kwargs)
    assert list(first.obs_names) == list(same.obs_names)
    assert list(first.obs_names) != list(other.obs_names)


def test_a_groups_sample_does_not_depend_on_what_else_was_requested(atlas) -> None:
    """Two arms of one study request overlapping groups; the overlap must be the same cells.

    A single shared RNG stream gives a group different cells depending on how many groups were
    drawn before it, so a difference between two arms acquires a sampling explanation that
    cannot be ruled out. Here Fib is asked for alone, second, and second under a different
    request order.
    """
    alone = load_group_sample(
        atlas, group_column="cell_type", per_group={"Fib": 20}, layer="counts", seed=7
    )
    after_lec = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"LEC": 10, "Fib": 20},
        layer="counts",
        seed=7,
    )
    after_mac = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"Mac": 30, "Fib": 20},
        layer="counts",
        seed=7,
    )
    fib = lambda out: sorted(out.obs_names[out.obs["cell_type"] == "Fib"])  # noqa: E731
    assert fib(alone) == fib(after_lec) == fib(after_mac)


def test_two_groups_at_the_same_cap_do_not_get_the_same_row_offsets(atlas) -> None:
    """Seeding per group must not collapse to seeding every group identically."""
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"Fib": 20, "Mac": 20},
        layer="counts",
        seed=7,
    )
    positions = {
        group: sorted(
            int(name.removeprefix("cell")) - offset
            for name in out.obs_names[out.obs["cell_type"] == group]
        )
        for group, offset in (("Fib", 40), ("Mac", 240))
    }
    assert positions["Fib"] != positions["Mac"]


def test_a_cap_larger_than_the_group_keeps_the_group(atlas) -> None:
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": 1000}, layer="counts"
    )
    assert out.n_obs == 40


def test_the_agreement_gate_is_applied_per_group_not_globally(atlas) -> None:
    """Gating a group whose label the second annotation lacks would delete the population."""
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"LEC": None, "Neutrophils": None},
        agreement_column="ref_state",
        layer="counts",
    )
    report = _report(out)
    # LEC is in the ref_state vocabulary, so it is gated: 5 of 40 are labelled VEC there.
    assert report["LEC"]["gate"] == "agrees_with_ref_state"
    assert report["LEC"]["n_selected"] == 35
    # Neutrophils is not, so it keeps the cell_type call rather than being emptied.
    assert report["Neutrophils"]["gate"] == "group_column"
    assert report["Neutrophils"]["n_selected"] == 10


def test_the_report_separates_available_from_eligible(atlas) -> None:
    """A count alone cannot distinguish a small population from a gated one."""
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"Fib": 100},
        agreement_column="ref_state",
        layer="counts",
    )
    row = _report(out)["Fib"]
    assert (row["n_available"], row["n_eligible"], row["cap"], row["n_selected"]) == (
        200,
        190,
        100,
        100,
    )


def test_a_group_absent_from_the_object_is_reported_not_raised(atlas) -> None:
    """A candidate-sender list usually comes from a different object."""
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None, "GHOST": 500}, layer="counts"
    )
    assert _report(out)["GHOST"]["n_selected"] == 0
    assert out.n_obs == 40


def test_no_group_matching_anything_is_refused(atlas) -> None:
    with pytest.raises(AnnDataLoadError, match="selected 0 cells"):
        load_group_sample(atlas, group_column="cell_type", per_group={"GHOST": None})


# --- genes -------------------------------------------------------------------------------


def test_genes_are_restricted_in_the_objects_own_order(atlas) -> None:
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"LEC": None},
        genes=["G5", "G1", "G0"],
        layer="counts",
    )
    assert list(out.var_names) == ["G0", "G1", "G5"]


def test_absent_gene_names_are_recorded_rather_than_dropped_in_silence(atlas) -> None:
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"LEC": None},
        genes=["G0", "NOTAGENE"],
        layer="counts",
    )
    record = out.uns["cellquorum_group_sample"]
    assert record["genes_absent"] == ["NOTAGENE"]
    assert (record["n_genes_requested"], record["n_genes_kept"]) == (2, 1)


def test_matching_no_gene_at_all_is_refused(atlas) -> None:
    with pytest.raises(AnnDataLoadError, match="0 of 2 requested genes"):
        load_group_sample(
            atlas, group_column="cell_type", per_group={"LEC": None}, genes=["X", "Y"]
        )


# --- values ------------------------------------------------------------------------------


def test_the_requested_layer_is_what_lands_in_x(atlas) -> None:
    """X in the fixture is all zeros and counts is not, so a silent fallback would show."""
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None}, layer="counts"
    )
    assert out.X.sum() > 0
    default = load_group_sample(atlas, group_column="cell_type", per_group={"LEC": None})
    assert default.X.sum() == 0


def test_the_values_are_the_atlas_values_for_those_cells_and_genes(atlas) -> None:
    full = ad.read_h5ad(atlas)
    out = load_group_sample(
        atlas,
        group_column="cell_type",
        per_group={"Fib": 15},
        genes=["G2", "G7"],
        layer="counts",
        seed=3,
    )
    expected = full[list(out.obs_names), ["G2", "G7"]].layers["counts"]
    assert np.array_equal(np.asarray(out.X.todense()), np.asarray(expected.todense()))


def test_an_absent_layer_is_refused(atlas) -> None:
    with pytest.raises(AnnDataLoadError, match="layers/nope"):
        load_group_sample(atlas, group_column="cell_type", per_group={"LEC": None}, layer="nope")


def test_an_absent_column_is_refused_by_name(atlas) -> None:
    with pytest.raises(AnnDataLoadError, match="group_column"):
        load_group_sample(atlas, group_column="nope", per_group={"LEC": None})
    with pytest.raises(AnnDataLoadError, match="agreement_column"):
        load_group_sample(
            atlas,
            group_column="cell_type",
            per_group={"LEC": None},
            agreement_column="nope",
        )


# --- provenance --------------------------------------------------------------------------


@pytest.fixture
def tagged_atlas(tmp_path):
    """An atlas whose layers are tagged, as every atlas this engine writes is."""
    rng = np.random.default_rng(1)
    n = len(CELL_TYPES)
    adata = ad.AnnData(
        X=sp.csr_matrix(np.zeros((n, len(GENES)), dtype=np.float32)),
        obs=pd.DataFrame({"cell_type": CELL_TYPES}, index=[f"cell{i}" for i in range(n)]),
        var=pd.DataFrame(index=GENES),
    )
    adata.layers["counts"] = sp.csr_matrix(rng.poisson(1.0, size=(n, len(GENES))).astype("f4"))
    adata.layers["cellquorum_normalized"] = sp.csr_matrix(
        np.log1p(rng.gamma(1.0, 1.0, size=(n, len(GENES)))).astype("f4")
    )
    set_layer_tag(adata, "counts", kind="counts")
    set_layer_tag(adata, "cellquorum_normalized", kind="lognorm", recipe="a_recipe_v1")
    path = tmp_path / "tagged.h5ad"
    adata.write_h5ad(path)
    return path


def test_the_read_layers_tag_arrives_on_x(tagged_atlas) -> None:
    """Without this every stage declaring an expected kind refuses a correctly read atlas."""
    out = load_group_sample(
        tagged_atlas,
        group_column="cell_type",
        per_group={"LEC": None},
        layer="cellquorum_normalized",
    )
    assert get_layer_tag(out, "X") == {"kind": "lognorm", "recipe": "a_recipe_v1"}


def test_the_tag_that_travels_is_the_layers_own_and_not_the_first_one_found(
    tagged_atlas,
) -> None:
    """Two tagged layers, one read: reading counts must not inherit the normalized tag."""
    out = load_group_sample(
        tagged_atlas, group_column="cell_type", per_group={"LEC": None}, layer="counts"
    )
    assert get_layer_tag(out, "X") == {"kind": "counts", "recipe": None}


def test_reading_x_itself_looks_for_x_s_tag(tagged_atlas) -> None:
    """X is untagged in the fixture, and an untagged layer must not acquire a tag by being read."""
    out = load_group_sample(tagged_atlas, group_column="cell_type", per_group={"LEC": None})
    assert get_layer_tag(out, "X") is None


def test_an_untagged_source_is_left_untagged(atlas) -> None:
    """This reader knows where values came from, not what they are; it invents nothing."""
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None}, layer="counts"
    )
    assert get_layer_tag(out, "X") is None
    assert out.uns["cellquorum_group_sample"]["layer_tag"] is None


def test_the_record_names_the_source_and_the_totals(atlas) -> None:
    out = load_group_sample(
        atlas, group_column="cell_type", per_group={"LEC": None, "Mac": 10}, layer="counts"
    )
    record = out.uns["cellquorum_group_sample"]
    assert record["n_before"] == len(CELL_TYPES)
    assert record["n_after"] == out.n_obs == 50
    assert record["source"].endswith("atlas.h5ad")
    assert record["layer"] == "counts"
