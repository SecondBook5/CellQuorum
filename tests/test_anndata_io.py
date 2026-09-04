"""Tests for CellQuorum AnnData I/O utilities."""

from __future__ import annotations

# Import Path for pytest tmp_path annotations.
from pathlib import Path

# Import AnnData for test object construction.
import anndata as ad

# Import NumPy for deterministic test matrices.
import numpy as np

# Import pandas for AnnData obs/var metadata.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import AnnData I/O utilities under test.
from cellquorum.io import (
    AnnDataLoadError,
    load_adata,
    normalize_adata_path,
    validate_adata_path,
)


def make_test_adata() -> ad.AnnData:
    """
    Build a small AnnData object for I/O tests.

    Returns:
        Small AnnData object with deterministic names and values.
    """

    # Build a deterministic matrix.
    matrix = np.array(
        [
            [1.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ]
    )

    # Build observation metadata.
    obs = pd.DataFrame(
        {
            "sample": ["sample_a", "sample_b"],
        },
        index=["cell_1", "cell_2"],
    )

    # Build variable metadata.
    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])

    # Return the AnnData object.
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_normalize_adata_path_accepts_string_path(tmp_path: Path) -> None:
    """
    Verify AnnData path normalization accepts string paths.

    User-facing config values commonly arrive as strings.
    """

    # Build a string path.
    path = tmp_path / "example.h5ad"

    # Normalize the path.
    normalized = normalize_adata_path(str(path))

    # Confirm the normalized path is a Path object.
    assert normalized == path


def test_normalize_adata_path_accepts_path_object(tmp_path: Path) -> None:
    """
    Verify AnnData path normalization accepts Path objects.

    Programmatic callers often pass pathlib paths directly.
    """

    # Build a Path object.
    path = tmp_path / "example.h5ad"

    # Normalize the path.
    normalized = normalize_adata_path(path)

    # Confirm the path is preserved.
    assert normalized == path


def test_normalize_adata_path_rejects_empty_string() -> None:
    """
    Verify AnnData path normalization rejects empty string paths.

    Path('') becomes '.', so this must be rejected before Path conversion.
    """

    # Confirm empty paths fail clearly.
    with pytest.raises(AnnDataLoadError, match="cannot be empty"):
        normalize_adata_path("")


def test_validate_adata_path_accepts_existing_h5ad_file(tmp_path: Path) -> None:
    """
    Verify AnnData path validation accepts an existing h5ad file.

    The validator should only check path-level constraints, not read contents.
    """

    # Build a valid h5ad path.
    path = tmp_path / "input.h5ad"

    # Write a valid AnnData object.
    make_test_adata().write_h5ad(path)

    # Validate the path.
    validated = validate_adata_path(path)

    # Confirm the validated path is returned.
    assert validated == path


def test_validate_adata_path_rejects_missing_file(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects missing files.

    Missing inputs should fail before AnnData attempts to read them.
    """

    # Build a missing file path.
    path = tmp_path / "missing.h5ad"

    # Confirm missing files fail clearly.
    with pytest.raises(AnnDataLoadError, match="does not exist"):
        validate_adata_path(path)


def test_validate_adata_path_rejects_directory(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects directories.

    Users should provide an h5ad file, not a directory.
    """

    # Confirm directories fail clearly.
    with pytest.raises(AnnDataLoadError, match="not a file"):
        validate_adata_path(tmp_path)


def test_validate_adata_path_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """
    Verify AnnData path validation rejects unsupported file suffixes.

    This first input loader intentionally supports h5ad only.
    """

    # Build an unsupported file path.
    path = tmp_path / "input.csv"

    # Write placeholder contents.
    path.write_text("not,h5ad\n", encoding="utf-8")

    # Confirm unsupported suffixes fail clearly.
    with pytest.raises(AnnDataLoadError, match="supports AnnData input only as '.h5ad'"):
        validate_adata_path(path)


def test_load_adata_reads_h5ad_file(tmp_path: Path) -> None:
    """
    Verify load_adata reads an h5ad file into AnnData.

    This is the minimal data-loading path needed before pipeline execution can
    populate context.adata.
    """

    # Build a test AnnData object.
    expected = make_test_adata()

    # Write the object to h5ad.
    path = tmp_path / "input.h5ad"
    expected.write_h5ad(path)

    # Load the AnnData object.
    observed = load_adata(path)

    # Confirm the returned object type.
    assert isinstance(observed, ad.AnnData)

    # Confirm shape round-tripped.
    assert observed.shape == expected.shape

    # Confirm observation names round-tripped.
    assert list(observed.obs_names) == ["cell_1", "cell_2"]

    # Confirm variable names round-tripped.
    assert list(observed.var_names) == ["gene_1", "gene_2", "gene_3"]

    # Confirm observation metadata round-tripped.
    assert observed.obs["sample"].tolist() == ["sample_a", "sample_b"]

    # Confirm matrix values round-tripped.
    np.testing.assert_array_equal(observed.X, expected.X)


def test_load_adata_accepts_string_path(tmp_path: Path) -> None:
    """
    Verify load_adata accepts string paths.

    YAML/config values will typically pass file paths as strings.
    """

    # Build and write a test AnnData file.
    path = tmp_path / "input.h5ad"
    make_test_adata().write_h5ad(path)

    # Load through a string path.
    observed = load_adata(str(path))

    # Confirm the file was loaded.
    assert observed.shape == (2, 3)


def test_load_adata_rejects_corrupt_h5ad_file(tmp_path: Path) -> None:
    """
    Verify load_adata wraps AnnData/HDF5 read failures.

    Corrupt h5ad files should raise a CellQuorum-specific data error.
    """

    # Build a corrupt h5ad path.
    path = tmp_path / "corrupt.h5ad"

    # Write invalid h5ad contents.
    path.write_text("this is not a valid h5ad file", encoding="utf-8")

    # Confirm corrupt files fail clearly.
    with pytest.raises(AnnDataLoadError, match="Failed to read AnnData file"):
        load_adata(path)


def test_load_adata_rejects_non_h5ad_path_before_reading(tmp_path: Path) -> None:
    """
    Verify load_adata rejects unsupported suffixes before reading.

    This keeps error messages clear for common user mistakes.
    """

    # Build an unsupported file path.
    path = tmp_path / "input.txt"

    # Write placeholder contents.
    path.write_text("not h5ad", encoding="utf-8")

    # Confirm unsupported files fail at validation time.
    with pytest.raises(AnnDataLoadError, match="supports AnnData input only"):
        load_adata(path)


def make_celltype_adata() -> ad.AnnData:
    """
    Build a small AnnData with a ``cell_type`` obs column for subset tests.

    Returns:
        AnnData with four cells across two cell types.
    """

    matrix = np.arange(12, dtype=float).reshape(4, 3)
    obs = pd.DataFrame(
        {"cell_type": ["Fibroblasts", "T/NK", "Fibroblasts", "Mast"]},
        index=["c1", "c2", "c3", "c4"],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_load_adata_subset_keeps_only_matching_rows(tmp_path: Path) -> None:
    """
    Verify load_adata restricts to rows whose column value is in the values.

    This is the load-time cell-type restriction a hypothesis relies on so it
    never has to pre-slice a separate file.
    """

    path = tmp_path / "input.h5ad"
    make_celltype_adata().write_h5ad(path)

    observed = load_adata(path, subset_column="cell_type", subset_values=["Fibroblasts"])

    # Only the two Fibroblast rows survive, in original order.
    assert list(observed.obs_names) == ["c1", "c3"]
    assert observed.obs["cell_type"].tolist() == ["Fibroblasts", "Fibroblasts"]


def test_load_adata_subset_records_provenance(tmp_path: Path) -> None:
    """
    Verify the applied subset is recorded on uns for run provenance.

    The run reads this to log n_before/n_after so the cut is never silent.
    """

    path = tmp_path / "input.h5ad"
    make_celltype_adata().write_h5ad(path)

    observed = load_adata(path, subset_column="cell_type", subset_values=["Fibroblasts"])

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["column"] == "cell_type"
    assert prov["values"] == ["Fibroblasts"]
    assert prov["n_before"] == 4
    assert prov["n_after"] == 2


def test_load_adata_subset_rejects_unknown_column(tmp_path: Path) -> None:
    """
    Verify a subset on a missing obs column fails loudly.

    A typo'd column must not silently return the whole object.
    """

    path = tmp_path / "input.h5ad"
    make_celltype_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="not found in obs"):
        load_adata(path, subset_column="celltype", subset_values=["Fibroblasts"])


def test_load_adata_subset_rejects_zero_match(tmp_path: Path) -> None:
    """
    Verify a subset that matches no rows fails rather than running on 0 cells.

    An empty slice is almost always a mislabeled value.
    """

    path = tmp_path / "input.h5ad"
    make_celltype_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="matched 0 of"):
        load_adata(path, subset_column="cell_type", subset_values=["Neuron"])


def test_load_adata_rejects_half_specified_subset(tmp_path: Path) -> None:
    """
    Verify a subset needs both column and values, or neither.

    A half-specified subset is a programming error and must fail clearly.
    """

    path = tmp_path / "input.h5ad"
    make_celltype_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="both subset_column and subset_values"):
        load_adata(path, subset_column="cell_type")


def make_two_annotation_adata() -> ad.AnnData:
    """
    Build an AnnData with two annotation columns that partly disagree.

    ``ref_state`` is deliberately missing a ``Mast`` level, the way a granular
    reference atlas can simply have no word for a population the marker-based
    labels do name.

    Returns:
        AnnData with five cells, two annotation columns, and one discordant cell.
    """

    matrix = np.arange(15, dtype=float).reshape(5, 3)
    obs = pd.DataFrame(
        {
            "cell_type": ["Fibroblasts", "T/NK", "Fibroblasts", "Mast", "T/NK"],
            "ref_state": ["Fibroblasts", "T/NK", "T/NK", "Fibroblasts", "T/NK"],
        },
        index=["c1", "c2", "c3", "c4", "c5"],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_load_adata_agreement_drops_cells_the_annotations_disagree_about(
    tmp_path: Path,
) -> None:
    """
    Verify require_agreement keeps only cells both annotations call the same thing.

    Cells two independent annotations disagree about are the ones most likely to
    be misassigned, and they matter most when two slices are compared: one built
    by agreement and one built by a single label are not filtered equally, so a
    difference between them can be the filter rather than the biology.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    observed = load_adata(
        path,
        subset_column="cell_type",
        subset_values=["Fibroblasts"],
        agreement_column="ref_state",
    )

    # c3 is called Fibroblast by one annotation and T/NK by the other, so it goes.
    assert list(observed.obs_names) == ["c1"]


def test_load_adata_agreement_records_what_it_cost(tmp_path: Path) -> None:
    """
    Verify provenance separates cells selected from cells that survived agreement.

    Recording only the survivors makes a concordance filter indistinguishable from
    a small population; the pair of numbers is what tells a reader whether the
    requirement did anything and how much.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    observed = load_adata(
        path,
        subset_column="cell_type",
        subset_values=["Fibroblasts"],
        agreement_column="ref_state",
    )

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["require_agreement"] == "ref_state"
    assert prov["n_selected"] == 2
    assert prov["n_discordant"] == 1
    assert prov["n_after"] == 1


def test_load_adata_agreement_records_none_when_not_requested(tmp_path: Path) -> None:
    """
    Verify a plain subset reports no agreement rather than an implied zero.

    ``n_discordant: 0`` on a run that never asked for concordance would read as
    "the annotations fully agreed", which is a claim the run did not make.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    observed = load_adata(path, subset_column="cell_type", subset_values=["Fibroblasts"])

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["require_agreement"] is None
    assert prov["n_discordant"] is None
    assert prov["n_selected"] == prov["n_after"] == 2


def test_load_adata_agreement_rejects_a_label_the_other_column_cannot_express(
    tmp_path: Path,
) -> None:
    """
    Verify a label absent from the agreement column's vocabulary is an error.

    This is the failure that makes the feature dangerous rather than merely
    strict. A reference that has no ``Mast`` level cannot agree with ``Mast``, so
    the requirement silently deletes the entire population -- and the cell count
    alone cannot distinguish that from a population the reference genuinely
    rejects. Failing here forces the choice to be made explicitly.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="vocabulary mismatch"):
        load_adata(
            path,
            subset_column="cell_type",
            subset_values=["Mast"],
            agreement_column="ref_state",
        )


def test_load_adata_agreement_rejects_an_unknown_agreement_column(tmp_path: Path) -> None:
    """
    Verify a typo'd agreement column fails instead of being ignored.

    Silently skipping it would produce an unfiltered slice from a config that
    reads as if it filtered.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="require_agreement"):
        load_adata(
            path,
            subset_column="cell_type",
            subset_values=["Fibroblasts"],
            agreement_column="refstate",
        )


def test_load_adata_agreement_requires_something_to_agree_with(tmp_path: Path) -> None:
    """
    Verify an agreement column without a subset column is refused.

    With no selected label there is no reference value to compare against, so the
    setting could only be silently dropped.
    """

    path = tmp_path / "input.h5ad"
    make_two_annotation_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="requires a subset_column"):
        load_adata(path, agreement_column="ref_state")


def make_clustered_adata() -> ad.AnnData:
    """
    Build an AnnData with a ``leiden`` column holding one artifact cluster.

    Shaped like the real case the exclusion rule exists for: a clustered atlas
    where one cluster is ambient debris drawn from several lineages, so it cannot
    be dropped by cell type and must be dropped by cluster.

    Returns:
        AnnData with six cells over three clusters; cluster ``22`` is the artifact
        and holds one cell of two different lineages.
    """

    matrix = np.arange(18, dtype=float).reshape(6, 3)
    obs = pd.DataFrame(
        {
            "cell_type": ["Fibroblasts", "Fibroblasts", "T/NK", "Fibroblasts", "T/NK", "Mast"],
            "leiden": ["0", "0", "1", "22", "22", "1"],
        },
        index=["c1", "c2", "c3", "c4", "c5", "c6"],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return ad.AnnData(X=matrix, obs=obs, var=var)


def test_load_adata_exclude_drops_a_cluster_without_naming_the_others(tmp_path: Path) -> None:
    """
    Verify an exclusion alone keeps every cell the rule does not name.

    This is the whole point of having the rule: dropping one artifact cluster from
    a 39-cluster partition by inclusion would mean listing the 38 real ones, which
    is unreadable and silently incomplete the next time the object gains a
    cluster. Exclusion states the one fact the analyst actually established.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    observed = load_adata(path, exclude_column="leiden", exclude_values=["22"])

    assert list(observed.obs_names) == ["c1", "c2", "c3", "c6"]
    assert "22" not in set(observed.obs["leiden"].astype(str))


def test_load_adata_exclude_composes_with_a_subset(tmp_path: Path) -> None:
    """
    Verify a lineage slice and an artifact exclusion apply together.

    A per-lineage analysis on a clustered atlas needs both at once: the debris
    cluster contains cells carrying the lineage label being analysed, so
    subsetting on the label alone still admits them.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    observed = load_adata(
        path,
        subset_column="cell_type",
        subset_values=["Fibroblasts"],
        exclude_column="leiden",
        exclude_values=["22"],
    )

    # c4 is a Fibroblast-labelled cell inside the artifact cluster.
    assert list(observed.obs_names) == ["c1", "c2"]


def test_load_adata_exclude_records_what_it_cost_this_slice(tmp_path: Path) -> None:
    """
    Verify n_excluded counts cells removed FROM the slice, not the cluster's size.

    The number a reader needs is what the exclusion did to this analysis. Logging
    the cluster's total instead would overstate the cut on every subset and make
    the provenance numbers stop adding up.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    observed = load_adata(
        path,
        subset_column="cell_type",
        subset_values=["Fibroblasts"],
        exclude_column="leiden",
        exclude_values=["22"],
    )

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["exclude_column"] == "leiden"
    assert prov["exclude_values"] == ["22"]
    assert prov["n_before"] == 6
    assert prov["n_selected"] == 3
    # Cluster 22 holds two cells; only one of them is in this lineage slice.
    assert prov["n_excluded"] == 1
    assert prov["n_after"] == 2


def test_load_adata_exclude_only_records_no_subset_column(tmp_path: Path) -> None:
    """
    Verify an exclusion-only load reports no inclusion rule rather than a fake one.

    ``column: leiden`` on a run that only excluded would read as "restricted to
    these clusters", which is the opposite of what happened.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    observed = load_adata(path, exclude_column="leiden", exclude_values=["22"])

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["column"] is None
    assert prov["values"] == []
    assert prov["n_selected"] == 6
    assert prov["n_excluded"] == 2
    assert prov["n_after"] == 4


def test_load_adata_reports_no_exclusion_when_none_was_asked_for(tmp_path: Path) -> None:
    """
    Verify a plain subset reports no exclusion rather than an implied zero.

    ``n_excluded: 0`` on a run that never excluded anything would read as "the
    artifact clusters were checked and none were in this slice", which is a claim
    the run did not make.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    observed = load_adata(path, subset_column="cell_type", subset_values=["Fibroblasts"])

    prov = observed.uns["cellquorum_input_subset"]
    assert prov["exclude_column"] is None
    assert prov["exclude_values"] is None
    assert prov["n_excluded"] is None


def test_load_adata_exclude_rejects_ids_this_partition_does_not_have(tmp_path: Path) -> None:
    """
    Verify an excluded value outside the column's vocabulary is an error.

    This is the failure the rule is most likely to hit in practice. Leiden ids
    belong to one clustering run, not to the cells, so a debris mask written
    against an earlier partition names clusters this object does not have: it
    removes nothing while the config reads as though the artifact was handled --
    strictly worse than no mask at all, because its presence stops anyone looking.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="are not values of"):
        load_adata(path, exclude_column="leiden", exclude_values=["18", "22"])


def test_load_adata_exclude_rejects_an_unknown_column(tmp_path: Path) -> None:
    """
    Verify a typo'd exclusion column fails instead of being ignored.

    Silently skipping it would return the artifact cells from a config that reads
    as if it dropped them.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="not found in obs"):
        load_adata(path, exclude_column="cluster", exclude_values=["22"])


def test_load_adata_exclude_rejects_removing_every_cell(tmp_path: Path) -> None:
    """
    Verify an exclusion that empties the object fails and names the exclusion rule.

    The error has to say which rule emptied the slice; an inclusion-flavoured
    message on an exclusion-only load would send the reader to check the wrong
    config key.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match=r"input\.exclude leiden not in"):
        load_adata(path, exclude_column="leiden", exclude_values=["0", "1", "22"])


def test_load_adata_rejects_half_specified_exclusion(tmp_path: Path) -> None:
    """
    Verify an exclusion needs both column and values, or neither.

    Half of a filter reads like a filter and removes nothing.
    """

    path = tmp_path / "input.h5ad"
    make_clustered_adata().write_h5ad(path)

    with pytest.raises(AnnDataLoadError, match="both exclude_column and exclude_values"):
        load_adata(path, exclude_column="leiden")
