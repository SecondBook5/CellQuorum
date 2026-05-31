"""Tests for CellQuorum manifest loading and validation."""

from __future__ import annotations

# Import Path for temporary manifest and data-root paths.
from pathlib import Path

# Import pandas for DataFrame-based manifest validation tests.
import pandas as pd

# Import pytest for exception assertions.
import pytest

# Import manifest validation utilities.
from cellquorum.io.manifest import (
    Manifest,
    ManifestError,
    ManifestRecord,
    load_manifest,
    validate_manifest_dataframe,
)


def test_validate_manifest_dataframe_accepts_minimal_manifest() -> None:
    """
    Verify that a minimal valid manifest can be validated.

    A CellQuorum manifest should require only `sample_id` and `path` at the
    lowest level. Optional biological metadata should improve downstream method
    gating, but it should not be required for a basic single-sample or
    exploratory workflow.
    """

    # Build a minimal manifest DataFrame.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "path": ["sample_1.h5ad", "sample_2.h5ad"],
        }
    )

    # Validate the manifest DataFrame.
    manifest = validate_manifest_dataframe(dataframe)

    # Confirm the returned object is a Manifest.
    assert isinstance(manifest, Manifest)

    # Confirm both records were retained.
    assert len(manifest) == 2

    # Confirm sample IDs are available in manifest order.
    assert manifest.sample_ids == ["sample_1", "sample_2"]

    # Confirm paths are preserved as relative paths without data_root.
    assert manifest.paths == [Path("sample_1.h5ad"), Path("sample_2.h5ad")]


def test_validate_manifest_dataframe_accepts_standard_metadata() -> None:
    """
    Verify that standard optional metadata columns are parsed.

    Donor, condition, batch, tissue, timepoint, assay, and species metadata will
    drive downstream method gates for donor-aware differential expression,
    paired designs, integration, and biological reporting.
    """

    # Build a manifest DataFrame with standard metadata columns.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["sample_1.h5ad"],
            "donor_id": ["donor_1"],
            "condition": ["control"],
            "batch": ["batch_1"],
            "tissue": ["blood"],
            "timepoint": ["baseline"],
            "assay": ["10x_3prime"],
            "species": ["human"],
        }
    )

    # Validate the manifest DataFrame.
    manifest = validate_manifest_dataframe(dataframe)

    # Retrieve the validated record.
    record = manifest.get_record("sample_1")

    # Confirm donor ID was parsed.
    assert record.donor_id == "donor_1"

    # Confirm condition was parsed.
    assert record.condition == "control"

    # Confirm batch was parsed.
    assert record.batch == "batch_1"

    # Confirm tissue was parsed.
    assert record.tissue == "blood"

    # Confirm timepoint was parsed.
    assert record.timepoint == "baseline"

    # Confirm assay was parsed.
    assert record.assay == "10x_3prime"

    # Confirm species was parsed.
    assert record.species == "human"


def test_validate_manifest_dataframe_preserves_extra_metadata() -> None:
    """
    Verify that project-specific metadata columns are preserved.

    CellQuorum should not discard metadata simply because the core package does
    not know about it yet. Project-specific fields such as treatment arm,
    sequencing center, pathology label, or imaging cohort may become important
    for reports and future method gates.
    """

    # Build a manifest DataFrame with extra project metadata.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["sample_1.h5ad"],
            "cohort": ["discovery"],
            "treatment_arm": ["anti_pd1"],
        }
    )

    # Validate the manifest DataFrame.
    manifest = validate_manifest_dataframe(dataframe)

    # Retrieve the validated record.
    record = manifest.get_record("sample_1")

    # Confirm the extra cohort metadata was preserved.
    assert record.extra_metadata["cohort"] == "discovery"

    # Confirm the extra treatment metadata was preserved.
    assert record.extra_metadata["treatment_arm"] == "anti_pd1"

    # Convert the manifest back to a DataFrame.
    output = manifest.to_dataframe()

    # Confirm the extra cohort column appears in the output table.
    assert "cohort" in output.columns

    # Confirm the extra treatment arm column appears in the output table.
    assert "treatment_arm" in output.columns


def test_validate_manifest_dataframe_resolves_paths_against_data_root(tmp_path: Path) -> None:
    """
    Verify that relative sample paths can be resolved against data_root.

    Data roots let project manifests remain portable. A manifest can store paths
    relative to a dataset directory, while CellQuorum resolves them at runtime.
    """

    # Create a fake data root.
    data_root = tmp_path / "data"

    # Build a manifest DataFrame with relative sample paths.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["samples/sample_1.h5ad"],
        }
    )

    # Validate the manifest with a data root.
    manifest = validate_manifest_dataframe(dataframe, data_root=data_root)

    # Confirm the path was resolved under the data root.
    assert manifest.paths == [(data_root / "samples" / "sample_1.h5ad").resolve()]

    # Confirm the manifest stores the resolved data root.
    assert manifest.data_root == data_root.resolve()


def test_validate_manifest_dataframe_preserves_absolute_paths(tmp_path: Path) -> None:
    """
    Verify that absolute paths remain absolute when data_root is provided.

    Users may mix project-relative and absolute paths during development. Absolute
    paths should not be joined to data_root.
    """

    # Create an absolute sample path.
    sample_path = tmp_path / "sample_1.h5ad"

    # Build a manifest DataFrame with an absolute sample path.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": [str(sample_path)],
        }
    )

    # Validate the manifest with a data root.
    manifest = validate_manifest_dataframe(dataframe, data_root=tmp_path / "data")

    # Confirm the absolute path was preserved.
    assert manifest.paths == [sample_path]


def test_validate_manifest_dataframe_rejects_non_dataframe_input() -> None:
    """
    Verify that non-DataFrame manifest inputs fail clearly.

    This protects the public validation function from ambiguous inputs and keeps
    error messages useful for API users.
    """

    # Confirm non-DataFrame input raises a TypeError.
    with pytest.raises(TypeError, match="expected a pandas DataFrame"):
        validate_manifest_dataframe({"sample_id": ["sample_1"]})  # type: ignore[arg-type]


def test_validate_manifest_dataframe_rejects_empty_manifest() -> None:
    """
    Verify that an empty manifest is rejected.

    A manifest with no sample rows cannot support QC, preprocessing, reporting,
    or provenance.
    """

    # Build an empty manifest DataFrame with required columns.
    dataframe = pd.DataFrame(columns=["sample_id", "path"])

    # Confirm the empty manifest raises a ManifestError.
    with pytest.raises(ManifestError, match="at least one sample row"):
        validate_manifest_dataframe(dataframe)


def test_validate_manifest_dataframe_rejects_missing_required_columns() -> None:
    """
    Verify that required manifest columns are enforced.

    CellQuorum cannot identify samples or load sample files without `sample_id`
    and `path`.
    """

    # Build a DataFrame missing the path column.
    dataframe = pd.DataFrame({"sample_id": ["sample_1"]})

    # Confirm validation fails with a required-column message.
    with pytest.raises(ManifestError, match="missing required column"):
        validate_manifest_dataframe(dataframe)


def test_validate_manifest_dataframe_rejects_empty_required_values() -> None:
    """
    Verify that empty sample IDs and paths are rejected.

    Empty identifiers or paths would create ambiguous downstream artifacts and
    impossible-to-debug data loading failures.
    """

    # Build a DataFrame with an empty sample ID.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "   "],
            "path": ["sample_1.h5ad", "sample_2.h5ad"],
        }
    )

    # Confirm validation fails on the empty required sample_id value.
    with pytest.raises(ManifestError, match="empty required 'sample_id'"):
        validate_manifest_dataframe(dataframe)

    # Build a DataFrame with an empty path.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "path": ["sample_1.h5ad", ""],
        }
    )

    # Confirm validation fails on the empty required path value.
    with pytest.raises(ManifestError, match="empty required 'path'"):
        validate_manifest_dataframe(dataframe)


def test_validate_manifest_dataframe_rejects_duplicate_sample_ids() -> None:
    """
    Verify that sample IDs must be unique.

    Duplicate sample identifiers would overwrite outputs, confuse provenance,
    and break donor-aware or sample-aware downstream analyses.
    """

    # Build a DataFrame with duplicate sample IDs.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_1"],
            "path": ["sample_1_a.h5ad", "sample_1_b.h5ad"],
        }
    )

    # Confirm duplicate sample IDs raise a manifest error.
    with pytest.raises(ManifestError, match="duplicate sample_id"):
        validate_manifest_dataframe(dataframe)


def test_validate_manifest_dataframe_strips_column_names() -> None:
    """
    Verify that harmless whitespace around column names is stripped.

    Spreadsheet-edited manifests commonly accumulate whitespace in headers.
    CellQuorum should clean that without making users manually repair trivial
    formatting.
    """

    # Build a DataFrame with whitespace around required column names.
    dataframe = pd.DataFrame(
        {
            " sample_id ": ["sample_1"],
            " path ": ["sample_1.h5ad"],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Confirm the sample ID was parsed correctly.
    assert manifest.sample_ids == ["sample_1"]

    # Confirm the path was parsed correctly.
    assert manifest.paths == [Path("sample_1.h5ad")]


def test_validate_manifest_dataframe_rejects_duplicate_cleaned_columns() -> None:
    """
    Verify that duplicated columns after cleanup are rejected.

    If `sample_id` and ` sample_id ` both exist, the manifest is ambiguous after
    cleanup and should fail early.
    """

    # Build a DataFrame with duplicate column names after stripping.
    dataframe = pd.DataFrame(
        [["sample_1", "sample_2", "sample_1.h5ad"]],
        columns=["sample_id", " sample_id ", "path"],
    )

    # Confirm validation rejects duplicated cleaned columns.
    with pytest.raises(ManifestError, match="duplicate column names"):
        validate_manifest_dataframe(dataframe)


def test_manifest_metadata_availability_reports_non_null_standard_fields() -> None:
    """
    Verify that metadata availability checks require non-null values.

    Method gates should not treat a completely empty metadata column as usable.
    This matters for donor-aware models, condition-aware tests, and batch-aware
    integration.
    """

    # Build a manifest with one useful metadata column and one empty metadata column.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "path": ["sample_1.h5ad", "sample_2.h5ad"],
            "condition": ["control", "treated"],
            "batch": ["", ""],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Build the metadata availability summary.
    availability = manifest.metadata_availability()

    # Confirm condition is available.
    assert availability["condition"] is True

    # Confirm batch is not available because all values are empty.
    assert availability["batch"] is False

    # Confirm donor_id is not available because the column is absent.
    assert availability["donor_id"] is False


def test_manifest_has_column_and_has_non_null_column() -> None:
    """
    Verify direct metadata column checks.

    These helpers provide simple building blocks for planner and method-gate
    logic.
    """

    # Build a manifest with one populated and one empty metadata field.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["sample_1.h5ad"],
            "condition": ["control"],
            "batch": [""],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Confirm the condition column exists.
    assert manifest.has_column("condition") is True

    # Confirm the condition column has a usable value.
    assert manifest.has_non_null_column("condition") is True

    # Confirm the batch column exists.
    assert manifest.has_column("batch") is True

    # Confirm the batch column has no usable values.
    assert manifest.has_non_null_column("batch") is False

    # Confirm a missing column is absent.
    assert manifest.has_column("missing_column") is False

    # Confirm a missing column has no usable values.
    assert manifest.has_non_null_column("missing_column") is False


def test_manifest_get_record_returns_record_by_sample_id() -> None:
    """
    Verify that manifest records can be retrieved by sample ID.

    Downstream stages often need to fetch sample-level metadata by identifier.
    """

    # Build a manifest DataFrame.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "path": ["sample_1.h5ad", "sample_2.h5ad"],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Retrieve the second sample.
    record = manifest.get_record("sample_2")

    # Confirm the returned object is a ManifestRecord.
    assert isinstance(record, ManifestRecord)

    # Confirm the correct record was returned.
    assert record.sample_id == "sample_2"


def test_manifest_get_record_rejects_unknown_sample_id() -> None:
    """
    Verify that unknown sample IDs fail clearly.

    Silent missing-record behavior would create confusing downstream errors.
    """

    # Build a manifest DataFrame.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["sample_1.h5ad"],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Confirm unknown sample IDs raise a ManifestError.
    with pytest.raises(ManifestError, match="was not found"):
        manifest.get_record("missing_sample")


def test_manifest_to_dataframe_round_trips_standard_and_extra_metadata() -> None:
    """
    Verify that Manifest.to_dataframe preserves standard and extra fields.

    Reports and provenance will rely on converting manifests back into tables, so
    the output should include required fields, standard metadata, and extra
    project metadata.
    """

    # Build a manifest DataFrame with standard and extra metadata.
    dataframe = pd.DataFrame(
        {
            "sample_id": ["sample_1"],
            "path": ["sample_1.h5ad"],
            "condition": ["control"],
            "custom_score": [1.5],
        }
    )

    # Validate the manifest.
    manifest = validate_manifest_dataframe(dataframe)

    # Convert the manifest back to a DataFrame.
    output = manifest.to_dataframe()

    # Confirm required sample_id column exists.
    assert output.loc[0, "sample_id"] == "sample_1"

    # Confirm required path column exists.
    assert output.loc[0, "path"] == "sample_1.h5ad"

    # Confirm standard metadata was preserved.
    assert output.loc[0, "condition"] == "control"

    # Confirm extra metadata was preserved.
    assert output.loc[0, "custom_score"] == 1.5


def test_load_manifest_reads_csv_file(tmp_path: Path) -> None:
    """
    Verify that load_manifest reads CSV manifests.

    CSV is the default manifest format most users will create manually or export
    from spreadsheets.
    """

    # Create a CSV manifest path.
    manifest_path = tmp_path / "manifest.csv"

    # Write a valid CSV manifest.
    manifest_path.write_text(
        "sample_id,path,condition\nsample_1,sample_1.h5ad,control\n",
        encoding="utf-8",
    )

    # Load the manifest.
    manifest = load_manifest(manifest_path)

    # Confirm the sample was loaded.
    assert manifest.sample_ids == ["sample_1"]

    # Confirm the source path was stored as an absolute path.
    assert manifest.source_path == manifest_path.resolve()

    # Confirm standard metadata was loaded.
    assert manifest.get_record("sample_1").condition == "control"


def test_load_manifest_reads_tsv_file(tmp_path: Path) -> None:
    """
    Verify that load_manifest reads TSV manifests.

    TSV files are useful when metadata fields may contain commas.
    """

    # Create a TSV manifest path.
    manifest_path = tmp_path / "manifest.tsv"

    # Write a valid TSV manifest.
    manifest_path.write_text(
        "sample_id\tpath\tcondition\nsample_1\tsample_1.h5ad\tcontrol\n",
        encoding="utf-8",
    )

    # Load the manifest.
    manifest = load_manifest(manifest_path)

    # Confirm the sample was loaded.
    assert manifest.sample_ids == ["sample_1"]

    # Confirm the path was loaded.
    assert manifest.paths == [Path("sample_1.h5ad")]


def test_load_manifest_resolves_paths_with_data_root(tmp_path: Path) -> None:
    """
    Verify that load_manifest resolves relative paths with data_root.

    This is the file-based equivalent of DataFrame validation with data_root.
    """

    # Create a data root.
    data_root = tmp_path / "data"

    # Create a CSV manifest path.
    manifest_path = tmp_path / "manifest.csv"

    # Write a valid CSV manifest with a relative path.
    manifest_path.write_text(
        "sample_id,path\nsample_1,samples/sample_1.h5ad\n",
        encoding="utf-8",
    )

    # Load the manifest with data_root.
    manifest = load_manifest(manifest_path, data_root=data_root)

    # Confirm the sample path was resolved under data_root.
    assert manifest.paths == [(data_root / "samples" / "sample_1.h5ad").resolve()]


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    """
    Verify that load_manifest rejects missing manifest files.

    This gives CLI and API users an immediate, clear error when the manifest path
    is wrong.
    """

    # Create a missing manifest path.
    manifest_path = tmp_path / "missing.csv"

    # Confirm loading the missing file raises a ManifestError.
    with pytest.raises(ManifestError, match="does not exist"):
        load_manifest(manifest_path)


def test_load_manifest_rejects_directory_path(tmp_path: Path) -> None:
    """
    Verify that load_manifest rejects directory paths.

    Passing a directory instead of a manifest file should fail before pandas tries
    to read it.
    """

    # Confirm loading a directory raises a ManifestError.
    with pytest.raises(ManifestError, match="not a file"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """
    Verify that load_manifest rejects unsupported manifest suffixes.

    CellQuorum currently supports CSV and TSV manifests only.
    """

    # Create an unsupported manifest path.
    manifest_path = tmp_path / "manifest.xlsx"

    # Write placeholder content so the file exists.
    manifest_path.write_text("sample_id,path\nsample_1,sample_1.h5ad\n", encoding="utf-8")

    # Confirm loading the unsupported suffix raises a ManifestError.
    with pytest.raises(ManifestError, match="must end in '.csv' or '.tsv'"):
        load_manifest(manifest_path)


def test_manifest_record_to_dict_includes_extra_metadata() -> None:
    """
    Verify that ManifestRecord.to_dict includes extra metadata.

    This protects provenance and reporting behavior for custom project columns.
    """

    # Build a manifest record with extra metadata.
    record = ManifestRecord(
        sample_id="sample_1",
        path=Path("sample_1.h5ad"),
        condition="control",
        extra_metadata={"cohort": "training"},
    )

    # Convert the record to a dictionary.
    payload = record.to_dict()

    # Confirm the required sample ID appears.
    assert payload["sample_id"] == "sample_1"

    # Confirm the path was stringified.
    assert payload["path"] == "sample_1.h5ad"

    # Confirm standard metadata appears.
    assert payload["condition"] == "control"

    # Confirm extra metadata appears.
    assert payload["cohort"] == "training"
