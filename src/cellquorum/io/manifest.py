"""Manifest loading and validation for CellQuorum."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cellquorum.core.exceptions import CellQuorumManifestError


class ManifestError(CellQuorumManifestError):
    """
    Report manifest loading or validation failures.

    CellQuorum relies on a manifest to connect sample identifiers, input files,
    donor/sample metadata, batches, conditions, tissues, and timepoints. Manifest
    errors should fail early with clear messages because downstream QC,
    preprocessing, differential testing, method gates, and reports all depend on
    the correctness of this table.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize a manifest validation error.

        Args:
            message: Human-readable error message describing the manifest problem.
        """

        # Initialize the CellQuorumManifestError base class with the provided message.
        super().__init__(message)


@dataclass(frozen=True)
class ManifestRecord:
    """
    Store one validated manifest row.

    Each row represents one input sample or sample-level object. The required
    fields are intentionally minimal: `sample_id` identifies the sample and
    `path` identifies the input file. Common metadata fields are promoted to
    explicit attributes because they control downstream biological and statistical
    logic. Any additional columns are preserved in `extra_metadata` so project-
    specific metadata is not lost.

    Args:
        sample_id: Unique sample identifier.
        path: Input data path for the sample.
        donor_id: Optional donor or patient identifier.
        condition: Optional biological or experimental condition.
        batch: Optional technical batch identifier.
        tissue: Optional tissue or anatomical source.
        timepoint: Optional timepoint label.
        assay: Optional assay label.
        species: Optional species or organism label.
        extra_metadata: Additional manifest columns preserved for downstream use.
    """

    sample_id: str

    path: Path | None = None

    # Store the CellRanger outs-parent directory locator (relative or absolute,
    # resolved later against ambient_correction.cellranger_root — NOT here).
    cellranger_path: str | None = None

    loom_path: str | None = None

    donor_id: str | None = None

    condition: str | None = None

    batch: str | None = None

    tissue: str | None = None

    timepoint: str | None = None

    assay: str | None = None

    species: str | None = None

    extra_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the manifest record to a dictionary.

        This representation is useful for provenance, testing, report generation,
        and conversion back into a pandas DataFrame.

        Returns:
            Dictionary containing required fields, standard metadata fields, and
            extra metadata.
        """

        record: dict[str, object] = {
            "sample_id": self.sample_id,
            "path": str(self.path) if self.path is not None else None,
            "cellranger_path": self.cellranger_path,
            "loom_path": self.loom_path,
            "donor_id": self.donor_id,
            "condition": self.condition,
            "batch": self.batch,
            "tissue": self.tissue,
            "timepoint": self.timepoint,
            "assay": self.assay,
            "species": self.species,
        }

        # Add any extra metadata columns after standard fields.
        record.update(self.extra_metadata)

        return record


@dataclass(frozen=True)
class Manifest:
    """
    Store a validated CellQuorum sample manifest.

    The manifest is the first structured bridge between raw user data and the
    pipeline. It should be strict about required identifiers and paths, but
    flexible enough to preserve project-specific metadata. Downstream method
    gates can ask this object whether donor, condition, batch, tissue, timepoint,
    assay, or species metadata is available before attempting analyses that
    require those fields.

    Args:
        records: Validated manifest records.
        source_path: Optional path to the manifest file.
        data_root: Optional root used to resolve relative input paths.
    """

    records: list[ManifestRecord]

    source_path: Path | None = None

    data_root: Path | None = None

    REQUIRED_COLUMNS: tuple[str, ...] = ("sample_id",)

    OPTIONAL_COLUMNS: tuple[str, ...] = (
        "path",
        "cellranger_path",
        "loom_path",
        "donor_id",
        "condition",
        "batch",
        "tissue",
        "timepoint",
        "assay",
        "species",
    )

    @property
    def sample_ids(self) -> list[str]:
        """
        Return manifest sample identifiers in manifest order.

        Returns:
            Ordered list of sample IDs.
        """

        return [record.sample_id for record in self.records]

    @property
    def paths(self) -> list[Path]:
        """
        Return manifest input paths in manifest order.

        Returns:
            Ordered list of input paths. Records without a file path (for example a
            loom-only sample) are omitted.
        """

        return [record.path for record in self.records if record.path is not None]

    def __len__(self) -> int:
        """
        Return the number of records in the manifest.

        Returns:
            Number of manifest records.
        """

        return len(self.records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        """
        Iterate over manifest records.

        Returns:
            Iterator over ManifestRecord objects.
        """

        return iter(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the manifest to a pandas DataFrame.

        Returns:
            DataFrame containing one row per manifest record.
        """

        rows = [record.to_dict() for record in self.records]

        return pd.DataFrame(rows)

    def metadata_columns(self) -> list[str]:
        """
        Return metadata columns available in the manifest.

        The required `sample_id` and `path` fields are excluded. Standard optional
        fields and project-specific extra metadata fields are included when they
        are present in at least one record.

        Returns:
            Ordered list of available metadata columns.
        """

        # Convert the manifest to a DataFrame for column inspection.
        dataframe = self.to_dataframe()

        # Define columns that are not metadata fields.
        excluded_columns = set(self.REQUIRED_COLUMNS)

        return [column for column in dataframe.columns if column not in excluded_columns]

    def has_column(self, column: str) -> bool:
        """
        Return whether a metadata column is available.

        Args:
            column: Column name to check.

        Returns:
            True if the column exists in the manifest, otherwise False.
        """

        return column in self.to_dataframe().columns

    def has_non_null_column(self, column: str) -> bool:
        """
        Return whether a column exists and contains at least one non-null value.

        Method gates should usually use this instead of only checking whether a
        column exists. A column filled entirely with missing values is not useful
        for donor-aware models, differential testing, batch correction, or
        condition-aware analysis.

        Args:
            column: Column name to inspect.

        Returns:
            True if the column exists and contains at least one non-null value.
        """

        # Convert the manifest to a DataFrame for value inspection.
        dataframe = self.to_dataframe()

        # Return False when the column is absent.
        if column not in dataframe.columns:
            return False

        return bool(dataframe[column].apply(_is_present_value).any())

    def metadata_availability(self) -> dict[str, bool]:
        """
        Summarize availability of standard metadata fields.

        Returns:
            Dictionary mapping standard optional metadata fields to availability.
        """

        # Build availability for every standard optional metadata column.
        return {column: self.has_non_null_column(column) for column in self.OPTIONAL_COLUMNS}

    def get_record(self, sample_id: str) -> ManifestRecord:
        """
        Retrieve one manifest record by sample ID.

        Args:
            sample_id: Sample identifier to retrieve.

        Returns:
            Matching ManifestRecord.

        Raises:
            ManifestError: If no record exists for the requested sample ID.
        """

        # Search for the requested sample ID.
        for record in self.records:
            if record.sample_id == sample_id:
                return record

        raise ManifestError(f"Sample ID '{sample_id}' was not found in the manifest.")


def _is_missing_value(value: Any) -> bool:
    """
    Return whether a manifest value should be treated as missing.

    Args:
        value: Candidate manifest cell value.

    Returns:
        True when the value is null, empty, or whitespace-only.
    """

    # Treat pandas null values as missing.
    if pd.isna(value):
        return True

    # Treat empty or whitespace-only strings as missing.
    if isinstance(value, str) and not value.strip():
        return True

    # Treat all other values as present.
    return False


def _is_present_value(value: object) -> bool:
    """
    Return whether a manifest value should be treated as present.

    Args:
        value: Candidate manifest cell value.

    Returns:
        True when the value is not missing.
    """

    return not _is_missing_value(value)


def _clean_optional_string(value: object) -> str | None:
    """
    Clean an optional manifest value into a string or None.

    Args:
        value: Candidate manifest cell value.

    Returns:
        Cleaned string value, or None when the value is missing.
    """

    if _is_missing_value(value):
        return None

    # Convert the value to a stripped string.
    return str(value).strip()


def _clean_required_string(value: object, *, column: str, row_number: int) -> str:
    """
    Clean and validate a required manifest string.

    Args:
        value: Candidate manifest value.
        column: Column name being validated.
        row_number: One-based row number within the manifest table.

    Returns:
        Cleaned required string.

    Raises:
        ManifestError: If the value is missing.
    """

    # Raise a clear error when the required value is missing.
    if _is_missing_value(value):
        raise ManifestError(f"Manifest row {row_number} has an empty required '{column}' value.")

    return str(value).strip()


def _detect_separator(path: Path) -> str:
    """
    Detect the manifest file separator from its suffix.

    Args:
        path: Manifest path.

    Returns:
        Separator string for pandas.read_csv.

    Raises:
        ManifestError: If the suffix is unsupported.
    """

    # Normalize the suffix for comparison.
    suffix = path.suffix.lower()

    # Return comma separator for CSV files.
    if suffix == ".csv":
        return ","

    # Return tab separator for TSV files.
    if suffix == ".tsv":
        return "\t"

    # Raise a clear error for unsupported manifest formats.
    raise ManifestError("Manifest files must end in '.csv' or '.tsv'. " f"Received: {path.name}")


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize manifest column names.

    Args:
        dataframe: Raw manifest DataFrame.

    Returns:
        DataFrame with stripped column names.

    Raises:
        ManifestError: If column names are empty or duplicated after stripping.
    """

    cleaned_columns = [str(column).strip() for column in dataframe.columns]

    if any(not column for column in cleaned_columns):
        raise ManifestError("Manifest contains one or more empty column names.")

    # Reject duplicated column names after cleanup.
    duplicated_columns = sorted(
        {column for column in cleaned_columns if cleaned_columns.count(column) > 1}
    )

    # Raise a clear error when duplicate columns exist.
    if duplicated_columns:
        raise ManifestError(
            "Manifest contains duplicate column names after cleanup: "
            f"{', '.join(duplicated_columns)}."
        )

    # Copy the DataFrame to avoid mutating caller-owned data.
    normalized = dataframe.copy()

    # Assign cleaned column names.
    normalized.columns = cleaned_columns

    return normalized


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that required manifest columns are present.

    Args:
        dataframe: Manifest DataFrame.

    Raises:
        ManifestError: If required columns are missing.
    """

    # Find required columns that are absent from the DataFrame.
    missing_columns = [
        column for column in Manifest.REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    # Raise a clear error when required columns are missing.
    if missing_columns:
        raise ManifestError(
            "Manifest is missing required column(s): " f"{', '.join(missing_columns)}."
        )


def _resolve_manifest_path(path_value: str, data_root: Path | None) -> Path:
    """
    Resolve a manifest input path.

    Relative paths are resolved against `data_root` when provided. When no
    `data_root` is provided, relative paths are preserved as relative paths
    instead of being resolved against the current working directory. This avoids
    silently making project manifests machine-specific.

    Args:
        path_value: Cleaned path value from the manifest.
        data_root: Optional root directory for relative input paths.

    Returns:
        Resolved or preserved Path.
    """

    # Convert the path string into a Path object.
    path = Path(path_value).expanduser()

    # Return absolute paths as-is after user expansion.
    if path.is_absolute():
        return path

    # Resolve relative paths against data_root when provided.
    if data_root is not None:
        return (data_root / path).expanduser().resolve()

    # Preserve relative paths when no data_root is provided.
    return path


def validate_manifest_dataframe(
    dataframe: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> Manifest:
    """
    Validate a manifest DataFrame into a Manifest object.

    This function is useful for tests, notebooks, and future API paths where a
    manifest table has already been loaded. It enforces required columns, unique
    sample IDs, non-empty input paths, and preservation of project-specific
    metadata.

    Args:
        dataframe: Manifest table to validate.
        source_path: Optional source path used for provenance.
        data_root: Optional root directory used to resolve relative input paths.

    Returns:
        Validated Manifest object.

    Raises:
        TypeError: If dataframe is not a pandas DataFrame.
        ManifestError: If the manifest is empty or invalid.
    """

    # Validate that the caller supplied a pandas DataFrame.
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "validate_manifest_dataframe expected a pandas DataFrame. "
            f"Received: {type(dataframe).__name__}."
        )

    if dataframe.empty:
        raise ManifestError("Manifest must contain at least one sample row.")

    # Normalize column names before validation.
    normalized = _normalize_columns(dataframe)

    _validate_required_columns(normalized)

    # Normalize the optional source path.
    resolved_source_path = Path(source_path).expanduser().resolve() if source_path else None

    # Normalize the optional data root.
    resolved_data_root = Path(data_root).expanduser().resolve() if data_root else None

    records: list[ManifestRecord] = []

    seen_sample_ids: set[str] = set()

    # Iterate over manifest rows with one-based row numbers.
    for row_offset, row in enumerate(normalized.to_dict(orient="records"), start=1):
        # Clean and validate the required sample ID.
        sample_id = _clean_required_string(
            row.get("sample_id"),
            column="sample_id",
            row_number=row_offset,
        )

        if sample_id in seen_sample_ids:
            raise ManifestError(f"Manifest contains duplicate sample_id: '{sample_id}'.")

        seen_sample_ids.add(sample_id)

        # Clean the optional path and cellranger_path locators.
        raw_path = _clean_optional_string(row.get("path"))
        cellranger_path = _clean_optional_string(row.get("cellranger_path"))

        # Require at least one input locator per row (fail loud, name the row).
        if raw_path is None and cellranger_path is None:
            raise ManifestError(
                f"Manifest row {row_offset} has neither a 'path' nor a "
                "'cellranger_path' — every sample needs an input locator."
            )

        # Resolve the file path only when present; cellranger_path stays verbatim.
        resolved_path = (
            _resolve_manifest_path(raw_path, resolved_data_root) if raw_path is not None else None
        )

        extra_metadata: dict[str, object] = {
            str(column): value
            for column, value in row.items()
            if column not in Manifest.REQUIRED_COLUMNS and column not in Manifest.OPTIONAL_COLUMNS
        }

        record = ManifestRecord(
            sample_id=sample_id,
            path=resolved_path,
            cellranger_path=cellranger_path,
            loom_path=_clean_optional_string(row.get("loom_path")),
            donor_id=_clean_optional_string(row.get("donor_id")),
            condition=_clean_optional_string(row.get("condition")),
            batch=_clean_optional_string(row.get("batch")),
            tissue=_clean_optional_string(row.get("tissue")),
            timepoint=_clean_optional_string(row.get("timepoint")),
            assay=_clean_optional_string(row.get("assay")),
            species=_clean_optional_string(row.get("species")),
            extra_metadata=extra_metadata,
        )

        records.append(record)

    return Manifest(
        records=records,
        source_path=resolved_source_path,
        data_root=resolved_data_root,
    )


def load_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> Manifest:
    """
    Load and validate a CellQuorum manifest from CSV or TSV.

    Args:
        manifest_path: Path to a CSV or TSV manifest file.
        data_root: Optional root directory used to resolve relative sample paths.

    Returns:
        Validated Manifest object.

    Raises:
        ManifestError: If the manifest path does not exist, is not a file, cannot
            be read, or fails validation.
    """

    # Normalize the manifest path.
    path = Path(manifest_path).expanduser()

    # Reject missing manifest files.
    if not path.exists():
        raise ManifestError(f"Manifest file does not exist: {path}")

    # Reject directories.
    if not path.is_file():
        raise ManifestError(f"Manifest path is not a file: {path}")

    # Detect the separator from the file suffix.
    separator = _detect_separator(path)

    # Try reading the manifest file.
    try:
        # Read all columns as object/string-compatible values without index inference.
        dataframe = pd.read_csv(path, sep=separator, dtype=object)

    # Convert pandas read failures into ManifestError.
    except Exception as error:
        raise ManifestError(f"Failed to read manifest file '{path}': {error}") from error

    return validate_manifest_dataframe(
        dataframe,
        source_path=path,
        data_root=data_root,
    )
