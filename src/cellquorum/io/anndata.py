"""AnnData input/output utilities for CellQuorum."""

from __future__ import annotations

# Import PathLike for filesystem-like input path typing.
from os import PathLike

# Import Path for robust filesystem validation.
from pathlib import Path

# Import AnnData for return type validation.
import anndata as ad

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError


class AnnDataLoadError(CellQuorumDataError):
    """
    Report AnnData loading failures.

    This error is raised when a user-supplied AnnData path is missing, malformed,
    points to an unsupported file type, or cannot be read into an AnnData object.
    """


def normalize_adata_path(path: str | PathLike[str] | Path) -> Path:
    """
    Normalize a user-supplied AnnData path.

    Args:
        path: Candidate AnnData path.

    Returns:
        Expanded Path object.

    Raises:
        AnnDataLoadError: If the path is empty or has an invalid type.
    """

    # Reject empty string paths before Path("") turns into the current directory.
    if isinstance(path, str) and not path.strip():
        raise AnnDataLoadError("AnnData path cannot be empty.")

    # Convert the candidate path into a Path object.
    try:
        normalized_path = Path(path).expanduser()

    # Convert invalid path-like objects into AnnData loading errors.
    except TypeError as error:
        raise AnnDataLoadError(
            "AnnData path must be a string or filesystem path-like object. "
            f"Received: {type(path).__name__}."
        ) from error

    # Return the normalized path.
    return normalized_path


def validate_adata_path(path: str | PathLike[str] | Path) -> Path:
    """
    Validate an AnnData input path.

    Args:
        path: Candidate AnnData path.

    Returns:
        Validated Path object.

    Raises:
        AnnDataLoadError: If the path is missing, not a file, or unsupported.
    """

    # Normalize the input path.
    normalized_path = normalize_adata_path(path)

    # Reject missing paths.
    if not normalized_path.exists():
        raise AnnDataLoadError(f"AnnData file does not exist: {normalized_path}")

    # Reject directories.
    if not normalized_path.is_file():
        raise AnnDataLoadError(f"AnnData path is not a file: {normalized_path}")

    # Reject unsupported file suffixes.
    if normalized_path.suffix.lower() != ".h5ad":
        raise AnnDataLoadError(
            "CellQuorum currently supports AnnData input only as '.h5ad'. "
            f"Received: {normalized_path.name}"
        )

    # Return the validated path.
    return normalized_path


def _load_adata_subset(
    validated_path: Path, subset_column: str, subset_values: list[str]
) -> ad.AnnData:
    """
    Load only the rows of an h5ad whose ``subset_column`` is in ``subset_values``.

    The object is opened in backed mode so its full X matrix never enters memory;
    obs (small) is read to build the row mask, and only the matching slice is
    materialized. This lets a hypothesis restrict a large shared global object to
    its cell type without a separate pre-sliced file and without the peak memory
    of loading every cell. The applied restriction is recorded on the returned
    object at ``uns['cellquorum_input_subset']`` so the run can log it.

    Args:
        validated_path: Validated h5ad path.
        subset_column: obs column to filter on.
        subset_values: accepted values; a row is kept when its value is in here.

    Returns:
        In-memory AnnData holding only the matching rows.

    Raises:
        AnnDataLoadError: If the column is absent or no rows match.
    """

    # Open backed so the full expression matrix is never read into memory.
    backed = ad.read_h5ad(validated_path, backed="r")

    try:
        # Fail loudly when the requested column is not present.
        if subset_column not in backed.obs.columns:
            available = list(backed.obs.columns)
            raise AnnDataLoadError(
                f"input.subset.column {subset_column!r} not found in obs. "
                f"Available columns include: {available[:25]}"
            )

        # Compare as strings so categorical/object/nullable dtypes all match.
        wanted = {str(value) for value in subset_values}
        column_as_str = backed.obs[subset_column].astype("string").astype(object)
        mask = column_as_str.isin(wanted).to_numpy()

        n_before = int(backed.n_obs)
        n_after = int(mask.sum())

        # An empty slice is almost always a mislabeled value; fail rather than
        # silently run a pipeline on zero cells.
        if n_after == 0:
            raise AnnDataLoadError(
                f"input.subset {subset_column} in {sorted(wanted)} matched 0 of "
                f"{n_before} cells. Check the value spelling against the obs column."
            )

        # Materialize only the matching rows from the still-open backed file.
        subset_adata = backed[mask].to_memory()

    finally:
        # Release the backed file handle regardless of success.
        if backed.isbacked and backed.file is not None:
            backed.file.close()

    # Record the applied restriction for run provenance.
    subset_adata.uns["cellquorum_input_subset"] = {
        "column": subset_column,
        "values": sorted(wanted),
        "n_before": n_before,
        "n_after": n_after,
    }

    return subset_adata


def load_adata(
    path: str | PathLike[str] | Path,
    *,
    subset_column: str | None = None,
    subset_values: list[str] | None = None,
) -> ad.AnnData:
    """
    Load an AnnData object from an h5ad file, optionally restricted to a slice.

    Args:
        path: Path to an h5ad file.
        subset_column: Optional obs column to restrict rows on.
        subset_values: Values to keep for ``subset_column``; required when it is
            given. When both are ``None`` the full object is read (default).

    Returns:
        Loaded AnnData object.

    Raises:
        AnnDataLoadError: If the file path is invalid, reading fails, or a
            partially specified subset is given.
    """

    # Validate the h5ad path before reading.
    validated_path = validate_adata_path(path)

    # A subset needs both a column and its values; reject a half-specified one.
    if (subset_column is None) != (subset_values is None):
        raise AnnDataLoadError(
            "load_adata subset requires both subset_column and subset_values, " "or neither."
        )

    # Try to read the AnnData object (full, or backed-mode slice).
    try:
        if subset_column is None:
            loaded_adata = ad.read_h5ad(validated_path)
        else:
            loaded_adata = _load_adata_subset(validated_path, subset_column, subset_values or [])

    # Preserve CellQuorum-specific errors (e.g. bad subset) without re-wrapping.
    except AnnDataLoadError:
        raise

    # Convert AnnData/HDF5 read failures into CellQuorum-specific errors.
    except Exception as error:
        raise AnnDataLoadError(f"Failed to read AnnData file '{validated_path}'.") from error

    # Validate the loaded object defensively.
    if not isinstance(loaded_adata, ad.AnnData):
        raise AnnDataLoadError(
            "Expected an AnnData object from read_h5ad, but received "
            f"{type(loaded_adata).__name__}."
        )

    # Return the loaded AnnData object.
    return loaded_adata


__all__ = [
    "AnnDataLoadError",
    "load_adata",
    "normalize_adata_path",
    "validate_adata_path",
]
