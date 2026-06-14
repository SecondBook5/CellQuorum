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


def load_adata(path: str | PathLike[str] | Path) -> ad.AnnData:
    """
    Load an AnnData object from an h5ad file.

    Args:
        path: Path to an h5ad file.

    Returns:
        Loaded AnnData object.

    Raises:
        AnnDataLoadError: If the file path is invalid or reading fails.
    """

    # Validate the h5ad path before reading.
    validated_path = validate_adata_path(path)

    # Try to read the AnnData object.
    try:
        loaded_adata = ad.read_h5ad(validated_path)

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
