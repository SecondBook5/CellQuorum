"""AnnData validation utilities for CellQuorum QC."""

from __future__ import annotations

# Import Sequence for validating requested metadata columns.
from collections.abc import Sequence

# Import dataclass helpers for structured validation summaries.
from dataclasses import dataclass, field

# Import AnnData for runtime input validation.
import anndata as ad

# Import NumPy for matrix dtype and finite-value checks.
import numpy as np

# Import sparse matrix helpers for sparse AnnData.X validation.
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumDataError

# Import shared CellQuorum data exception base.
from cellquorum.stages.qc._types import ExpressionMatrix

# Import QC configuration models.
from cellquorum.stages.qc.config import QCConfig, QCDuplicateNameConfig


class QCInputValidationError(CellQuorumDataError):
    """
    Report invalid AnnData inputs for CellQuorum QC.

    QC should fail before metric calculation when the input object is missing,
    empty, non-numeric, negative, malformed, or inconsistent with the requested
    QC configuration. Failing here prevents downstream thresholding and artifact
    code from producing misleading results.
    """


@dataclass(frozen=True)
class QCInputValidationSummary:
    """
    Store a structured summary of QC input validation.

    This summary is intended for tests, provenance, and stage notes. It records
    the AnnData dimensions, the matrix source selected for QC, duplicate-name
    status, requested groupby columns, and non-fatal warnings.

    Args:
        n_obs: Number of observations in the AnnData object.
        n_vars: Number of variables in the AnnData object.
        matrix_n_obs: Number of observations in the selected QC matrix.
        matrix_n_vars: Number of variables in the selected QC matrix.
        matrix_source: Source of the QC matrix, such as X, raw.X, or layers[counts].
        matrix_type: Runtime class name of the selected QC matrix.
        has_raw: Whether AnnData.raw is present.
        obs_names_unique: Whether observation names are unique.
        var_names_unique: Whether variable names are unique.
        requested_groupby: MAD groupby columns requested by config.
        warnings: Non-fatal validation warnings.
    """

    # Store the number of AnnData observations.
    n_obs: int

    # Store the number of AnnData variables.
    n_vars: int

    # Store the selected matrix observation count.
    matrix_n_obs: int

    # Store the selected matrix variable count.
    matrix_n_vars: int

    # Store the selected matrix source label.
    matrix_source: str

    # Store the selected matrix runtime type.
    matrix_type: str

    # Store whether AnnData.raw is present.
    has_raw: bool

    # Store whether observation names are unique.
    obs_names_unique: bool

    # Store whether variable names are unique.
    var_names_unique: bool

    # Store requested MAD groupby columns.
    requested_groupby: tuple[str, ...] = field(default_factory=tuple)

    # Store non-fatal validation warnings.
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the validation summary into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the validation summary.
        """

        # Return a JSON-friendly dictionary.
        return {
            "n_obs": self.n_obs,
            "n_vars": self.n_vars,
            "matrix_n_obs": self.matrix_n_obs,
            "matrix_n_vars": self.matrix_n_vars,
            "matrix_source": self.matrix_source,
            "matrix_type": self.matrix_type,
            "has_raw": self.has_raw,
            "obs_names_unique": self.obs_names_unique,
            "var_names_unique": self.var_names_unique,
            "requested_groupby": list(self.requested_groupby),
            "warnings": list(self.warnings),
        }


def validate_qc_input_adata(
    adata: object,
    config: QCConfig | None = None,
) -> QCInputValidationSummary:
    """
    Validate an AnnData object before CellQuorum QC.

    The QC engine assumes a non-empty AnnData object with a numeric, finite,
    non-negative count-like matrix. This function validates those assumptions,
    checks config-selected matrix sources such as layers or raw.X, checks
    duplicate-name policy, and verifies that requested MAD groupby columns exist.

    Args:
        adata: Candidate AnnData object.
        config: Optional QC configuration. Defaults to QCConfig().

    Returns:
        Structured QC input validation summary.

    Raises:
        QCInputValidationError: If the input is not valid for QC.
    """

    # Resolve the QC configuration.
    qc_config = QCConfig() if config is None else config

    # Validate the QC configuration type.
    if not isinstance(qc_config, QCConfig):
        raise QCInputValidationError(
            "validate_qc_input_adata expected config to be a QCConfig object. "
            f"Received: {type(qc_config).__name__}."
        )

    # Validate that the input is an AnnData object.
    if not isinstance(adata, ad.AnnData):
        raise QCInputValidationError(
            "QC input must be an AnnData object. " f"Received: {type(adata).__name__}."
        )

    # Validate that the AnnData object contains observations.
    if adata.n_obs <= 0:
        raise QCInputValidationError("QC input AnnData must contain at least one observation.")

    # Validate that the AnnData object contains variables.
    if adata.n_vars <= 0:
        raise QCInputValidationError("QC input AnnData must contain at least one variable.")

    # Resolve the matrix requested by QC configuration.
    matrix, matrix_source = get_qc_matrix(adata, qc_config)

    # Validate the selected matrix and collect its shape.
    matrix_n_obs, matrix_n_vars = validate_qc_matrix(
        matrix,
        expected_n_obs=adata.n_obs,
        matrix_source=matrix_source,
    )

    # Validate MAD groupby columns requested by config.
    validate_mad_groupby_columns(adata, qc_config)

    # Initialize non-fatal validation warnings.
    warnings: list[str] = []

    # Extend warnings from observation-name duplicate policy.
    warnings.extend(
        validate_duplicate_name_policy(
            names_are_unique=adata.obs_names.is_unique,
            policy=qc_config.duplicate_names,
            axis_name="obs_names",
        )
    )

    # Extend warnings from variable-name duplicate policy.
    warnings.extend(
        validate_duplicate_name_policy(
            names_are_unique=adata.var_names.is_unique,
            policy=qc_config.duplicate_names,
            axis_name="var_names",
        )
    )

    # Build and return the validation summary.
    return QCInputValidationSummary(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        matrix_n_obs=matrix_n_obs,
        matrix_n_vars=matrix_n_vars,
        matrix_source=matrix_source,
        matrix_type=type(matrix).__name__,
        has_raw=adata.raw is not None,
        obs_names_unique=adata.obs_names.is_unique,
        var_names_unique=adata.var_names.is_unique,
        requested_groupby=tuple(qc_config.mad.groupby),
        warnings=tuple(warnings),
    )


def get_qc_matrix(adata: ad.AnnData, config: QCConfig) -> tuple[object, str]:
    """
    Resolve the matrix source requested for QC metric calculation.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Returns:
        Tuple containing the selected matrix and a source label.

    Raises:
        QCInputValidationError: If the requested matrix source is unavailable or ambiguous.
    """

    # Reject simultaneous raw.X and layer selection because the source is ambiguous.
    if config.metrics.use_raw and config.metrics.layer is not None:
        raise QCInputValidationError(
            "QC metric configuration cannot request both use_raw=true and a layer. "
            "Choose either AnnData.raw.X or one AnnData layer."
        )

    # Use AnnData.raw.X when requested.
    if config.metrics.use_raw:
        # Validate that AnnData.raw exists.
        if adata.raw is None:
            raise QCInputValidationError(
                "QC metric configuration requested use_raw=true, but AnnData.raw is missing."
            )

        # Return raw.X and its source label.
        return adata.raw.X, "raw.X"

    # Use a named AnnData layer when requested.
    if config.metrics.layer is not None:
        # Validate that the requested layer exists.
        if config.metrics.layer not in adata.layers:
            raise QCInputValidationError(
                f"QC metric layer '{config.metrics.layer}' was requested but is missing "
                "from AnnData.layers."
            )

        # Return the requested layer and its source label.
        return adata.layers[config.metrics.layer], f"layers[{config.metrics.layer}]"

    # Return AnnData.X by default.
    return adata.X, "X"


def validate_qc_matrix(
    matrix: ExpressionMatrix,
    *,
    expected_n_obs: int,
    matrix_source: str,
) -> tuple[int, int]:
    """
    Validate a selected QC matrix.

    QC metrics should be calculated from a two-dimensional numeric, finite,
    non-negative matrix. This function supports dense NumPy-like arrays and
    scipy sparse matrices.

    Args:
        matrix: Candidate matrix object.
        expected_n_obs: Expected observation count from AnnData.
        matrix_source: Human-readable matrix source label.

    Returns:
        Tuple of matrix observation count and matrix variable count.

    Raises:
        QCInputValidationError: If the matrix is malformed or unsuitable for QC.
    """

    # Validate that the matrix exposes a shape.
    if not hasattr(matrix, "shape"):
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' does not expose a shape attribute."
        )

    # Read the matrix shape.
    shape = matrix.shape

    # Validate that the matrix is two-dimensional.
    if len(shape) != 2:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must be two-dimensional. "
            f"Received shape: {shape}."
        )

    # Extract the matrix dimensions.
    matrix_n_obs = int(shape[0])
    matrix_n_vars = int(shape[1])

    # Validate that the observation dimension matches AnnData.obs.
    if matrix_n_obs != expected_n_obs:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' has {matrix_n_obs} observations, "
            f"but AnnData has {expected_n_obs} observations."
        )

    # Validate that the matrix contains at least one variable.
    if matrix_n_vars <= 0:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must contain at least one variable."
        )

    # Validate sparse matrices through their stored data.
    if sp.issparse(matrix):
        # Validate the sparse matrix dtype.
        _validate_numeric_dtype(matrix.dtype, matrix_source=matrix_source)

        # Validate finite sparse values.
        _validate_sparse_finite_values(matrix, matrix_source=matrix_source)

        # Validate non-negative sparse values.
        _validate_sparse_non_negative_values(matrix, matrix_source=matrix_source)

        # Return the validated sparse matrix shape.
        return matrix_n_obs, matrix_n_vars

    # Convert dense array-like matrices into NumPy arrays for validation.
    dense_matrix = np.asarray(matrix)

    # Validate the dense matrix dtype.
    _validate_numeric_dtype(dense_matrix.dtype, matrix_source=matrix_source)

    # Validate finite dense values.
    _validate_dense_finite_values(dense_matrix, matrix_source=matrix_source)

    # Validate non-negative dense values.
    _validate_dense_non_negative_values(dense_matrix, matrix_source=matrix_source)

    # Return the validated dense matrix shape.
    return matrix_n_obs, matrix_n_vars


def validate_mad_groupby_columns(adata: ad.AnnData, config: QCConfig) -> None:
    """
    Validate that requested MAD groupby columns exist in AnnData.obs.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Raises:
        QCInputValidationError: If one or more requested groupby columns are missing.
    """

    # Return early when MAD thresholding is disabled.
    if not config.mad.enabled:
        return

    # Return early when no groupby columns were requested.
    if not config.mad.groupby:
        return

    # Identify missing observation columns.
    missing_columns = [column for column in config.mad.groupby if column not in adata.obs.columns]

    # Raise a clear error when requested columns are absent.
    if missing_columns:
        raise QCInputValidationError(
            "MAD group-wise QC requested missing AnnData.obs column(s): "
            f"{', '.join(missing_columns)}."
        )


def require_obs_columns(adata: ad.AnnData, columns: Sequence[str]) -> None:
    """
    Require one or more AnnData.obs columns.

    Args:
        adata: AnnData object.
        columns: Observation columns that must exist.

    Raises:
        QCInputValidationError: If any requested column is missing.
    """

    # Validate that columns was not provided as a single string.
    if isinstance(columns, str):
        raise QCInputValidationError("columns must be a sequence of strings, not a string.")

    # Identify missing observation columns.
    missing_columns = [column for column in columns if column not in adata.obs.columns]

    # Raise a clear error when columns are missing.
    if missing_columns:
        raise QCInputValidationError(
            "AnnData.obs is missing required column(s): " f"{', '.join(missing_columns)}."
        )


def summarize_adata_shape(adata: ad.AnnData) -> dict[str, int]:
    """
    Summarize AnnData shape for QC provenance.

    Args:
        adata: AnnData object.

    Returns:
        Dictionary containing n_obs and n_vars.

    Raises:
        QCInputValidationError: If the input is not AnnData.
    """

    # Validate the AnnData input type.
    if not isinstance(adata, ad.AnnData):
        raise QCInputValidationError(
            "summarize_adata_shape expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

    # Return the shape summary.
    return {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
    }


def validate_duplicate_name_policy(
    *,
    names_are_unique: bool,
    policy: QCDuplicateNameConfig,
    axis_name: str,
) -> list[str]:
    """
    Validate duplicate-name status against configured duplicate-name policy.

    Args:
        names_are_unique: Whether the relevant AnnData index is unique.
        policy: Duplicate-name policy configuration.
        axis_name: Either obs_names or var_names.

    Returns:
        Non-fatal warning messages.

    Raises:
        QCInputValidationError: If duplicate names exist and policy is error.
    """

    # Return no warnings when names are already unique.
    if names_are_unique:
        return []

    # Resolve the axis-specific policy value.
    if axis_name == "obs_names":
        axis_policy = policy.obs_names

    # Resolve the variable-name policy.
    elif axis_name == "var_names":
        axis_policy = policy.var_names

    # Reject unsupported axis names defensively.
    else:
        raise QCInputValidationError(
            "Duplicate-name validation axis must be 'obs_names' or 'var_names'. "
            f"Received: {axis_name}."
        )

    # Raise an error when duplicate names are forbidden.
    if axis_policy == "error":
        raise QCInputValidationError(
            f"AnnData.{axis_name} contains duplicate values and policy is 'error'."
        )

    # Return no warnings when duplicates are intentionally ignored.
    if axis_policy == "ignore":
        return []

    # Return a warning when duplicate names should be made unique later.
    if axis_policy == "make_unique":
        return [
            f"AnnData.{axis_name} contains duplicate values and policy is "
            "'make_unique'. The QC stage should make names unique before metric calculation."
        ]

    # Return a warning when duplicate names should only be reported.
    if axis_policy == "warn":
        return [f"AnnData.{axis_name} contains duplicate values."]

    # Raise defensively if an impossible policy value appears.
    raise QCInputValidationError(
        f"Unsupported duplicate-name policy '{axis_policy}' for AnnData.{axis_name}."
    )


def _validate_numeric_dtype(dtype: np.dtype[object], *, matrix_source: str) -> None:
    """
    Validate that a matrix dtype is numeric.

    Args:
        dtype: Candidate NumPy dtype.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If dtype is not numeric.
    """

    # Reject non-numeric dtypes.
    if not np.issubdtype(dtype, np.number):
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must be numeric. " f"Received dtype: {dtype}."
        )


def _validate_sparse_finite_values(spmatrix: sp.spmatrix, *, matrix_source: str) -> None:
    """
    Validate that sparse matrix stored values are finite.

    Args:
        spmatrix: Sparse matrix to inspect.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If stored values contain NaN or infinity.
    """

    # Return early when the sparse matrix has no stored values.
    if spmatrix.data.size == 0:
        return

    # Reject NaN or infinite stored values.
    if not np.isfinite(spmatrix.data).all():
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' contains NaN or infinite values."
        )


def _validate_sparse_non_negative_values(
    spmatrix: sp.spmatrix,
    *,
    matrix_source: str,
) -> None:
    """
    Validate that sparse matrix stored values are non-negative.

    Args:
        spmatrix: Sparse matrix to inspect.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If stored values contain negative values.
    """

    # Return early when the sparse matrix has no stored values.
    if spmatrix.data.size == 0:
        return

    # Reject negative stored values.
    if np.min(spmatrix.data) < 0:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' contains negative values."
        )


def _validate_dense_finite_values(
    matrix: np.ndarray[object, object],
    *,
    matrix_source: str,
) -> None:
    """
    Validate that dense matrix values are finite.

    Args:
        matrix: Dense NumPy matrix.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If values contain NaN or infinity.
    """

    # Return early when the dense matrix is empty.
    if matrix.size == 0:
        return

    # Reject NaN or infinite dense values.
    if not np.isfinite(matrix).all():
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' contains NaN or infinite values."
        )


def _validate_dense_non_negative_values(
    matrix: np.ndarray[object, object],
    *,
    matrix_source: str,
) -> None:
    """
    Validate that dense matrix values are non-negative.

    Args:
        matrix: Dense NumPy matrix.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If values contain negative values.
    """

    # Return early when the dense matrix is empty.
    if matrix.size == 0:
        return

    # Reject negative dense values.
    if np.min(matrix) < 0:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' contains negative values."
        )


__all__ = [
    "QCInputValidationError",
    "QCInputValidationSummary",
    "get_qc_matrix",
    "require_obs_columns",
    "summarize_adata_shape",
    "validate_duplicate_name_policy",
    "validate_mad_groupby_columns",
    "validate_qc_input_adata",
    "validate_qc_matrix",
]
