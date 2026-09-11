"""AnnData validation utilities for CellQuorum QC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from cellquorum.core.exceptions import CellQuorumDataError
from cellquorum.stages.qc._types import ExpressionMatrix
from cellquorum.stages.qc.config import QCConfig, QCDuplicateNameConfig


class QCInputValidationError(CellQuorumDataError):
    """Report invalid AnnData inputs for CellQuorum QC."""


@dataclass(frozen=True)
class QCInputValidationSummary:
    """Store a structured summary of QC input validation.

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
        requested_groupby: Mixture-model groupby columns requested by config.
        warnings: Non-fatal validation warnings.
    """

    n_obs: int
    n_vars: int
    matrix_n_obs: int
    matrix_n_vars: int
    matrix_source: str
    matrix_type: str
    has_raw: bool
    obs_names_unique: bool
    var_names_unique: bool
    requested_groupby: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Convert the validation summary into a JSON-friendly dictionary.

        Returns:
            Dictionary representation of the validation summary.
        """

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
    """Validate an AnnData object before CellQuorum QC.

    Args:
        adata: Candidate AnnData object.
        config: Optional QC configuration. Defaults to QCConfig().

    Returns:
        Structured QC input validation summary.

    Raises:
        QCInputValidationError: If the input is not valid for QC.
    """

    qc_config = QCConfig() if config is None else config

    if not isinstance(qc_config, QCConfig):
        raise QCInputValidationError(
            "validate_qc_input_adata expected config to be a QCConfig object. "
            f"Received: {type(qc_config).__name__}."
        )

    if not isinstance(adata, ad.AnnData):
        raise QCInputValidationError(
            "QC input must be an AnnData object. " f"Received: {type(adata).__name__}."
        )

    if adata.n_obs <= 0:
        raise QCInputValidationError("QC input AnnData must contain at least one observation.")

    if adata.n_vars <= 0:
        raise QCInputValidationError("QC input AnnData must contain at least one variable.")

    matrix, matrix_source = get_qc_matrix(adata, qc_config)

    matrix_n_obs, matrix_n_vars = validate_qc_matrix(
        matrix,
        expected_n_obs=adata.n_obs,
        matrix_source=matrix_source,
    )

    validate_mixture_groupby_columns(adata, qc_config)

    warnings: list[str] = []

    warnings.extend(
        validate_duplicate_name_policy(
            names_are_unique=adata.obs_names.is_unique,
            policy=qc_config.duplicate_names,
            axis_name="obs_names",
        )
    )

    warnings.extend(
        validate_duplicate_name_policy(
            names_are_unique=adata.var_names.is_unique,
            policy=qc_config.duplicate_names,
            axis_name="var_names",
        )
    )

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
        requested_groupby=tuple(qc_config.mito_mixture.groupby),
        warnings=tuple(warnings),
    )


def get_qc_matrix(adata: ad.AnnData, config: QCConfig) -> tuple[ExpressionMatrix, str]:
    """Resolve the matrix source requested for QC metric calculation.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Returns:
        Tuple containing the selected matrix and a source label.

    Raises:
        QCInputValidationError: If the requested matrix source is unavailable or ambiguous.
    """

    return resolve_qc_matrix(adata, layer=config.metrics.layer, use_raw=config.metrics.use_raw)


def resolve_qc_matrix(
    adata: ad.AnnData, *, layer: str | None = None, use_raw: bool = False
) -> tuple[ExpressionMatrix, str]:
    """Resolve one explicit count source for metrics, lineage, and gene evidence."""

    if use_raw and layer is not None:
        raise QCInputValidationError(
            "QC metric configuration cannot request both use_raw=true and a layer. "
            "Choose either AnnData.raw.X or one AnnData layer."
        )

    if use_raw:
        if adata.raw is None:
            raise QCInputValidationError(
                "QC metric configuration requested use_raw=true, but AnnData.raw is missing."
            )

        return adata.raw.X, "raw.X"

    if layer is not None:
        if layer not in adata.layers:
            raise QCInputValidationError(
                f"QC metric layer '{layer}' was requested but is missing " "from AnnData.layers."
            )

        return adata.layers[layer], f"layers[{layer}]"

    return adata.X, "X"


def validate_qc_matrix(
    matrix: ExpressionMatrix,
    *,
    expected_n_obs: int,
    matrix_source: str,
) -> tuple[int, int]:
    """Validate a selected QC matrix.

    Args:
        matrix: Candidate matrix object.
        expected_n_obs: Expected observation count from AnnData.
        matrix_source: Human-readable matrix source label.

    Returns:
        Tuple of matrix observation count and matrix variable count.

    Raises:
        QCInputValidationError: If the matrix is malformed or unsuitable for QC.
    """

    if not hasattr(matrix, "shape"):
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' does not expose a shape attribute."
        )

    shape = matrix.shape

    if len(shape) != 2:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must be two-dimensional. "
            f"Received shape: {shape}."
        )

    matrix_n_obs = int(shape[0])
    matrix_n_vars = int(shape[1])

    if matrix_n_obs != expected_n_obs:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' has {matrix_n_obs} observations, "
            f"but AnnData has {expected_n_obs} observations."
        )

    if matrix_n_vars <= 0:
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must contain at least one variable."
        )

    if sp.issparse(matrix):
        _validate_numeric_dtype(matrix.dtype, matrix_source=matrix_source)
        values = (
            matrix.data
            if matrix.format in {"csr", "csc", "coo", "bsr", "dia"}
            else matrix.tocoo(copy=False).data
        )
    else:
        values = np.asarray(matrix)
        _validate_numeric_dtype(values.dtype, matrix_source=matrix_source)
    _validate_count_values(values, matrix_source=matrix_source)
    return matrix_n_obs, matrix_n_vars


def validate_mixture_groupby_columns(adata: ad.AnnData, config: QCConfig) -> None:
    """Validate that the mixture model's groupby columns exist in ``AnnData.obs``.

    Args:
        adata: AnnData object.
        config: QC configuration.

    Raises:
        QCInputValidationError: If one or more requested groupby columns are missing.
    """

    if not config.mito_mixture.enabled:
        return

    requested: list[str] = list(config.mito_mixture.groupby)
    for level in config.mito_mixture.fallback_groupby:
        requested.extend(level)

    missing_columns = [
        column for column in dict.fromkeys(requested) if column not in adata.obs.columns
    ]

    if missing_columns:
        raise QCInputValidationError(
            "Mixture-model group-wise QC requested missing AnnData.obs column(s): "
            f"{', '.join(missing_columns)}."
        )


def require_obs_columns(adata: ad.AnnData, columns: Sequence[str]) -> None:
    """Require one or more AnnData.obs columns.

    Args:
        adata: AnnData object.
        columns: Observation columns that must exist.

    Raises:
        QCInputValidationError: If any requested column is missing.
    """

    if isinstance(columns, str):
        raise QCInputValidationError("columns must be a sequence of strings, not a string.")

    missing_columns = [column for column in columns if column not in adata.obs.columns]

    if missing_columns:
        raise QCInputValidationError(
            "AnnData.obs is missing required column(s): " f"{', '.join(missing_columns)}."
        )


def summarize_adata_shape(adata: ad.AnnData) -> dict[str, int]:
    """Summarize AnnData shape for QC provenance.

    Args:
        adata: AnnData object.

    Returns:
        Dictionary containing n_obs and n_vars.

    Raises:
        QCInputValidationError: If the input is not AnnData.
    """

    if not isinstance(adata, ad.AnnData):
        raise QCInputValidationError(
            "summarize_adata_shape expected an AnnData object. "
            f"Received: {type(adata).__name__}."
        )

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
    """Validate duplicate-name status against configured duplicate-name policy.

    Args:
        names_are_unique: Whether the relevant AnnData index is unique.
        policy: Duplicate-name policy configuration.
        axis_name: Either obs_names or var_names.

    Returns:
        Non-fatal warning messages.

    Raises:
        QCInputValidationError: If duplicate names exist and policy is error.
    """

    if names_are_unique:
        return []

    if axis_name == "obs_names":
        axis_policy = policy.obs_names

    elif axis_name == "var_names":
        axis_policy = policy.var_names

    else:
        raise QCInputValidationError(
            "Duplicate-name validation axis must be 'obs_names' or 'var_names'. "
            f"Received: {axis_name}."
        )

    if axis_policy == "error":
        raise QCInputValidationError(
            f"AnnData.{axis_name} contains duplicate values and policy is 'error'."
        )

    if axis_policy == "ignore":
        return []

    if axis_policy == "make_unique":
        return [
            f"AnnData.{axis_name} contains duplicate values and policy is "
            "'make_unique'. The QC stage should make names unique before metric calculation."
        ]

    if axis_policy == "warn":
        return [f"AnnData.{axis_name} contains duplicate values."]

    raise QCInputValidationError(
        f"Unsupported duplicate-name policy '{axis_policy}' for AnnData.{axis_name}."
    )


def _validate_numeric_dtype(dtype: np.dtype[Any], *, matrix_source: str) -> None:
    """Validate that a matrix dtype is numeric.

    Args:
        dtype: Candidate NumPy dtype.
        matrix_source: Human-readable matrix source label.

    Raises:
        QCInputValidationError: If dtype is not numeric.
    """

    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' must be numeric (real-valued). "
            f"Received dtype: {dtype}."
        )


_FINITE_CHECK_CHUNK_SIZE = 1_048_576


def _all_finite_bounded(values: npt.NDArray[Any]) -> bool:
    """Check values using at most one chunk of boolean scratch space."""
    with np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_FINITE_CHECK_CHUNK_SIZE,
    ) as chunks:
        return all(bool(np.isfinite(chunk).all()) for chunk in chunks)


def _validate_count_values(values: npt.NDArray[Any], *, matrix_source: str) -> None:
    """Apply the same finite, non-negative contract to dense or stored sparse values."""
    if not _all_finite_bounded(values):
        raise QCInputValidationError(
            f"QC matrix source '{matrix_source}' contains NaN or infinite values."
        )
    if values.size and np.min(values) < 0:
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
    "validate_mixture_groupby_columns",
    "validate_qc_input_adata",
    "validate_qc_matrix",
]
