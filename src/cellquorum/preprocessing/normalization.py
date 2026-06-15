"""Normalization transformations for scRNA-seq count matrices."""

from __future__ import annotations

# Import dataclass and field for structured result objects.
from dataclasses import dataclass, field

# Import AnnData for single-cell data handling.
import anndata as ad

# Import numpy for count matrix transformations.
import numpy as np

# Import scipy sparse for sparse matrix support.
import scipy.sparse as sp

# Import shared CellQuorum data exception.
from cellquorum.core.exceptions import CellQuorumDataError

# Import normalization configuration.
from cellquorum.preprocessing.config import NormalizationConfig


class PreprocessingNormalizationError(CellQuorumDataError):
    """
    Report normalization failures.

    Normalization failures can happen from invalid input matrices, negative
    counts, zero-depth cells, unsupported sparse formats, or configuration
    errors.
    """


@dataclass(frozen=True)
class NormalizationResult:
    """
    Store the result of normalizing an AnnData object.

    Args:
        adata: Normalized AnnData object with output layer written.
        recipe: Normalization recipe that was applied.
        input_layer: Input layer name or None for X.
        output_layer: Output layer name where normalized values were written.
        preserve_counts_layer: Layer name where raw counts were preserved.
        diagnostics: Structured diagnostic metrics.
        warnings: Warning messages produced during normalization.
    """

    # Store the normalized AnnData object.
    adata: ad.AnnData

    # Store the recipe that was applied.
    recipe: str

    # Store the input layer name.
    input_layer: str | None

    # Store the output layer name.
    output_layer: str

    # Store the raw-counts preservation layer name.
    preserve_counts_layer: str

    # Store diagnostic metrics.
    diagnostics: dict[str, object] = field(default_factory=dict)

    # Store warnings.
    warnings: list[str] = field(default_factory=list)


def normalize_adata(
    adata: ad.AnnData,
    config: NormalizationConfig,
    *,
    copy: bool = True,
) -> NormalizationResult:
    """
    Normalize an AnnData object using a configured recipe.

    Args:
        adata: Input AnnData object.
        config: Normalization configuration.
        copy: Whether to copy the AnnData object before mutation.

    Returns:
        NormalizationResult with normalized AnnData and diagnostics.

    Raises:
        PreprocessingNormalizationError: If normalization fails.
    """

    # Validate input types.
    if not isinstance(adata, ad.AnnData):
        raise PreprocessingNormalizationError(
            "normalize_adata expected an AnnData object. " f"Received: {type(adata).__name__}."
        )

    # Validate config type.
    if not isinstance(config, NormalizationConfig):
        raise PreprocessingNormalizationError(
            "config must be a NormalizationConfig. " f"Received: {type(config).__name__}."
        )

    # Copy AnnData when requested.
    working_adata = adata.copy() if copy else adata

    # Retrieve input matrix before preserving to validate layer existence.
    input_matrix = get_input_matrix(working_adata, config.input_layer)

    # Preserve raw counts before normalization.
    working_adata = preserve_raw_counts(
        working_adata,
        input_layer=config.input_layer,
        preserve_layer=config.preserve_counts_layer,
    )

    # Validate input matrix.
    validate_count_matrix(input_matrix)

    # Apply the configured normalization recipe.
    normalized_matrix, recipe_diagnostics, recipe_warnings = apply_normalization_recipe(
        matrix=input_matrix,
        recipe=config.recipe,
        target_sum=config.target_sum,
        pseudocount=config.pseudocount,
    )

    # Write normalized matrix to output layer.
    write_normalized_layer(
        working_adata,
        normalized_matrix=normalized_matrix,
        output_layer=config.output_layer,
        overwrite=config.overwrite,
    )

    # Write provenance to uns.
    write_normalization_provenance(
        working_adata,
        config=config,
        diagnostics=recipe_diagnostics,
    )

    # Build final diagnostics.
    diagnostics = build_normalization_diagnostics(
        adata=working_adata,
        input_matrix=input_matrix,
        config=config,
        recipe_diagnostics=recipe_diagnostics,
    )

    # Return the normalization result.
    return NormalizationResult(
        adata=working_adata,
        recipe=config.recipe,
        input_layer=config.input_layer,
        output_layer=config.output_layer,
        preserve_counts_layer=config.preserve_counts_layer,
        diagnostics=diagnostics,
        warnings=recipe_warnings,
    )


def get_input_matrix(adata: ad.AnnData, input_layer: str | None) -> np.ndarray | sp.spmatrix:
    """
    Retrieve input matrix from AnnData.

    Args:
        adata: AnnData object.
        input_layer: Input layer name or None for X.

    Returns:
        Input count matrix.

    Raises:
        PreprocessingNormalizationError: If the input layer is missing.
    """

    # Retrieve X when input_layer is None.
    if input_layer is None:
        return adata.X

    # Retrieve named layer.
    if input_layer not in adata.layers:
        raise PreprocessingNormalizationError(
            f"Input layer '{input_layer}' not found in AnnData.layers."
        )

    # Return the input layer.
    return adata.layers[input_layer]


def validate_count_matrix(matrix: np.ndarray | sp.spmatrix) -> None:
    """
    Validate that a matrix is suitable for normalization.

    Args:
        matrix: Count matrix.

    Raises:
        PreprocessingNormalizationError: If the matrix is invalid.
    """

    # Check for dense or sparse matrix types.
    is_sparse = sp.issparse(matrix)

    # Validate that the matrix is numeric.
    if not np.issubdtype(matrix.dtype, np.number):
        raise PreprocessingNormalizationError("Count matrix must be numeric.")

    # Convert to dense for validation when sparse.
    check_matrix = matrix.toarray() if is_sparse else matrix

    # Reject matrices with non-finite values.
    if not np.isfinite(check_matrix).all():
        raise PreprocessingNormalizationError(
            "Count matrix contains non-finite values (NaN or Inf)."
        )

    # Reject matrices with negative values.
    if (check_matrix < 0).any():
        raise PreprocessingNormalizationError("Count matrix contains negative values.")


def preserve_raw_counts(
    adata: ad.AnnData,
    input_layer: str | None,
    preserve_layer: str,
) -> ad.AnnData:
    """
    Preserve raw counts in a separate layer.

    Args:
        adata: AnnData object.
        input_layer: Input layer name or None for X.
        preserve_layer: Preservation layer name.

    Returns:
        AnnData with preserved raw counts layer.
    """

    # Skip preservation when layer already exists.
    if preserve_layer in adata.layers:
        return adata

    # Preserve X when input_layer is None.
    if input_layer is None:
        adata.layers[preserve_layer] = adata.X.copy()
    else:
        # Preserve named input layer.
        adata.layers[preserve_layer] = adata.layers[input_layer].copy()

    # Return AnnData with preserved counts.
    return adata


def apply_normalization_recipe(
    matrix: np.ndarray | sp.spmatrix,
    recipe: str,
    target_sum: float,
    pseudocount: float,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply a normalization recipe to a count matrix.

    Args:
        matrix: Input count matrix.
        recipe: Recipe name.
        target_sum: Target sum for scaling recipes.
        pseudocount: Pseudocount for log recipes.

    Returns:
        Tuple of (normalized matrix, diagnostics, warnings).

    Raises:
        PreprocessingNormalizationError: If the recipe is unknown or fails.
    """

    # Route to recipe implementation.
    if recipe == "none":
        return apply_recipe_none(matrix)
    elif recipe == "cellquorum_pf_v1":
        return apply_recipe_pf_v1(matrix)
    elif recipe == "cellquorum_log1p_cp10k_v1":
        return apply_recipe_log1p_cp10k_v1(matrix, target_sum)
    elif recipe == "cellquorum_log1p_pf_v1":
        return apply_recipe_log1p_pf_v1(matrix)
    elif recipe == "cellquorum_pf_log1p_pf_v1":
        return apply_recipe_pf_log1p_pf_v1(matrix, pseudocount)
    else:
        raise PreprocessingNormalizationError(f"Unknown normalization recipe: {recipe}.")


def apply_recipe_none(
    matrix: np.ndarray | sp.spmatrix,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply the 'none' recipe (pass-through).

    Args:
        matrix: Input matrix.

    Returns:
        Tuple of (matrix copy, diagnostics, warnings).
    """

    # Copy the matrix unchanged.
    output = matrix.copy()

    # Return with empty diagnostics and warnings.
    return output, {}, []


def apply_recipe_pf_v1(
    matrix: np.ndarray | sp.spmatrix,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply the 'cellquorum_pf_v1' recipe (proportional fractions).

    Each cell is scaled by its total count to produce per-cell proportions:
    x_ij / sum_j(x_ij)

    Args:
        matrix: Input count matrix (cells x genes).

    Returns:
        Tuple of (normalized matrix, diagnostics, warnings).
    """

    # Compute per-cell total counts.
    is_sparse = sp.issparse(matrix)
    cell_totals = np.asarray(matrix.sum(axis=1)).flatten()

    # Detect zero-depth cells.
    zero_depth_mask = cell_totals == 0
    n_zero_depth = int(zero_depth_mask.sum())

    # Build warnings for zero-depth cells.
    warnings = []
    if n_zero_depth > 0:
        warnings.append(
            f"Found {n_zero_depth} zero-depth cells. "
            "These will produce zero or NaN normalized values."
        )

    # Avoid division by zero.
    safe_cell_totals = np.where(cell_totals > 0, cell_totals, 1.0)

    # Compute proportional fractions.
    if is_sparse:
        # Sparse division by row.
        output = matrix.multiply(1.0 / safe_cell_totals[:, np.newaxis])
        # Convert back to CSR format for AnnData compatibility.
        output = output.tocsr()
    else:
        # Dense division.
        output = matrix / safe_cell_totals[:, np.newaxis]

    # Build diagnostics.
    diagnostics = {
        "n_zero_depth_cells": n_zero_depth,
    }

    # Return normalized matrix, diagnostics, and warnings.
    return output, diagnostics, warnings


def apply_recipe_log1p_cp10k_v1(
    matrix: np.ndarray | sp.spmatrix,
    target_sum: float,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply the 'cellquorum_log1p_cp10k_v1' recipe (counts per 10k + log1p).

    Each cell is scaled to target_sum and log1p-transformed:
    log(1 + x_ij / sum_j(x_ij) * target_sum)

    Args:
        matrix: Input count matrix (cells x genes).
        target_sum: Target sum for scaling.

    Returns:
        Tuple of (normalized matrix, diagnostics, warnings).
    """

    # Compute per-cell total counts.
    is_sparse = sp.issparse(matrix)
    cell_totals = np.asarray(matrix.sum(axis=1)).flatten()

    # Detect zero-depth cells.
    zero_depth_mask = cell_totals == 0
    n_zero_depth = int(zero_depth_mask.sum())

    # Build warnings for zero-depth cells.
    warnings = []
    if n_zero_depth > 0:
        warnings.append(
            f"Found {n_zero_depth} zero-depth cells. " "These will produce zero normalized values."
        )

    # Avoid division by zero.
    safe_cell_totals = np.where(cell_totals > 0, cell_totals, 1.0)

    # Compute scaled counts.
    if is_sparse:
        scaled = matrix.multiply(target_sum / safe_cell_totals[:, np.newaxis])
    else:
        scaled = matrix * (target_sum / safe_cell_totals[:, np.newaxis])

    # Apply log1p transformation.
    if is_sparse:
        output = scaled.copy()
        output.data = np.log1p(output.data)
        # Convert back to CSR format for AnnData compatibility.
        output = output.tocsr()
    else:
        output = np.log1p(scaled)

    # Build diagnostics.
    diagnostics = {
        "n_zero_depth_cells": n_zero_depth,
        "target_sum": target_sum,
    }

    # Return normalized matrix, diagnostics, and warnings.
    return output, diagnostics, warnings


def apply_recipe_log1p_pf_v1(
    matrix: np.ndarray | sp.spmatrix,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply the 'cellquorum_log1p_pf_v1' recipe (log1p proportional fractions).

    Each cell is scaled to proportions and log1p-transformed:
    log(1 + x_ij / sum_j(x_ij))

    Args:
        matrix: Input count matrix (cells x genes).

    Returns:
        Tuple of (normalized matrix, diagnostics, warnings).
    """

    # Compute per-cell total counts.
    is_sparse = sp.issparse(matrix)
    cell_totals = np.asarray(matrix.sum(axis=1)).flatten()

    # Detect zero-depth cells.
    zero_depth_mask = cell_totals == 0
    n_zero_depth = int(zero_depth_mask.sum())

    # Build warnings for zero-depth cells.
    warnings = []
    if n_zero_depth > 0:
        warnings.append(
            f"Found {n_zero_depth} zero-depth cells. " "These will produce zero normalized values."
        )

    # Avoid division by zero.
    safe_cell_totals = np.where(cell_totals > 0, cell_totals, 1.0)

    # Compute proportional fractions.
    if is_sparse:
        pf = matrix.multiply(1.0 / safe_cell_totals[:, np.newaxis])
    else:
        pf = matrix / safe_cell_totals[:, np.newaxis]

    # Apply log1p transformation.
    if is_sparse:
        output = pf.copy()
        output.data = np.log1p(output.data)
        # Convert back to CSR format for AnnData compatibility.
        output = output.tocsr()
    else:
        output = np.log1p(pf)

    # Build diagnostics.
    diagnostics = {
        "n_zero_depth_cells": n_zero_depth,
    }

    # Return normalized matrix, diagnostics, and warnings.
    return output, diagnostics, warnings


def apply_recipe_pf_log1p_pf_v1(
    matrix: np.ndarray | sp.spmatrix,
    pseudocount: float,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply the 'cellquorum_pf_log1p_pf_v1' recipe (shifted CLR-like).

    This is a shifted centered-log-ratio-like transformation:
    u_ij = x_ij / sum_j(x_ij)
    z_ij = log(u_ij + pseudocount) - mean_j(log(u_j + pseudocount))

    Note: This transformation is mathematically dense (all values become non-zero
    after centering). Sparse input matrices are densified before computation.

    Args:
        matrix: Input count matrix (cells x genes).
        pseudocount: Pseudocount added before log.

    Returns:
        Tuple of (normalized matrix, diagnostics, warnings).
    """

    # Compute per-cell total counts.
    is_sparse = sp.issparse(matrix)
    cell_totals = np.asarray(matrix.sum(axis=1)).flatten()

    # Detect zero-depth cells.
    zero_depth_mask = cell_totals == 0
    n_zero_depth = int(zero_depth_mask.sum())

    # Build warnings.
    warnings = []
    if n_zero_depth > 0:
        warnings.append(
            f"Found {n_zero_depth} zero-depth cells. "
            "These will produce NaN or Inf normalized values."
        )

    # Warn about densification for sparse input.
    densified_sparse_input = False
    if is_sparse:
        warnings.append(
            "CLR-like transformation densifies sparse matrices because centering produces "
            "mathematically dense output."
        )
        densified_sparse_input = True

    # Avoid division by zero.
    safe_cell_totals = np.where(cell_totals > 0, cell_totals, 1.0)

    # Densify sparse matrix for CLR-like computation.
    working_matrix = matrix.toarray() if is_sparse else matrix

    # Compute proportional fractions u.
    u = working_matrix / safe_cell_totals[:, np.newaxis]

    # Add pseudocount and apply log.
    u_plus = np.log(u + pseudocount)

    # Compute per-cell mean of log(u + pseudocount).
    per_cell_mean = u_plus.mean(axis=1)

    # Center by subtracting per-cell mean.
    output = u_plus - per_cell_mean[:, np.newaxis]

    # Build diagnostics.
    diagnostics = {
        "n_zero_depth_cells": n_zero_depth,
        "pseudocount": pseudocount,
        "densified_sparse_input": densified_sparse_input,
        "output_is_dense": True,
    }

    # Return normalized matrix, diagnostics, and warnings.
    return output, diagnostics, warnings


def write_normalized_layer(
    adata: ad.AnnData,
    normalized_matrix: np.ndarray | sp.spmatrix,
    output_layer: str,
    overwrite: bool,
) -> None:
    """
    Write normalized matrix to AnnData layer.

    Args:
        adata: AnnData object.
        normalized_matrix: Normalized matrix.
        output_layer: Output layer name.
        overwrite: Whether to overwrite existing layers.

    Raises:
        PreprocessingNormalizationError: If layer exists and overwrite is False.
    """

    # Check for existing output layer.
    if output_layer in adata.layers and not overwrite:
        raise PreprocessingNormalizationError(
            f"Output layer '{output_layer}' already exists. " "Set overwrite=True to replace it."
        )

    # Write normalized matrix to output layer.
    adata.layers[output_layer] = normalized_matrix


def write_normalization_provenance(
    adata: ad.AnnData,
    config: NormalizationConfig,
    diagnostics: dict[str, object],
) -> None:
    """
    Write normalization provenance to AnnData.uns.

    Args:
        adata: AnnData object.
        config: Normalization configuration.
        diagnostics: Diagnostic metrics.
    """

    # Initialize uns structure.
    if "cellquorum" not in adata.uns:
        adata.uns["cellquorum"] = {}

    # Initialize preprocessing structure.
    if "preprocessing" not in adata.uns["cellquorum"]:
        adata.uns["cellquorum"]["preprocessing"] = {}

    # Write normalization provenance.
    adata.uns["cellquorum"]["preprocessing"]["normalization"] = {
        "recipe": config.recipe,
        "input_layer": config.input_layer,
        "output_layer": config.output_layer,
        "preserve_counts_layer": config.preserve_counts_layer,
        "target_sum": config.target_sum,
        "pseudocount": config.pseudocount,
        "diagnostics": diagnostics,
    }


def build_normalization_diagnostics(
    adata: ad.AnnData,
    input_matrix: np.ndarray | sp.spmatrix,
    config: NormalizationConfig,
    recipe_diagnostics: dict[str, object],
) -> dict[str, object]:
    """
    Build final normalization diagnostics.

    Args:
        adata: AnnData object.
        input_matrix: Input count matrix.
        config: Normalization configuration.
        recipe_diagnostics: Recipe-specific diagnostics.

    Returns:
        Complete diagnostics dictionary.
    """

    # Compute input matrix summary statistics.
    cell_totals = np.asarray(input_matrix.sum(axis=1)).flatten()

    # Build complete diagnostics.
    diagnostics = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "input_total_counts_min": float(cell_totals.min()),
        "input_total_counts_max": float(cell_totals.max()),
        "recipe": config.recipe,
        "input_layer": config.input_layer,
        "output_layer": config.output_layer,
        "preserve_counts_layer": config.preserve_counts_layer,
        "target_sum": config.target_sum,
        "pseudocount": config.pseudocount,
        **recipe_diagnostics,
    }

    # Return diagnostics.
    return diagnostics


__all__ = [
    "NormalizationResult",
    "PreprocessingNormalizationError",
    "apply_normalization_recipe",
    "apply_recipe_log1p_cp10k_v1",
    "apply_recipe_log1p_pf_v1",
    "apply_recipe_none",
    "apply_recipe_pf_log1p_pf_v1",
    "apply_recipe_pf_v1",
    "build_normalization_diagnostics",
    "get_input_matrix",
    "normalize_adata",
    "preserve_raw_counts",
    "validate_count_matrix",
    "write_normalization_provenance",
    "write_normalized_layer",
]
