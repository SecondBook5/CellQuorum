"""Normalization transformations for scRNA-seq count matrices."""

from __future__ import annotations

# Import dataclass and field for structured result objects.
from dataclasses import dataclass, field

# Import Path for backend scratch-file exchange.
from pathlib import Path

# Import AnnData for single-cell data handling.
import anndata as ad

# Import numpy for count matrix transformations.
import numpy as np

# Import scipy sparse for sparse matrix support.
import scipy.sparse as sp

# Import layer tagging for provenance contracts.
from cellquorum.contracts import set_layer_tag

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
    use_gpu: bool = False,
    backend: object | None = None,
    scratch_dir: str | Path | None = None,
) -> NormalizationResult:
    """
    Normalize an AnnData object using a configured recipe.

    Args:
        adata: Input AnnData object.
        config: Normalization configuration.
        copy: Whether to copy the AnnData object before mutation.
        use_gpu: Whether to use GPU acceleration (requires cupy).
        backend: Optional scclr subprocess backend, required by the PFlog1pPF
            recipe (``cellquorum_pf_log1p_pf_v1``), which runs the real
            Booeshaghi/Pachter transform in the isolated scclr environment.
        scratch_dir: Scratch directory for backend file exchange.

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

    # PFlog1pPF is the real Booeshaghi/Pachter transform, run through the
    # isolated scclr environment (sparse, no densify). It returns a sparse layer
    # plus a per-cell row_center, so it takes a dedicated path rather than the
    # pure-matrix recipe router.
    if config.recipe == "cellquorum_pf_log1p_pf_v1":
        normalized_matrix, recipe_diagnostics, recipe_warnings = run_scclr_pflog(
            matrix=input_matrix,
            adata=working_adata,
            config=config,
            backend=backend,
            scratch_dir=scratch_dir,
        )
    else:
        # Apply the configured (pure-matrix) normalization recipe.
        normalized_matrix, recipe_diagnostics, recipe_warnings = apply_normalization_recipe(
            matrix=input_matrix,
            recipe=config.recipe,
            target_sum=config.target_sum,
            pseudocount=config.pseudocount,
            use_gpu=use_gpu,
        )

    # Write normalized matrix to output layer.
    write_normalized_layer(
        working_adata,
        normalized_matrix=normalized_matrix,
        output_layer=config.output_layer,
        overwrite=config.overwrite,
    )

    # scclr PFlog1pPF returns a per-cell row_center: persist it to obs so the
    # downstream sparse PCA can reconstruct the implicit-centered matrix without
    # densifying. Pop it out of the diagnostics (numpy array, not JSON-friendly)
    # before provenance is written.
    row_center = recipe_diagnostics.pop("scclr_row_center", None)
    if row_center is not None and len(row_center) == working_adata.n_obs:
        working_adata.obs[f"{config.output_layer}_row_center"] = np.asarray(row_center)

    # Write provenance to uns.
    write_normalization_provenance(
        working_adata,
        config=config,
        diagnostics=recipe_diagnostics,
    )

    # Tag the preserved counts and normalized layers so downstream DataContracts
    # can verify layer identity by provenance, not by name alone (closes the
    # expected_kind seam from the Phase-1 review).
    set_layer_tag(working_adata, config.preserve_counts_layer, kind="counts")
    set_layer_tag(
        working_adata,
        config.output_layer,
        kind="lognorm",
        recipe=config.recipe,
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
    use_gpu: bool = False,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, object], list[str]]:
    """
    Apply a normalization recipe to a count matrix.

    Args:
        matrix: Input count matrix.
        recipe: Recipe name.
        target_sum: Target sum for scaling recipes.
        pseudocount: Pseudocount for log recipes.
        use_gpu: Whether to use GPU acceleration (only applies to pf_log1p_pf_v1).

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
        # PFlog1pPF runs through the scclr backend (see run_scclr_pflog), which
        # needs AnnData + backend context this pure-matrix router does not have.
        raise PreprocessingNormalizationError(
            "The 'cellquorum_pf_log1p_pf_v1' (PFlog1pPF) recipe runs through the scclr "
            "backend and is dispatched by normalize_adata, not apply_normalization_recipe."
        )
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


def run_scclr_pflog(
    *,
    matrix: np.ndarray | sp.spmatrix,
    adata: ad.AnnData,
    config: NormalizationConfig,
    backend: object | None,
    scratch_dir: str | Path | None,
) -> tuple[sp.spmatrix, dict[str, object], list[str]]:
    """
    Run the real PFlog1pPF (scclr) normalization via the isolated-env backend.

    This is the true Booeshaghi/Pachter transform: ``center(log1p(4·alpha·x))``
    stored sparsity-preserving as the sparse log values plus a per-cell
    ``row_center`` vector (dense value is ``sparse[i,j] - row_center[i]``, never
    materialized). The compute runs in the isolated ``scclr`` environment via the
    subprocess backend; nothing is densified here.

    The per-cell ``row_center`` and the ``k``/``alpha`` scalars are returned in
    the diagnostics dict under ``scclr_row_center``/``scclr_k``/``scclr_alpha`` so
    ``normalize_adata`` can persist them to ``obs``/``uns``.

    Args:
        matrix: Input raw-count matrix (cells x genes).
        adata: Working AnnData (used for obs_names length / provenance).
        config: Normalization configuration (reads ``scclr_target``).
        backend: scclr subprocess backend (must be available).
        scratch_dir: Directory for temp file exchange with the backend.

    Returns:
        Tuple of (sparse PFlog matrix, diagnostics, warnings).

    Raises:
        PreprocessingNormalizationError: If the scclr backend is unavailable or
            the helper fails — there is no fallback (the real transform is the
            only PFlog1pPF path).
    """

    import json
    import tempfile

    from cellquorum.backends.scclr_backend import PFLOG_HELPER, ScclrBackend

    # Require an available scclr backend — fail loud, no silent fallback.
    if backend is None or not isinstance(backend, ScclrBackend):
        raise PreprocessingNormalizationError(
            "The 'cellquorum_pf_log1p_pf_v1' (PFlog1pPF) recipe requires the scclr backend. "
            "Create the isolated env, e.g. "
            "`micromamba create -n scclr python=3.13 rust maturin pip` then "
            "`micromamba run -n scclr pip install -e /path/to/scclr`."
        )
    status = backend.status()
    if not status.available:
        raise PreprocessingNormalizationError(
            "The scclr backend is unavailable "
            f"(missing: {', '.join(status.missing) or 'unknown'}). "
            "PFlog1pPF normalization cannot run without the isolated scclr environment."
        )

    # Resolve the scratch directory for backend file exchange.
    scratch = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)

    # scclr expects a CSR counts matrix.
    counts = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))

    target = str(getattr(config, "scclr_target", "auto"))

    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp_path = Path(tmp)
        counts_path = tmp_path / "counts.npz"
        matrix_out = tmp_path / "pflog.npz"
        meta_out = tmp_path / "meta.json"
        sp.save_npz(counts_path, counts)

        result = backend.run_helper(
            PFLOG_HELPER,
            [
                "normalize",
                str(counts_path),
                str(matrix_out),
                str(meta_out),
                "--target",
                target,
            ],
        )
        if result.returncode != 0 or not matrix_out.is_file():
            raise PreprocessingNormalizationError(
                "scclr PFlog1pPF normalization failed: "
                f"{result.stderr.strip()[:500] or 'no stderr'}"
            )

        normalized = sp.load_npz(matrix_out)
        meta = json.loads(meta_out.read_text())

    row_center = np.asarray(meta.get("row_center", []), dtype=float)

    diagnostics: dict[str, object] = {
        "recipe_impl": "scclr",
        "scclr_target": target,
        "scclr_k": meta.get("k"),
        "scclr_alpha": meta.get("alpha"),
        "scclr_row_center": row_center,
        "output_is_sparse": True,
    }
    return normalized, diagnostics, []


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
    "run_scclr_pflog",
    "apply_recipe_pf_v1",
    "build_normalization_diagnostics",
    "get_input_matrix",
    "normalize_adata",
    "preserve_raw_counts",
    "validate_count_matrix",
    "write_normalization_provenance",
    "write_normalized_layer",
]
