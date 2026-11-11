"""Access AnnData annotations and expression arrays for pandas and SciPy operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad


def resolve_use_rep(adata: ad.AnnData, configured: str | None, fallback: list[str]) -> str | None:
    """Select the requested embedding or the first available fallback in the given order."""
    if configured and configured in adata.obsm:
        return configured
    return next((name for name in fallback if name in adata.obsm), None)


def obs_frame(adata: ad.AnnData) -> pd.DataFrame:
    """Return the object's writable cell metadata, materializing only that axis if lazy."""
    if not isinstance(adata.obs, pd.DataFrame):
        adata.obs = adata.obs.to_memory()
    return adata.obs


def var_frame(adata: ad.AnnData) -> pd.DataFrame:
    """Return the object's writable feature metadata, materializing only that axis if lazy."""
    if not isinstance(adata.var, pd.DataFrame):
        adata.var = adata.var.to_memory()
    return adata.var


def memory_matrix(value: object) -> np.ndarray | sparse.spmatrix | sparse.sparray:
    """Obtain a CPU expression matrix for a NumPy/SciPy consumer, retaining sparsity."""
    if value is None:
        raise ValueError("The requested expression matrix is absent")
    to_memory = getattr(value, "to_memory", None)
    if callable(to_memory):
        value = to_memory()
    if isinstance(value, sparse.spmatrix | sparse.sparray):
        return value
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional expression matrix")
    return matrix


def copy_x_to_layer(adata: ad.AnnData, name: str) -> None:
    """Preserve the original array backend when copying X into a named layer."""
    matrix = adata.X
    copy = getattr(matrix, "copy", None)
    adata.layers[name] = copy() if callable(copy) else memory_matrix(matrix).copy()


def dense_matrix(value: object) -> np.ndarray:
    """Materialize the selected matrix for a consumer requiring a dense array."""
    matrix = memory_matrix(value)
    return matrix if isinstance(matrix, np.ndarray) else matrix.toarray()
