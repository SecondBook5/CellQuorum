"""Shared input resolvers for the trajectory-visualization methods."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

_BASIS_FALLBACK = ("X_umap", "X_diffmap")
_PSEUDOTIME_KEYS = ("dpt_pseudotime", "palantir_pseudotime", "velocity_pseudotime")


class VizInputError(RuntimeError):
    """A required figure input is absent or the wrong dtype."""


def resolve_basis(adata: ad.AnnData, configured: str | None) -> str | None:
    if configured is not None and configured in adata.obsm:
        return configured
    for key in _BASIS_FALLBACK:
        if key in adata.obsm:
            return key
    return None


def available_pseudotimes(adata: ad.AnnData, configured: list[str] | None) -> list[str]:
    present = [k for k in _PSEUDOTIME_KEYS if k in adata.obs]
    if configured is not None:
        wanted = set(configured)
        present = [k for k in present if k in wanted]
    return sorted(present)


def numeric_obs(adata: ad.AnnData, key: str) -> np.ndarray:
    try:
        return np.asarray(adata.obs[key], dtype="float64")
    except (ValueError, TypeError) as exc:
        raise VizInputError(f"obs['{key}'] is not numeric: {exc}") from exc


def results_file(context: object, *parts: str) -> Path:
    return Path(context.paths.results).joinpath("trajectory", *parts)


__all__ = ["VizInputError", "resolve_basis", "available_pseudotimes", "numeric_obs", "results_file"]
