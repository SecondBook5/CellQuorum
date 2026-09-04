"""Shared utilities for trajectory-visualization methods.

Combines:
- input resolvers (basis, pseudotime, numeric obs, results paths)
- house-style figure saving (theme, save, artifact wrapping)
- biology-agnostic plotting primitives (scatter, bar, heatmap, violin)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP, apply_cellquorum_theme

# ══════════════════════════════════════════════════════════════════════════════
# Input resolvers (from inputs.py)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Figure saving (from save.py)
# ══════════════════════════════════════════════════════════════════════════════


def apply_theme() -> None:
    """Apply the house theme plus trajectory-viz vector-font overrides."""
    apply_cellquorum_theme()
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


# save_figure/figure_artifacts are re-exported, not redefined: the local copy was
# a bare savefig loop, which is what left a 38 KB truncated velocity_stream.pdf
# and no PNG when the stream figure raised mid-write. See visualization.figio.


# ══════════════════════════════════════════════════════════════════════════════
# Plotting primitives (from plots.py)
# ══════════════════════════════════════════════════════════════════════════════


def _signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    mag = max(abs(lo), abs(hi), 1e-6)
    eps = mag * 1e-3
    return TwoSlopeNorm(vmin=min(lo, -eps), vcenter=0.0, vmax=max(hi, eps))


def embedding_scatter(coords: Any, values: Any, *, title: str, cbar_label: str) -> Figure:
    coords = np.asarray(coords, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.asarray(values, dtype=float),
        s=8,
        cmap=SEQUENTIAL_CMAP,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label=cbar_label)
    return fig


def signed_diverging_bar(labels: Any, values: Any, *, title: str) -> Figure:
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="stable")
    labels = [labels[i] for i in order]
    values = values[order]
    norm = _signed_norm(values)
    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(5, max(2.0, 0.4 * len(labels))))
    ax.barh(range(len(labels)), values, color=[cmap(norm(v)) for v in values])
    ax.axvline(0, color="0.4", linewidth=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    return fig


def matrix_heatmap(
    matrix: Any, row_labels: Any, col_labels: Any, *, title: str, cbar_label: str
) -> Figure:
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(col_labels), 1.2 + 0.4 * len(row_labels)))
    im = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    return fig


def grouped_violin(groups: Any, *, title: str, ylabel: str) -> Figure:
    keys = sorted(groups)
    data = [np.asarray(groups[k], dtype=float) for k in keys]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(keys)), 4))
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


__all__ = [
    # Input resolvers
    "VizInputError",
    "resolve_basis",
    "available_pseudotimes",
    "numeric_obs",
    "results_file",
    # Figure saving
    "apply_theme",
    "save_figure",
    "figure_artifacts",
    # Plotting primitives
    "embedding_scatter",
    "signed_diverging_bar",
    "matrix_heatmap",
    "grouped_violin",
]
