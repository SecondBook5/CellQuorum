"""Shared utilities for trajectory-visualization methods.

Combines:
- input resolvers (basis, pseudotime, numeric obs, results paths)
- house-style figure saving (theme, save, artifact wrapping)
- biology-agnostic plotting primitives (scatter, bar, heatmap, violin)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cellquorum.visualization.figio import figure_artifacts, save_figure
from cellquorum.visualization.figstyle import (
    MUTED_PALETTE,
    SEQUENTIAL_CMAP,
    apply_cellquorum_theme,
)

if TYPE_CHECKING:
    from cellquorum.core.context import PipelineContext

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


def results_file(context: PipelineContext, *parts: str) -> Path:
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
    """Scatter colored by a continuous obs value, with an outlier-robust scale.

    vmin/vmax are clipped to the 1st/99th percentile of the FINITE values
    rather than the raw min/max. Real bug this fixes: dpt_pseudotime on a
    real cohort had 99.84% of cells sitting in [0, 0.03] of its own range,
    with a handful of legitimate finite outlier cells reaching up to 1.0. A
    raw-min/max scale spans the full range for those outliers' sake and
    crushes the 99.84% majority into one indistinguishable shade -- a figure
    that looks broken even though every value is real. Values outside the
    clipped range still get drawn (at the nearest end color, via the
    colorbar's "extend" arrows), just not allowed to stretch the scale.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    finite_vals = values[finite]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    if finite_vals.size:
        vmin, vmax = (float(v) for v in np.percentile(finite_vals, [1, 99]))
        if vmax <= vmin:
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    else:
        vmin, vmax = 0.0, 1.0

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=8,
        cmap=SEQUENTIAL_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    extend = "neither"
    if finite_vals.size:
        below, above = finite_vals.min() < vmin, finite_vals.max() > vmax
        extend = "both" if (below and above) else "min" if below else "max" if above else "neither"
    fig.colorbar(sc, ax=ax, label=cbar_label, extend=extend)
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
    matrix: Any,
    row_labels: Any,
    col_labels: Any,
    *,
    title: str,
    cbar_label: str,
    row_group_breaks: list[int] | None = None,
) -> Figure:
    """Signed matrix as a diverging, zero-centered heatmap.

    ``row_group_breaks`` draws a thin separator line after each given row
    index (e.g. where consecutive rows change which column they peak in) --
    makes a real grouping in the row order read as intentional structure
    rather than an unexplained seam between otherwise-similar blocks.
    """
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(col_labels), 1.2 + 0.4 * len(row_labels)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=_signed_norm(matrix))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in row_group_breaks or []:
        ax.axhline(row - 0.5, color="white", lw=1.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    return fig


def off_diagonal_vmax(mat: np.ndarray) -> float:
    """Max of a square matrix's off-diagonal entries; 1.0 if all are zero.

    Used to scale a coarse-grained transition matrix's color range: a
    well-separated set of macrostates is EXPECTED to have a near-identity
    matrix (each state mostly transitions to itself), so a color scale built
    from the raw 0-1 range is dominated by that expected diagonal and
    drowns the actually-informative off-diagonal cross-state flow.
    """
    mat = np.asarray(mat, dtype=float)
    off = mat[~np.eye(mat.shape[0], dtype=bool)]
    vmax = float(off.max()) if off.size else 0.0
    return vmax if vmax > 0 else 1.0


def coarse_transition_heatmap(
    coarse_T: Any,
    stationary: Any | None = None,
    *,
    labels: dict[str, str] | None = None,
    title: str = "coarse-grained transition matrix",
) -> Figure:
    """Coarse-grained macrostate transition matrix, off-diagonal-focused.

    Diagonal cells (self-transition probability) are drawn in a fixed
    neutral fill, separate from the colorbar -- a near-1.0 diagonal is the
    EXPECTED signature of stable, well-separated macrostates, not a defect,
    but sharing one color scale with it drowns the off-diagonal cross-state
    flow that is actually informative. Only off-diagonal cells are colored,
    scaled to their own range (off_diagonal_vmax).

    Args:
        coarse_T: square (n_states, n_states) frame/array of transition
            probabilities, row-normalized. A pandas.DataFrame's index/columns
            supply the state names; a bare array falls back to positional
            names.
        stationary: optional length-n_states stationary distribution, drawn
            as a labeled strip below the matrix.
        labels: optional {state name -> display label}; a name with no entry
            keeps its bare form.
        title: figure title.
    """
    names = [str(c) for c in coarse_T.columns] if hasattr(coarse_T, "columns") else None
    mat = np.asarray(coarse_T, dtype=float)
    n = mat.shape[0]
    if names is None:
        names = [str(i) for i in range(n)]
    disp_names = [(labels or {}).get(nm, nm) for nm in names]

    cmap = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=off_diagonal_vmax(mat))

    # Fixed INCHES budget for x-tick labels + caption + stationary strip below
    # the matrix, independent of n -- an add_axes rect at a hardcoded figure
    # FRACTION does not scale as the matrix (and so the figure) grows with n,
    # which is what previously let the caption collide with the tick labels
    # and the stationary strip.
    bottom_in = 1.7
    fig_w = 1.3 + 0.75 * n
    fig_h = 1.3 + 0.75 * n + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    bottom_frac = bottom_in / fig_h
    ax = fig.add_axes([0.16, bottom_frac, 0.66, 0.88 - bottom_frac])
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if i == j:
                color, text_color = "#e3e3e3", "#555555"
            else:
                color = cmap(norm(val))
                text_color = "white" if norm(val) > 0.55 else "#222222"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5))
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(disp_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(disp_names)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Caption and stationary strip sit BELOW the rotated x-tick labels, at
    # fixed inch offsets from the figure's bottom edge (not the ax's), so
    # they never collide with those labels regardless of n.
    fig.text(
        0.5,
        0.62 / fig_h,
        "grey diagonal = probability of staying in the same macrostate; near 1.0 means the\n"
        "macrostates are stable and well-separated. Color covers only cross-state flow.",
        transform=fig.transFigure,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#6b6b6b",
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="cross-state transition prob.", fraction=0.046, pad=0.18)

    if stationary is not None:
        vals = np.asarray(stationary, dtype=float)
        bar_ax = fig.add_axes([0.16, 0.05 / fig_h, 0.66, 0.30 / fig_h])
        bar_ax.imshow(vals[None, :], aspect="auto", cmap="Blues", vmin=0, vmax=vals.max())
        bar_ax.set_xticks([])
        bar_ax.set_yticks([0])
        bar_ax.set_yticklabels(["stationary dist"], fontsize=7.5)
        for k, v in enumerate(vals):
            bar_ax.text(k, 0, f"{v:.2f}", ha="center", va="center", fontsize=7.5)

    return fig


def grouped_violin(
    groups: Any, *, title: str, ylabel: str, labels: dict[str, str] | None = None
) -> Figure:
    """One raincloud (half-violin + compact box + median + jittered points)
    per group, via the house raincloud primitive.

    Real complaint: a bare ``ax.violinplot`` reads as a different, unstyled
    tool produced this figure, next to every QC distribution figure in this
    codebase using ``figstyle.raincloud`` for the same job. This now reuses
    that exact primitive instead of reimplementing a plainer version of it.

    ``labels`` is an optional {raw group key -> display name} override for
    the x-tick TEXT ONLY -- sort order stays keyed on the raw group values
    (a bare "0"/"1"/"4" is illegible on its own, but sorting by display name
    instead would silently scramble whatever order the raw keys implied).
    """
    from cellquorum.visualization.figstyle import raincloud

    keys = sorted(groups)
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(keys) + 1), 4))
    for i, key in enumerate(keys):
        color = MUTED_PALETTE[i % len(MUTED_PALETTE)]
        raincloud(ax, np.asarray(groups[key], dtype=float), float(i), color=color, seed=i)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([(labels or {}).get(k, k) for k in keys], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(keys) - 0.3)
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
    "off_diagonal_vmax",
    "coarse_transition_heatmap",
]
