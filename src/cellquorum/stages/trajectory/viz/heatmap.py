"""Biology-agnostic condition-split pseudotime heatmap primitives.

Ported from the house Fiškin-recipe heatmap: per-condition pseudotime binning +
5-bin smoothing, per-gene 0-1 normalization (done by the caller), peak-bin gene
ordering on the combined profile, and a gridspec layout with stacked annotation
tracks (pseudotime gradient, a continuous score, a categorical state) above one
expression heatmap per condition column. No file I/O, no config, no biology.
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from cellquorum.visualization.figstyle import SEQUENTIAL_CMAP


def bin_masks(pt: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sort order over pt, bin id per cell in that sorted order)."""
    pt = np.asarray(pt, dtype=float)
    order = np.argsort(pt)
    s = pt[order]
    edges = np.linspace(s.min(), s.max(), n_bins + 1)
    binid = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)
    return order, binid


def binned_profile(pt: np.ndarray, mat: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean expression per pseudotime bin, then 5-bin moving-average smoothing."""
    order, binid = bin_masks(pt, n_bins)
    m = np.asarray(mat, dtype=float)[order]
    prof = np.zeros((n_bins, m.shape[1]))
    for b in range(n_bins):
        sel = binid == b
        prof[b] = m[sel].mean(0) if sel.any() else (prof[b - 1] if b else 0)
    k = np.ones(5) / 5
    prof = np.vstack([np.convolve(prof[:, j], k, mode="same") for j in range(prof.shape[1])]).T
    return prof


def binned_tracks(
    pt: np.ndarray, score: np.ndarray, state_codes: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean score and modal state code (forward-filled over empty bins)."""
    order, binid = bin_masks(pt, n_bins)
    sc_o = np.asarray(score, dtype=float)[order]
    st_o = np.asarray(state_codes)[order]
    score_track = np.full(n_bins, np.nan)
    state_track = np.full(n_bins, -1)
    for b in range(n_bins):
        sel = binid == b
        if sel.any():
            score_track[b] = np.nanmean(sc_o[sel])
            vals, cnts = np.unique(st_o[sel], return_counts=True)
            state_track[b] = vals[np.argmax(cnts)]
        elif b:
            score_track[b] = score_track[b - 1]
            state_track[b] = state_track[b - 1]
    for b in range(1, n_bins):
        if np.isnan(score_track[b]):
            score_track[b] = score_track[b - 1]
    return score_track, state_track.astype(int)


def peak_bin_order(combined_profile: np.ndarray) -> np.ndarray:
    """Gene order by peak (argmax) bin of the combined profile."""
    return np.argsort(np.argmax(np.asarray(combined_profile), axis=0))


def condition_split_heatmap(
    profiles: dict,
    tracks: dict,
    gene_labels: list,
    gene_order: np.ndarray,
    *,
    condition_order: list,
    state_cats: list,
    state_colors: list,
    present_state_codes: list,
    expr_cmap: str = SEQUENTIAL_CMAP,
    title: str = "",
) -> Figure:
    """Build the stacked-annotation, condition-split expression heatmap."""
    n_cols = len(condition_order)
    n_bins = next(iter(profiles.values())).shape[0]

    # Which annotation rows are present.
    any_score = any(not np.all(np.isnan(tracks[c][0])) for c in condition_order)
    has_state = bool(state_cats)

    rows = ["pt"]
    if any_score:
        rows.append("score")
    if has_state:
        rows.append("state")
    rows.append("expr")
    n_ann = len(rows) - 1

    hm_h = max(3.0, 0.30 * len(gene_labels))
    fig = Figure(figsize=(5.5 * n_cols, hm_h + 1.0))
    height_ratios = [0.28] * n_ann + [hm_h]
    gs = gridspec.GridSpec(
        len(rows),
        n_cols,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.06,
        left=0.14,
        right=0.88,
        top=0.90,
        bottom=0.06,
    )

    state_cmap = ListedColormap(state_colors) if state_colors else None
    state_norm = (
        BoundaryNorm(np.arange(-0.5, len(state_cats) + 0.5), len(state_cats))
        if state_cats
        else None
    )

    score_all = np.concatenate([tracks[c][0] for c in condition_order]) if any_score else None
    if any_score and np.isfinite(score_all).any():
        s_vmin, s_vmax = np.nanpercentile(score_all[np.isfinite(score_all)], [2, 98])
    else:
        s_vmin, s_vmax = 0.0, 1.0

    pt_grad = np.linspace(0, 1, n_bins)[None, :]
    im_score = None
    im_expr = None
    for j, c in enumerate(condition_order):
        r = 0
        ax_pt = fig.add_subplot(gs[r, j])
        ax_pt.imshow(pt_grad, aspect="auto", cmap="Spectral_r", vmin=0, vmax=1)
        ax_pt.set_xticks([])
        ax_pt.set_yticks([])
        ax_pt.set_title(str(c), fontsize=11, fontweight="bold", pad=6)
        if j == 0:
            ax_pt.set_ylabel("Pseudotime", rotation=0, ha="right", va="center", fontsize=7.5)
        r += 1
        if any_score:
            ax_s = fig.add_subplot(gs[r, j])
            im_score = ax_s.imshow(
                tracks[c][0][None, :], aspect="auto", cmap="Reds", vmin=s_vmin, vmax=s_vmax
            )
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            if j == 0:
                ax_s.set_ylabel("Score", rotation=0, ha="right", va="center", fontsize=7.5)
            r += 1
        if has_state:
            ax_st = fig.add_subplot(gs[r, j])
            ax_st.imshow(tracks[c][1][None, :], aspect="auto", cmap=state_cmap, norm=state_norm)
            ax_st.set_xticks([])
            ax_st.set_yticks([])
            if j == 0:
                ax_st.set_ylabel("State", rotation=0, ha="right", va="center", fontsize=7.5)
            r += 1
        ax = fig.add_subplot(gs[r, j])
        im_expr = ax.imshow(
            profiles[c][:, gene_order].T,
            aspect="auto",
            cmap=expr_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_xlabel("pseudotime", fontsize=8)
        if j == 0:
            ax.set_yticks(range(len(gene_labels)))
            ax.set_yticklabels(
                [gene_labels[i] for i in gene_order], fontsize=6.5, fontstyle="italic"
            )
        else:
            ax.set_yticks([])

    if im_score is not None:
        cax_g = fig.add_axes([0.905, 0.74, 0.014, 0.12])
        cb_g = fig.colorbar(im_score, cax=cax_g, ticks=[s_vmin, s_vmax])
        cb_g.ax.set_yticklabels(["Min", "Max"], fontsize=7)
        cb_g.set_label("Score", fontsize=7.5)
    if im_expr is not None:
        cax = fig.add_axes([0.905, 0.40, 0.014, 0.24])
        fig.colorbar(im_expr, cax=cax, label="scaled expr")
    if has_state and present_state_codes:
        handles = [
            Patch(facecolor=state_colors[i], edgecolor="none", label=state_cats[i])
            for i in present_state_codes
            if 0 <= i < len(state_cats)
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.90, 0.32),
                frameon=False,
                fontsize=6.8,
                title="State",
                title_fontsize=7.5,
                handlelength=1.0,
                handleheight=1.0,
                labelspacing=0.35,
            )
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    return fig


__all__ = [
    "bin_masks",
    "binned_profile",
    "binned_tracks",
    "peak_bin_order",
    "condition_split_heatmap",
]
