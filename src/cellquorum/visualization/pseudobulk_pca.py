"""Pseudobulk PCA: one point per sample, so the replicate is the sample and not the cell.

A cell-level PCA shows whether cells separate; a pseudobulk PCA shows whether the *samples* do,
which is the claim a donor-aware analysis makes. Drawn with the variance each axis explains in
its label (a plot of two components that explain 4% each says something different from one that
explains 60%) and with the two samples of a donor joined, so a condition effect that is shared
by all donors reads as parallel lines and one that is driven by a single donor does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellquorum.visualization.figstyle import apply_theme

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

_MARKERS = ("o", "^", "s", "D", "v", "P", "X")


def pseudobulk_pca(
    expression: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    condition_col: str,
    group_col: str | None = None,
    pair_col: str | None = None,
    n_top: int = 2000,
    condition_colors: dict[str, str] | None = None,
    group_order: list[str] | None = None,
) -> Figure:
    """PCA of samples on the ``n_top`` most variable genes.

    Args:
        expression: Samples by genes, on a log scale (log CPM).
        meta: One row per sample (same index) carrying the other columns.
        condition_col: Colour of a point.
        group_col: Marker shape of a point (a cell state, a lineage). ``None`` draws one shape.
        pair_col: Samples sharing a value in the same group are joined when both conditions are
            present (the donor). ``None`` draws no joins.
        n_top: Most variable genes the components are computed on.
        condition_colors: Optional colour per condition level; its key order is the order of the
            conditions in the legend (levels it does not name follow, in the order they appear).
        group_order: Optional order of the groups (marker shapes) in the legend.

    Raises:
        ValueError: With fewer than three samples.
    """
    import matplotlib.pyplot as plt

    if len(expression) < 3:
        raise ValueError("a PCA needs at least three samples")
    apply_theme("compact")
    x = expression.to_numpy(dtype=float)
    keep = np.argsort(x.var(axis=0))[::-1][: min(n_top, x.shape[1])]
    centred = x[:, keep] - x[:, keep].mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    scores = u * s
    explained = 100 * s**2 / (s**2).sum()
    meta = meta.loc[expression.index]
    present = list(dict.fromkeys(meta[condition_col].astype(str)))
    conditions = [c for c in (condition_colors or {}) if c in present] + [
        c for c in present if c not in (condition_colors or {})
    ]
    palette = {
        c: (condition_colors or {}).get(c, plt.get_cmap("tab10")(i))
        for i, c in enumerate(conditions)
    }
    seen = list(dict.fromkeys(meta[group_col].astype(str))) if group_col else [""]
    groups = [g for g in (group_order or []) if g in seen] + [
        g for g in seen if g not in (group_order or [])
    ]
    shapes = {g: _MARKERS[i % len(_MARKERS)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    cond = meta[condition_col].astype(str).to_numpy()
    grp = meta[group_col].astype(str).to_numpy() if group_col else np.full(len(meta), "")
    if pair_col is not None:
        pair = meta[pair_col].astype(str).to_numpy()
        for g in groups:
            for p in dict.fromkeys(pair[grp == g]):
                idx = np.flatnonzero((grp == g) & (pair == p))
                if len(idx) == 2 and len({cond[i] for i in idx}) == 2:
                    (line,) = ax.plot(
                        scores[idx, 0], scores[idx, 1], color="#c9ced6", lw=0.8, zorder=1
                    )
                    line.set_gid("pair")
    for g in groups:
        for c in conditions:
            m = (grp == g) & (cond == c)
            if m.any():
                sc = ax.scatter(
                    scores[m, 0],
                    scores[m, 1],
                    s=44,
                    marker=shapes[g],
                    color=palette[c],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
                sc.set_gid("samples")
    ax.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
    for g in groups:
        if g:
            ax.scatter([], [], marker=shapes[g], color="0.45", s=44, label=g)
    for c in conditions:
        ax.scatter([], [], color=palette[c], s=44, label=c)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


__all__ = ["pseudobulk_pca"]
