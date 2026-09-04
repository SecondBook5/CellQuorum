"""Activity-along-pseudotime cascade heatmap (biology-agnostic drawing lib).

Given a per-cell activity matrix (cells × sources) and a per-cell pseudotime,
this bins activity along pseudotime, z-scores each source across bins to show
its temporal *shape* (not absolute magnitude), ranks sources by ``|Spearman
rho|`` with pseudotime, and draws a center-of-mass-ordered ``RdBu_r`` heatmap
with signed-rho margin ticks. That is the "cascade" view: rows ordered early →
late by where each program peaks along the trajectory.

No decoupler, no biology here: the caller supplies the already-scored activity
frame and the pseudotime vector; source names are shown verbatim (an optional
``clean_label`` strips only decoupler's ``HALLMARK_`` collection prefix, which
is a resource-naming convention, not a study assumption).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import spearmanr

# Row z-scores are clipped to this symmetric range before drawing, so a single
# extreme bin cannot wash out the diverging colormap.
_ZCLIP = 2.5
_MUTED = "#6b7075"


def rank_by_pseudotime(activity: pd.DataFrame, pseudotime: np.ndarray) -> pd.DataFrame:
    """Rank activity sources by Spearman correlation with pseudotime.

    Args:
        activity: cells × sources activity matrix (rows aligned to ``pseudotime``).
        pseudotime: per-cell pseudotime, same length/order as ``activity``.

    Returns:
        A frame with columns ``name``, ``rho`` (signed Spearman), and ``abs``
        (``|rho|``), sorted by ``abs`` descending. Sources that are constant or
        all-NaN over the finite-pseudotime cells get ``rho = 0``.
    """
    pt = np.asarray(pseudotime, dtype=float)
    finite = np.isfinite(pt)
    rows = []
    for name in activity.columns:
        col = np.asarray(activity[name], dtype=float)
        mask = finite & np.isfinite(col)
        rho = 0.0
        if mask.sum() >= 3 and np.ptp(col[mask]) > 0 and np.ptp(pt[mask]) > 0:
            r, _ = spearmanr(col[mask], pt[mask])
            rho = 0.0 if np.isnan(r) else float(r)
        rows.append({"name": str(name), "rho": rho, "abs": abs(rho)})
    return pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)


def binned_matrix(
    activity: pd.DataFrame, pseudotime: np.ndarray, names: list[str], n_bins: int
) -> pd.DataFrame:
    """Mean activity per source across equal-width pseudotime bins.

    Returns a ``source × bin`` frame (rows reindexed to ``names``); empty bins
    stay NaN. Mirrors ``dc.pp.bin_order`` + per-bin mean without the decoupler
    dependency.
    """
    pt = np.asarray(pseudotime, dtype=float)
    finite = np.isfinite(pt)
    lo, hi = float(pt[finite].min()), float(pt[finite].max())
    if hi <= lo:  # degenerate pseudotime: one bin
        edges = np.array([lo, lo + 1e-9])
        n_bins = 1
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    # Right-closed bins so the max pseudotime lands in the last bin, not one past.
    idx = np.clip(np.digitize(pt, edges[1:-1], right=False), 0, n_bins - 1)
    sub = activity.loc[:, names].loc[finite].copy()
    sub["__bin__"] = idx[finite]
    mat = sub.groupby("__bin__").mean().T
    mat = mat.reindex(columns=range(n_bins))
    return mat.reindex(names)


def center_of_mass_order(mat: pd.DataFrame) -> list[str]:
    """Order rows by where their activity peaks along pseudotime (early → late)."""
    cols = mat.columns.to_numpy(dtype=float)
    com: dict[str, float] = {}
    for name, row in mat.iterrows():
        w = row.to_numpy(dtype=float)
        w = np.clip(w - np.nanmin(w), 0, None)
        total = np.nansum(w)
        com[str(name)] = float(np.nansum(cols * w) / total) if total > 0 else float(cols.mean())
    return sorted((str(n) for n in mat.index), key=lambda n: com[n])


def clean_label(name: str) -> str:
    """Prettify a source name (strip decoupler's ``HALLMARK_`` prefix only)."""
    if name.startswith("HALLMARK_"):
        return name[len("HALLMARK_") :].replace("_", " ").title()
    return name


def cascade_heatmap(
    activity: pd.DataFrame,
    pseudotime: np.ndarray,
    *,
    top: int | None = None,
    n_bins: int = 20,
    title: str = "",
    xlab: str = "",
    cbar_label: str = "activity  (row z-score)",
) -> Figure | None:
    """Draw the activity-along-pseudotime cascade heatmap.

    Ranks sources by ``|Spearman rho|`` with pseudotime, keeps the top ``top``
    (or all when ``None``), bins their activity along pseudotime, z-scores each
    source across bins, orders rows by center-of-mass (early → late), and draws
    an ``RdBu_r`` heatmap with signed-rho direction ticks in the right margin.

    Returns the figure, or ``None`` when no source is associated with
    pseudotime (nothing to draw).
    """
    ranked = rank_by_pseudotime(activity, pseudotime)
    ranked = ranked[ranked["abs"] > 0]
    if top is not None:
        ranked = ranked.head(top)
    names = ranked["name"].tolist()
    if not names:
        return None

    mat = binned_matrix(activity, pseudotime, names, n_bins)
    order = center_of_mass_order(mat)
    mat = mat.reindex(order)

    # z-score each source across bins (temporal shape, not absolute magnitude)
    z = mat.to_numpy(dtype=float)
    mu = np.nanmean(z, axis=1, keepdims=True)
    sd = np.nanstd(z, axis=1, keepdims=True)
    z = np.divide(z - mu, sd, out=np.zeros_like(z), where=sd > 0)
    z = np.clip(np.nan_to_num(z, nan=0.0), -_ZCLIP, _ZCLIP)

    rho = dict(zip(ranked["name"], ranked["rho"], strict=True))
    n = len(order)
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.30 * n + 1.5)))
    im = ax.imshow(
        z,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-_ZCLIP,
        vmax=_ZCLIP,
        extent=(0, 1, 0, n),
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([clean_label(m) for m in order[::-1]], fontsize=7.5)
    # signed-rho direction tick on the right (↑ rises, ↓ falls along pseudotime)
    axr = ax.twinx()
    axr.set_ylim(0, n)
    axr.set_yticks(np.arange(n) + 0.5)
    axr.set_yticklabels(
        [
            ("↑" if rho.get(m, 0.0) >= 0 else "↓") + f" {abs(rho.get(m, 0.0)):.2f}"
            for m in order[::-1]
        ],
        fontsize=6.6,
        color=_MUTED,
    )
    axr.tick_params(length=0)
    ax.set_xlabel(f"pseudotime  ({xlab})" if xlab else "pseudotime →", fontsize=9)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    cb = fig.colorbar(im, ax=axr, fraction=0.025, pad=0.10)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)
    fig.tight_layout()
    return fig


__all__ = [
    "rank_by_pseudotime",
    "binned_matrix",
    "center_of_mass_order",
    "clean_label",
    "cascade_heatmap",
]
