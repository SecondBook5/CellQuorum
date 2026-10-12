"""Activity change by group: the donor-aware, depth-adjusted change of each source, as a heatmap.

A per-cell activity score (transcription-factor regulons, pathway footprints) rises with the number
of genes detected in a cell, and a condition that deepens its cells moves every score at once. The
quantity worth drawing is therefore the change between conditions that remains once each donor
pair's depth difference is removed, tested across donor pairs. Computing it is the analysis'
job; this draws a table that carries it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

_DARK = "#202428"
_MUTED = "#8A8A8A"


def plot_activity_change(
    table: pd.DataFrame,
    *,
    groups: Sequence[str],
    value_label: str,
    source_col: str = "source",
    group_col: str = "group",
    delta_col: str = "delta",
    fdr_col: str = "fdr",
    n_col: str = "n_pairs",
    alpha: float = 0.05,
    min_pairs: int = 3,
    max_rows: int = 28,
    title: str = "",
) -> Figure:
    """Sources x groups heatmap of the change in activity, significant cells outlined.

    ``table`` has one row per source and group: the change (``delta_col``), its FDR and the
    number of donor pairs behind it. A group with fewer than ``min_pairs`` donor pairs is
    marked ``n<3`` and not coloured: a change measured on two donors is not a result. Every
    source that is significant in some group is kept whatever ``max_rows`` says; the rest of
    the rows are the largest changes. Columns follow ``groups`` as given.
    """
    needed = {source_col, group_col, delta_col, fdr_col, n_col}
    absent = sorted(needed - set(table.columns))
    if absent:
        raise ValueError("missing column(s): " + ", ".join(absent))
    present = set(table[group_col].astype(str))
    lacking = [g for g in groups if g not in present]
    if lacking:
        raise ValueError("no rows for group(s): " + ", ".join(lacking))

    frame = table[table[group_col].astype(str).isin(groups)].copy()
    frame[group_col] = frame[group_col].astype(str)
    delta = frame.pivot(index=source_col, columns=group_col, values=delta_col).reindex(
        columns=list(groups)
    )
    fdr = frame.pivot(index=source_col, columns=group_col, values=fdr_col).reindex(
        columns=list(groups)
    )
    pairs = frame.pivot(index=source_col, columns=group_col, values=n_col).reindex(
        columns=list(groups)
    )
    enough = pairs >= min_pairs
    delta_ok = delta.where(enough)
    significant = (fdr < alpha) & enough

    keep_sig = list(significant.index[significant.any(axis=1)])
    strongest = delta_ok.abs().max(axis=1).sort_values(ascending=False, kind="mergesort")
    rest = [s for s in strongest.index if s not in set(keep_sig) and np.isfinite(strongest[s])]
    keep = keep_sig + rest[: max(0, max_rows - len(keep_sig))]
    order = list(delta_ok.loc[keep].mean(axis=1).sort_values(ascending=False).index)

    shown = delta_ok.loc[order]
    values = shown.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(np.percentile(finite, 97)) if finite.size else 1.0
    vmax = max(vmax, 1e-9)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(3.6, 1.15 * len(groups) + 3.0), 0.3 * len(order) + 1.6))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", norm=norm, aspect="auto")
    sig_shown = significant.loc[order].to_numpy()
    for i in range(len(order)):
        for j in range(len(groups)):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, f"n<{min_pairs}", ha="center", va="center", fontsize=7, color=_MUTED)
                continue
            strong = abs(value) > 0.65 * vmax
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if sig_shown[i, j] else "normal",
                color="white" if strong else (_DARK if sig_shown[i, j] else "0.45"),
            )
            if sig_shown[i, j]:
                ax.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=_DARK, linewidth=1.4)
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=24)
    cbar: Any = fig.colorbar(image, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(value_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    fig.text(
        0.01,
        0.005,
        f"bold, outlined: FDR < {alpha:g}      n<{min_pairs}: fewer than {min_pairs} donor pairs",
        fontsize=7,
        color="0.35",
        va="bottom",
    )
    return fig


__all__ = ["plot_activity_change"]
