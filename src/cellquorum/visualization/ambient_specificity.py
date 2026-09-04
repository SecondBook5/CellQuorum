"""Before/after marker-specificity check for ambient-RNA correction.

The standard way to show that ambient correction WORKED, rather than that it ran.
Follows the form DecontX popularised as ``plotDecontXMarkerPercentage``: for each
cell type, the percentage of its cells expressing each lineage's marker set, drawn
before correction and after.

The logic: a marker set should be expressed by its OWN lineage. Ambient mRNA is
free-floating, so it contaminates every droplet roughly uniformly — which shows up
as off-diagonal signal, keratin detected in fibroblasts and haemoglobin detected
in everything. Correction working means the OFF-DIAGONAL shrinks while the
diagonal is preserved. A correction that flattened both would be removing real
biology.

This is why a single contamination fraction per library is not enough on its own:
rho says how much was removed, and only this says whether the right counts were.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from cellquorum.visualization.figstyle import apply_cellquorum_theme

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

_INK = "#1a1a1a"
_MUTED = "#6b6b6b"
# Detection threshold: a cell "expresses" a marker set if any member is observed.
# Deliberately presence/absence rather than a level — ambient contamination shows
# up first as spurious DETECTION, and a mean would be dominated by real
# expressers in the diagonal cell.
_MIN_COUNTS = 1


def marker_detection_matrix(
    counts: np.ndarray,
    genes: Sequence[str],
    cell_types: Sequence[str],
    panel: Mapping[str, Sequence[str]],
    *,
    type_order: list[str] | None = None,
    min_counts: int = _MIN_COUNTS,
) -> pd.DataFrame:
    """Percent of each cell type's cells detecting each marker set.

    Rows are cell types, columns are marker sets, values are percentages. Genes
    absent from ``genes`` are dropped from their set; a set losing all its genes
    yields a NaN column rather than a misleading zero.
    """
    import pandas as pd

    types = np.asarray([str(t) for t in cell_types])
    order = type_order or sorted(set(types))
    gene_index = {str(g): i for i, g in enumerate(genes)}

    rows: dict[str, dict[str, float]] = {}
    for cell_type in order:
        mask = types == cell_type
        rows[cell_type] = {}
        if not mask.any():
            rows[cell_type] = {block: np.nan for block in panel}
            continue
        block_counts = counts[mask]
        for block, block_genes in panel.items():
            columns = [gene_index[str(g)] for g in block_genes if str(g) in gene_index]
            if not columns:
                rows[cell_type][block] = np.nan
                continue
            detected = (block_counts[:, columns] >= min_counts).any(axis=1)
            rows[cell_type][block] = 100.0 * float(detected.mean())
    return pd.DataFrame(rows).T.loc[order, list(panel)]


def off_diagonal_summary(matrix: pd.DataFrame) -> tuple[float, float]:
    """Return (mean off-diagonal %, mean diagonal %) for a specificity matrix.

    The pair is the headline: correction should lower the first and leave the
    second alone. Only cell types that are also marker sets contribute.
    """
    shared = [name for name in matrix.index if name in matrix.columns]
    if not shared:
        return float("nan"), float("nan")
    diagonal, off = [], []
    for row in shared:
        for column in matrix.columns:
            value = matrix.loc[row, column]
            if not np.isfinite(value):
                continue
            (diagonal if row == column else off).append(float(value))
    return (
        float(np.mean(off)) if off else float("nan"),
        float(np.mean(diagonal)) if diagonal else float("nan"),
    )


def specificity_figure(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    title: str | None = None,
    cmap: str = "Purples",
) -> Figure:
    """Before / after / difference matrices for marker specificity.

    Three panels sharing one colour scale for the first two, so a cell that
    darkens or lightens is directly comparable across them. The third is the
    signed change on a diverging scale, where the readable outcome is
    off-diagonal blue (removed) with a diagonal that stays near zero.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    common_rows = [r for r in before.index if r in after.index]
    common_cols = [c for c in before.columns if c in after.columns]
    before = before.loc[common_rows, common_cols]
    after = after.loc[common_rows, common_cols]
    difference = after - before

    vmax = float(np.nanmax([before.to_numpy(), after.to_numpy()]))
    shared_norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    delta_limit = float(np.nanmax(np.abs(difference.to_numpy()))) or 1.0
    delta_norm = mpl.colors.TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    n_rows, n_cols = before.shape
    cell = 0.42
    fig = plt.figure(figsize=(3 * (n_cols * cell + 1.1) + 1.6, n_rows * cell + 2.5))
    grid = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.07],
        wspace=0.18,
        left=0.11,
        right=0.94,
        top=0.84,
        bottom=0.20,
    )

    panels = [
        (before, "before correction", shared_norm, cmap),
        (after, "after correction", shared_norm, cmap),
        (difference, "change (after − before)", delta_norm, "RdBu_r"),
    ]
    axes = []
    for position, (matrix, label, norm, colormap) in enumerate(panels):
        ax = fig.add_subplot(grid[0, position])
        axes.append(ax)
        mesh = ax.imshow(matrix.to_numpy(), aspect="auto", norm=norm, cmap=colormap)
        for i in range(n_rows):
            for j in range(n_cols):
                value = matrix.to_numpy()[i, j]
                if not np.isfinite(value):
                    continue
                rgba = plt.get_cmap(colormap)(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_INK if luminance > 0.6 else "white",
                )
        # Outline the diagonal: the cells that SHOULD stay dark.
        for i, row in enumerate(matrix.index):
            if row in matrix.columns:
                j = list(matrix.columns).index(row)
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#00a03e", lw=1.6, zorder=5
                    )
                )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=7)
        if position == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(matrix.index, fontsize=7.5)
            ax.set_ylabel("cell type", fontsize=8.5)
        else:
            ax.set_yticks([])
        ax.set_title(label, fontsize=9)
        ax.tick_params(length=2)
        if position == 2:
            cax = fig.add_subplot(grid[0, 3])
            fig.colorbar(mesh, cax=cax)
            cax.tick_params(labelsize=6.5)
            cax.set_ylabel("Δ percentage points", fontsize=7)

    off_before, diag_before = off_diagonal_summary(before)
    off_after, diag_after = off_diagonal_summary(after)
    fig.suptitle(
        title or "Ambient correction — marker specificity by cell type",
        fontsize=11.5,
        x=0.11,
        ha="left",
        y=0.95,
    )
    fig.text(
        0.11,
        0.055,
        "% of each cell type's cells detecting each marker set. Green outline = the "
        "diagonal (a lineage's own markers).\n"
        f"OFF-diagonal mean {off_before:.1f}% → {off_after:.1f}% "
        f"({off_after - off_before:+.1f}) · diagonal mean {diag_before:.1f}% → "
        f"{diag_after:.1f}% ({diag_after - diag_before:+.1f}).\n"
        "Correction works when the off-diagonal falls and the diagonal does not.",
        fontsize=7.5,
        color=_MUTED,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    return fig


def apply_theme() -> None:
    apply_cellquorum_theme()


__all__ = [
    "apply_theme",
    "marker_detection_matrix",
    "off_diagonal_summary",
    "specificity_figure",
]
