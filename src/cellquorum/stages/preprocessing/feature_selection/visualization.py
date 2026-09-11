"""HVG selection diagnostic: mean expression vs. dispersion/variance.

The three supported flavors (seurat, seurat_v3, pearson_residuals) each write a
DIFFERENT var column for the y-axis metric -- dispersions_norm, variances_norm, and
residual_variances respectively, per scanpy/sc.experimental. A figure hardcoded to one
flavor's column would silently break for the other two, so the column is resolved from
whichever is actually present rather than assumed from the configured method name.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import (
    CELLQUORUM_GRAY,
    CELLQUORUM_RED,
    apply_cellquorum_axis_style,
    save_cellquorum_figure,
)

# Checked in priority order; only one is normally present per run (one flavor per run).
_DISPERSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dispersions_norm", "Normalized dispersion"),
    ("variances_norm", "Normalized variance"),
    ("residual_variances", "Residual variance"),
)


def _resolve_dispersion_column(columns: Iterable[str]) -> tuple[str, str] | None:
    """Return (column, axis label) for whichever dispersion/variance metric is present."""
    present = set(columns)
    for column, label in _DISPERSION_COLUMNS:
        if column in present:
            return column, label
    return None


def write_hvg_figure(
    var: pd.DataFrame,
    output_path: Path,
    *,
    method: str = "seurat_v3",
    dpi: int = 150,
) -> Path | None:
    """Render mean-vs-dispersion with selected HVGs highlighted, or skip cleanly.

    Non-selected genes are drawn first, faint and small; selected genes are drawn on
    top, so the (typically much smaller) selected set is never hidden under the bulk of
    unselected genes the way a single flat scatter would hide it.

    Returns None (writes nothing) when var lacks 'highly_variable' or a recognized
    dispersion/variance column -- both are HVGMethod's own contract, not this figure's
    to reconstruct.
    """
    if "highly_variable" not in var.columns or "means" not in var.columns:
        return None
    resolved = _resolve_dispersion_column(var.columns)
    if resolved is None:
        return None
    y_column, y_label = resolved

    means = pd.to_numeric(var["means"], errors="coerce").to_numpy(dtype=float)
    dispersion = pd.to_numeric(var[y_column], errors="coerce").to_numpy(dtype=float)
    selected = var["highly_variable"].to_numpy(dtype=bool)
    usable = np.isfinite(means) & np.isfinite(dispersion) & (means > 0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    not_selected = usable & ~selected
    ax.scatter(
        means[not_selected],
        dispersion[not_selected],
        s=6,
        alpha=0.35,
        color=CELLQUORUM_GRAY,
        edgecolor="none",
        label=f"Not selected (n={int(not_selected.sum())})",
        zorder=1,
    )
    on = usable & selected
    ax.scatter(
        means[on],
        dispersion[on],
        s=10,
        alpha=0.85,
        color=CELLQUORUM_RED,
        edgecolor="none",
        label=f"Selected HVG (n={int(on.sum())})",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean expression (log)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Highly variable genes ({method})")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    apply_cellquorum_axis_style(ax)

    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


__all__ = ["write_hvg_figure"]
