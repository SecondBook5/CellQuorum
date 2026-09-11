"""HVG selection diagnostic: mean expression vs. RAW variance, with a fitted trend.

Deliberately plots the RAW (not normalized) variance/dispersion. The normalized columns
(variances_norm / dispersions_norm / residual_variances) have, by construction, already
had the mean-variance trend divided out -- that is what "normalized against the trend"
means -- so plotting them produces a flat band with no curve to show. The classic HVG
diagnostic (scran::modelGeneVar, Seurat's VariableFeaturePlot) shows the trend itself: a
smooth fit through the raw mean-variance relationship, with genes that sit above it (more
variable than their expression level alone predicts) selected.

seurat_v3 and pearson_residuals both write raw LINEAR variance to 'variances'. seurat (v1)
differs in two ways, not one: it writes 'dispersions' instead of 'variances', AND that
column (plus 'means') is already log-transformed internally by scanpy before storage --
confirmed empirically (a 'seurat' run's 'means' correlates at 0.99 with log1p of the raw
per-gene mean, and 'dispersions' is uniformly negative, consistent with log(var/mean) on
compressed values). Applying a further log transform/log-scale axis to an already-log
quantity double-logs it, so the two column families need genuinely different axis
handling, not just different labels.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from cellquorum.visualization.figstyle import (
    CELLQUORUM_RED,
    NORMAL_BLUE,
    TEXT,
    apply_cellquorum_axis_style,
    apply_cellquorum_theme,
    save_cellquorum_figure,
)

# (column, axis label, already log-transformed by scanpy). Checked in priority order.
_RAW_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("dispersions", "Dispersion (log)", True),
    ("variances", "Variance", False),
)


def _resolve_dispersion_column(columns: Iterable[str]) -> tuple[str, str, bool] | None:
    """Return (column, axis label, already_log) for whichever metric is present."""
    present = set(columns)
    for column, label, already_log in _RAW_COLUMNS:
        if column in present:
            return column, label, already_log
    return None


def _fit_trend(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust LOWESS fit of the mean-variance trend, in whatever space x/y already are.

    ``it=3`` robustifying iterations down-weight outliers (genes far above the trend are
    exactly the HVG candidates, and must not be allowed to drag the trend up toward them).
    ``delta`` enables the same linear-interpolation speedup R's lowess uses, since an exact
    fit at every one of tens of thousands of genes is not needed for a smooth reference line.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    span = float(x.max() - x.min())
    fitted = lowess(
        y, x, frac=0.3, it=3, delta=0.01 * span if span > 0 else 0.0, return_sorted=True
    )
    return fitted[:, 0], fitted[:, 1]


def write_hvg_figure(
    var: pd.DataFrame,
    output_path: Path,
    *,
    method: str = "seurat_v3",
    dpi: int = 150,
) -> Path | None:
    """Render mean-vs-variance with a fitted trend and selected HVGs highlighted.

    Non-selected genes draw first, small and faint; selected genes draw on top, so the
    (typically much smaller) selected set is never hidden under the bulk. Returns None
    (writes nothing) when var lacks 'highly_variable' or a recognized raw variance/
    dispersion column -- both are HVGMethod's own contract, not this figure's to
    reconstruct.
    """
    if "highly_variable" not in var.columns or "means" not in var.columns:
        return None
    resolved = _resolve_dispersion_column(var.columns)
    if resolved is None:
        return None
    y_column, y_label, already_log = resolved

    means = pd.to_numeric(var["means"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(var[y_column], errors="coerce").to_numpy(dtype=float)
    selected = var["highly_variable"].to_numpy(dtype=bool)

    if already_log:
        # scanpy already stored log-space quantities; a further log transform or
        # log-scale axis would double-log them.
        usable = np.isfinite(means) & np.isfinite(values)
        x_plot, y_plot = means, values
        x_label = "Mean expression (log1p)"
    else:
        usable = np.isfinite(means) & np.isfinite(values) & (means > 0) & (values > 0)
        x_plot, y_plot = means, values
        x_label = "Mean expression (log)"

    apply_cellquorum_theme()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    fit_x = np.log10(x_plot[usable]) if not already_log else x_plot[usable]
    fit_y = np.log10(y_plot[usable]) if not already_log else y_plot[usable]
    if usable.sum() >= 10:
        trend_x, trend_y = _fit_trend(fit_x, fit_y)
        plot_trend_x = trend_x if already_log else 10.0**trend_x
        plot_trend_y = trend_y if already_log else 10.0**trend_y
        ax.plot(
            plot_trend_x,
            plot_trend_y,
            color=NORMAL_BLUE,
            linewidth=2.0,
            zorder=3,
            label="Fitted trend",
        )

    not_selected = usable & ~selected
    ax.scatter(
        x_plot[not_selected],
        y_plot[not_selected],
        s=6,
        alpha=0.35,
        color=TEXT,
        edgecolor="none",
        label=f"Not selected  (n = {int(not_selected.sum()):,})",
        zorder=1,
        rasterized=True,
    )
    on = usable & selected
    ax.scatter(
        x_plot[on],
        y_plot[on],
        s=11,
        alpha=0.85,
        color=CELLQUORUM_RED,
        edgecolor="none",
        label=f"Selected HVG  (n = {int(on.sum()):,})",
        zorder=2,
    )

    if not already_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Highly Variable Gene Selection", fontweight="bold", pad=10)
    # Reorder so the legend reads trend -> not-selected -> selected, not draw order.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (0 if "trend" in labels[i].lower() else 1, i))
    legend = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        title=method,
        title_fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8.5,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_title().set_style("italic")
    legend.get_title().set_color(TEXT)
    apply_cellquorum_axis_style(ax)

    save_cellquorum_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    return output_path


__all__ = ["write_hvg_figure"]
